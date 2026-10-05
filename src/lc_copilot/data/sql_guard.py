"""Safe Talk2Data: every model-generated SQL statement passes this AST-level guard before it runs.

Defence in depth — the guard is one layer of four:
  1. AST policy here (single SELECT, table allow-list, column deny-list, function deny-list, row cap)
  2. read-only database connection (``mode=ro`` + ``PRAGMA query_only``)
  3. per-query wall-clock timeout (SQLite progress handler)
  4. least-privilege tool scopes on the MCP server
Regex-based "does it start with SELECT" checks are trivially bypassed; parsing is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from ..resilience import PermanentError

ALLOWED_TABLES: dict[str, set[str]] = {
    "counterparties": {"counterparty_id", "name", "country", "role", "risk_rating"},
    "letters_of_credit": {
        "lc_number", "applicant_id", "beneficiary_id", "currency", "amount", "issue_date",
        "expiry_date", "latest_shipment", "status", "rules",
    },
    "presentations": {"presentation_id", "lc_number", "presented_on", "amount", "channel", "status"},
    "discrepancies": {"discrepancy_id", "presentation_id", "code", "description", "waived"},
}
DENIED_COLUMNS = {"tax_id", "contact_email"}  # PII never leaves the database through the agent
DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "sqlite_offset"}
_FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command",
              "Pragma", "Attach", "Detach", "Merge", "TruncateTable", "Transaction", "Commit", "Set")
    if hasattr(exp, n)
)


class SQLPolicyViolation(PermanentError):
    pass


@dataclass
class GuardedSQL:
    sql: str
    tables: list[str]
    notes: list[str] = field(default_factory=list)


def schema_description() -> str:
    """Schema the model is allowed to see — denied columns are not even described."""
    return "\n".join(f"{t}({', '.join(sorted(cols))})" for t, cols in ALLOWED_TABLES.items())


def guard_sql(sql: str, max_rows: int = 200) -> GuardedSQL:
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLPolicyViolation(f"unparseable SQL: {e.errors[0]['description'] if e.errors else e}") from e
    if len(statements) != 1:
        raise SQLPolicyViolation("exactly one statement is allowed")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLPolicyViolation(f"only SELECT queries are allowed (got {type(tree).__name__})")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLPolicyViolation(f"forbidden operation: {type(node).__name__}")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = sorted({t.name.lower() for t in tree.find_all(exp.Table)} - cte_names)
    for t in tables:
        if t not in ALLOWED_TABLES:
            raise SQLPolicyViolation(f"table not allowed: {t}")

    for col in tree.find_all(exp.Column):
        if col.name.lower() in DENIED_COLUMNS:
            raise SQLPolicyViolation(f"column not allowed: {col.name}")
    if any(isinstance(s, exp.Star) for s in tree.find_all(exp.Star)) and "counterparties" in tables:
        raise SQLPolicyViolation("SELECT * on counterparties is not allowed; list columns explicitly")

    for fn in tree.find_all(exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if name in DENIED_FUNCTIONS:
            raise SQLPolicyViolation(f"function not allowed: {name}")

    notes = []
    if isinstance(tree, exp.Select):
        limit = tree.args.get("limit")
        current = None
        if limit is not None:
            try:
                current = int(limit.expression.name)
            except (AttributeError, ValueError):
                current = None
        if current is None or current > max_rows:
            tree = tree.limit(max_rows)
            notes.append(f"row cap applied: LIMIT {max_rows}")
    else:
        tree = exp.select("*").from_(tree.subquery("u")).limit(max_rows)
        notes.append(f"row cap applied: LIMIT {max_rows}")

    return GuardedSQL(tree.sql(dialect="sqlite"), tables, notes)
