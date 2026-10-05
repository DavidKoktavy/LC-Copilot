import pytest

from lc_copilot.data.sql_guard import SQLPolicyViolation, guard_sql, schema_description
from tests.conftest import run


@pytest.mark.parametrize("sql", [
    "SELECT lc_number, amount FROM letters_of_credit WHERE status = 'active'",
    "SELECT status, COUNT(*) AS n FROM presentations GROUP BY status",
    "WITH d AS (SELECT presentation_id FROM discrepancies WHERE waived = 0) SELECT COUNT(*) FROM d",
    "SELECT c.name, l.amount FROM letters_of_credit l JOIN counterparties c ON c.counterparty_id = l.beneficiary_id",
])
def test_allows_read_only_queries(sql):
    assert guard_sql(sql).sql.upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize("sql,reason", [
    ("DELETE FROM presentations", "only SELECT"),
    ("UPDATE letters_of_credit SET status = 'cancelled'", "only SELECT"),
    ("DROP TABLE discrepancies", "only SELECT"),
    ("SELECT 1; DROP TABLE presentations", "exactly one statement"),
    ("SELECT name, tax_id FROM counterparties", "column not allowed: tax_id"),
    ("SELECT name FROM counterparties WHERE contact_email LIKE '%@%'", "column not allowed: contact_email"),
    ("SELECT (SELECT tax_id FROM counterparties LIMIT 1) FROM presentations", "column not allowed: tax_id"),
    ("SELECT * FROM counterparties", "SELECT *"),
    ("SELECT * FROM (SELECT * FROM counterparties)", "SELECT *"),
    ("SELECT name FROM sqlite_master", "table not allowed"),
    ("SELECT load_extension('evil.so')", "function not allowed"),
    ("PRAGMA table_info(counterparties)", "only SELECT"),
    ("ATTACH DATABASE 'x.db' AS x", "only SELECT"),
    ("SELEC lc_number FROM", "unparseable|only SELECT"),
])
def test_blocks_unsafe_queries(sql, reason):
    with pytest.raises(SQLPolicyViolation, match=reason):
        guard_sql(sql)


def test_row_cap_is_enforced():
    assert guard_sql("SELECT lc_number FROM letters_of_credit LIMIT 100000", max_rows=50).sql.endswith("LIMIT 50")
    assert guard_sql("SELECT lc_number FROM letters_of_credit LIMIT 3", max_rows=50).sql.endswith("LIMIT 3")


def test_schema_shown_to_model_hides_pii():
    s = schema_description()
    assert "tax_id" not in s and "contact_email" not in s and "letters_of_credit" in s


def test_database_connection_is_read_only(svc):
    # even if the guard were bypassed, the connection itself refuses writes
    import sqlite3

    with pytest.raises((sqlite3.OperationalError, Exception)):
        svc.db._run("DELETE FROM presentations")
    rows = run(svc.db.query("SELECT COUNT(*) AS n FROM presentations")).rows
    assert rows == [{"n": 6}]
