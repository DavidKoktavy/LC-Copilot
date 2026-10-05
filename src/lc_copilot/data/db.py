"""Read-only database access with a hard per-query timeout.

SQLite keeps the demo self-contained; the same contract (guarded SQL in, rows out, timeout,
read-only credentials) maps directly to Azure SQL / Postgres with a read-only role.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..observability import span
from ..resilience import PermanentError, TransientError
from .sql_guard import guard_sql


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[dict]
    notes: list[str]

    def as_dict(self) -> dict:
        return {"sql": self.sql, "columns": self.columns, "rows": self.rows, "notes": self.notes}


def ensure_db(db_path: Path, seed_sql: Path) -> None:
    if db_path.exists():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(seed_sql.read_text(encoding="utf-8"))
        con.commit()
    finally:
        con.close()


class ReadOnlyDB:
    def __init__(self, db_path: Path, timeout_s: float = 2.0, max_rows: int = 200) -> None:
        self.db_path = db_path
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def _run(self, sql: str) -> tuple[list[str], list[tuple]]:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        deadline = time.monotonic() + self.timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            con.execute("PRAGMA query_only = ON")
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            return cols, cur.fetchmany(self.max_rows)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise TransientError(f"query exceeded {self.timeout_s}s timeout") from e
            raise PermanentError(f"query failed: {e}") from e
        finally:
            con.close()

    async def query(self, sql: str) -> QueryResult:
        guarded = guard_sql(sql, self.max_rows)
        with span("tool.sql", sql=guarded.sql, tables=guarded.tables) as sp:
            cols, rows = await asyncio.to_thread(self._run, guarded.sql)
            sp.set(rows=len(rows))
        return QueryResult(guarded.sql, cols, [dict(zip(cols, r)) for r in rows], guarded.notes)
