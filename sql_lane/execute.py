"""Execute: runs the reviewed SQL against the database (step2-sql-lane-rules.md, B4).

Plain code, no LLM. Uses DATABASE_URL (the project owner's decision), so the protections
the crm_readonly role would give are set on every run instead: a read-only transaction,
a 5-second statement timeout, search_path = analytics and the Brazil time zone. Every run
is rolled back, so nothing can persist.

One connection is shared across runs (opened on first use). If the database dropped it
(Neon closes idle connections), it is reopened once and the query re-run; a second
connection failure is raised.

Outcomes:
  rows               -> columns, rows (at most 1,000, JSON-safe), truncated
  SQL error/timeout  -> one error with stage "execute"; columns/rows cleared (retry can fix it)
  connection failure -> raised (retrying the SQL can't fix it; the graph reports it)
"""
import datetime as dt
import os
import threading
from decimal import Decimal

import psycopg2
import psycopg2.errors

from sql_lane.state import SqlLaneState

MAX_ROWS = 1000
STATEMENT_TIMEOUT = "5s"
TIME_ZONE = "America/Sao_Paulo"
TIMEOUT_MESSAGE = "Query took longer than 5 seconds; simplify it or aggregate earlier."

# Fixed statements (no values from the question or the LLM are ever formatted into SQL).
SESSION_SETUP = (
    f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'",
    "SET LOCAL search_path = analytics",
    f"SET LOCAL TIME ZONE '{TIME_ZONE}'",
)


class ConnectionFailure(Exception):
    """The database can't be reached or the connection broke: not a mistake in the SQL."""


def _is_connection_problem(exc: Exception) -> bool:
    """A broken or unreachable connection has no Postgres error code; a SQL error has one."""
    if isinstance(exc, psycopg2.InterfaceError):          # e.g. "connection already closed"
        return True
    return isinstance(exc, psycopg2.OperationalError) and getattr(exc, "pgcode", None) is None


def json_safe(value):
    """Convert one database value to a type the JSON checkpointer can store."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return str(value)                                     # timedelta, UUID, ... as text


class SharedConnection:
    """One read-only connection reused across runs, reopened when the database drops it.

    A lock lets only one query use it at a time (a psycopg2 connection runs one
    transaction at a time).
    """

    def __init__(self, dsn_env: str = "DATABASE_URL"):
        self._dsn_env = dsn_env
        self._conn = None
        self._lock = threading.Lock()

    def _connect(self):
        try:
            conn = psycopg2.connect(os.environ[self._dsn_env])
        except psycopg2.Error as e:
            raise ConnectionFailure(f"Could not connect to the database: {type(e).__name__}") from e
        conn.set_session(readonly=True, autocommit=False)   # the database refuses any write
        return conn

    def _reset(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except psycopg2.Error:
                pass
        self._conn = None

    def _run_once(self, sql: str) -> tuple[list[str], list[list], bool]:
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        try:
            with self._conn.cursor() as cur:
                for statement in SESSION_SETUP:
                    cur.execute(statement)
                cur.execute(sql)                             # no parameters: % is left as is
                if cur.description is None:                  # not a row-returning statement
                    return [], [], False
                columns = [d.name for d in cur.description]
                fetched = cur.fetchmany(MAX_ROWS + 1)
        finally:
            if not self._conn.closed:
                try:
                    self._conn.rollback()                    # nothing ever persists
                except psycopg2.Error:
                    pass
        truncated = len(fetched) > MAX_ROWS
        rows = [[json_safe(v) for v in row] for row in fetched[:MAX_ROWS]]
        return columns, rows, truncated

    def run(self, sql: str) -> tuple[list[str], list[list], bool]:
        """Run `sql`; returns (columns, rows, truncated).

        Raises psycopg2.Error for a mistake in the SQL (including QueryCanceled on
        timeout) and ConnectionFailure when the database can't be reached, after one
        reconnect-and-retry.
        """
        with self._lock:
            try:
                return self._run_once(sql)
            except psycopg2.Error as e:
                if not _is_connection_problem(e):
                    raise
                self._reset()                                # dropped connection: one retry
            try:
                return self._run_once(sql)
            except psycopg2.Error as e:
                if not _is_connection_problem(e):
                    raise
                self._reset()
                raise ConnectionFailure(f"Lost the database connection: {type(e).__name__}") from e


def make_execute(connection: SharedConnection | None = None):
    """Build the Execute node around one shared connection (created here if not given)."""
    connection = connection or SharedConnection()

    def execute(state: SqlLaneState) -> dict:
        sql = state["sql"][-1]
        try:
            columns, rows, truncated = connection.run(sql)
        except psycopg2.errors.QueryCanceled:                # must come before psycopg2.Error
            return _failure(TIMEOUT_MESSAGE)
        except psycopg2.Error as e:                          # a mistake in the SQL
            message = (getattr(e, "pgerror", None) or str(e)).strip()
            return _failure(message.removeprefix("ERROR:").strip())
        # ConnectionFailure is not caught: it propagates to the graph
        return {"columns": columns, "rows": rows, "truncated": truncated}

    return execute


def _failure(message: str) -> dict:
    """One execute error; the previous attempt's rows are cleared so they can't look current."""
    return {"errors": [{"stage": "execute", "message": message}],
            "columns": [], "rows": [], "truncated": False}
