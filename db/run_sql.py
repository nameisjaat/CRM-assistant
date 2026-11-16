"""Run db/checks.sql or db/transform.sql without psql.

    python db/run_sql.py db/checks.sql
    python db/run_sql.py db/transform.sql

The .sql files stay valid psql scripts. This runner understands the two psql
meta-commands they use: `\\echo` prints a label, `\\set ...` is skipped.

checks.sql
    Each check runs in its own READ ONLY transaction, so it cannot change anything.
    The label and the result rows are printed. A failing check is reported and the
    run continues with the next one; the exit code is 1 if any check failed.

transform.sql
    Everything runs in ONE transaction. At the first error the runner prints the
    failing statement and the database error, rolls back (analytics is left as it
    was) and exits with code 1. On success, the crm_readonly password is set from
    READONLY_PASSWORD (never printed or written to a file), then it commits.

Connection string: DATABASE_URL (from the environment or the project's .env).
"""
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def split_sql(text):
    """Split SQL text into statements on top-level semicolons.

    Ignores semicolons inside 'strings', "identifiers", -- comments,
    /* block comments */ and $tag$ dollar-quoted bodies $tag$.
    """
    statements, buf, i, n = [], [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "-" and text.startswith("--", i):
            end = text.find("\n", i)
            end = n if end == -1 else end
            buf.append(text[i:end]); i = end
        elif ch == "/" and text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            buf.append(text[i:end]); i = end
        elif ch in ("'", '"'):
            j = i + 1
            while j < n:
                if text[j] == ch:
                    if j + 1 < n and text[j + 1] == ch:  # doubled quote = escaped
                        j += 2
                        continue
                    break
                j += 1
            buf.append(text[i:j + 1]); i = j + 1
        elif ch == "$":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if j < n and text[j] == "$":
                tag = text[i:j + 1]
                end = text.find(tag, j + 1)
                end = n if end == -1 else end + len(tag)
                buf.append(text[i:end]); i = end
            else:
                buf.append(ch); i += 1
        elif ch == ";":
            statements.append("".join(buf).strip()); buf = []; i += 1
        else:
            buf.append(ch); i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return [s for s in statements if strip_comments(s)]


def strip_comments(stmt):
    return "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--")).strip()


def parse_script(path):
    """Return a list of ("echo", label) and ("sql", statement) items in file order."""
    items, chunk = [], []

    def flush():
        if chunk:
            items.extend(("sql", s) for s in split_sql("\n".join(chunk)))
            chunk.clear()

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("\\echo"):
            flush()
            items.append(("echo", stripped[len("\\echo"):].strip().strip("'")))
        elif stripped.startswith("\\"):
            flush()  # \set ON_ERROR_STOP etc.: psql-only, the runner already stops on error
        else:
            chunk.append(line)
    flush()
    return items


def print_result(cur):
    if cur.description is None:
        return
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    widths = [max(len(str(c)), *(len(str(r[k])) for r in rows)) if rows else len(str(c))
              for k, c in enumerate(cols)]
    print("  " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(("NULL" if v is None else str(v)).ljust(w) for v, w in zip(r, widths)))
    print(f"  ({len(rows)} row{'s' if len(rows) != 1 else ''})")


def run_checks(conn, items):
    failed = 0
    for kind, value in items:
        if kind == "echo":
            print(f"\n== {value}")
            continue
        conn.set_session(readonly=True, autocommit=False)
        try:
            with conn.cursor() as cur:
                cur.execute(value)
                print_result(cur)
        except psycopg2.Error as e:
            failed += 1
            print(f"  ERROR: {str(e).strip()}")
        finally:
            conn.rollback()
    return 1 if failed else 0


def run_transform(conn, items):
    password = os.environ.get("READONLY_PASSWORD")
    if not password:
        print("READONLY_PASSWORD is not set; it is needed for the crm_readonly role.")
        return 1
    statements = [v for k, v in items if k == "sql"]
    with conn.cursor() as cur:
        for n, stmt in enumerate(statements, 1):
            try:
                cur.execute(stmt)
            except psycopg2.Error as e:
                conn.rollback()
                print(f"FAILED at statement {n} of {len(statements)} (rolled back, nothing changed):\n")
                print(stmt)
                print(f"\nERROR: {str(e).strip()}")
                return 1
        cur.execute("ALTER ROLE crm_readonly PASSWORD %s", (password,))
    conn.commit()
    print(f"OK: {len(statements)} statements committed; crm_readonly password set from READONLY_PASSWORD.")
    return 0


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    load_dotenv(ROOT / ".env")
    items = parse_script(path)
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        if path.name == "checks.sql":
            code = run_checks(conn, items)
        elif path.name == "transform.sql":
            code = run_transform(conn, items)
        else:
            sys.exit("Expected db/checks.sql or db/transform.sql")
    finally:
        conn.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
