#!/usr/bin/env python
"""
create_user.py — Manage Cortex users in the local SQLite database

Usage:
  python create_user.py add   <username> <password>   # create a new user
  python create_user.py list                           # list all users
  python create_user.py delete <username>              # remove a user
  python create_user.py passwd <username> <newpassword> # change password
"""
import sqlite3, sys
from pathlib import Path
from passlib.context import CryptContext

DB_PATH = Path(__file__).parent / "cortex.db"
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c

def ensure_schema():
    c = get_conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    """)
    c.commit()
    c.close()

def cmd_add(username: str, password: str):
    if len(username.strip()) < 3:
        sys.exit("Error: username must be at least 3 characters")
    if len(password) < 8:
        sys.exit("Error: password must be at least 8 characters")
    ensure_schema()
    c = get_conn()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                  (username.strip(), pwd_ctx.hash(password)))
        c.commit()
        print(f"✓ User '{username.strip()}' created")
    except sqlite3.IntegrityError:
        sys.exit(f"Error: user '{username}' already exists")
    finally:
        c.close()

def cmd_list():
    ensure_schema()
    c = get_conn()
    rows = c.execute("SELECT username, created_at FROM users ORDER BY id").fetchall()
    c.close()
    if not rows:
        print("No users found.")
        return
    print(f"{'Username':<20} {'Created'}")
    print("-" * 40)
    for r in rows:
        print(f"{r['username']:<20} {r['created_at']}")

def cmd_delete(username: str):
    ensure_schema()
    c = get_conn()
    n = c.execute("DELETE FROM users WHERE username=?", (username.strip(),)).rowcount
    c.commit()
    c.close()
    if n:
        print(f"✓ User '{username}' deleted")
    else:
        sys.exit(f"Error: user '{username}' not found")

def cmd_passwd(username: str, new_password: str):
    if len(new_password) < 8:
        sys.exit("Error: password must be at least 8 characters")
    ensure_schema()
    c = get_conn()
    n = c.execute("UPDATE users SET password_hash=? WHERE username=?",
                  (pwd_ctx.hash(new_password), username.strip())).rowcount
    c.commit()
    c.close()
    if n:
        print(f"✓ Password updated for '{username}'")
    else:
        sys.exit(f"Error: user '{username}' not found")

COMMANDS = {"add": cmd_add, "list": cmd_list, "delete": cmd_delete, "passwd": cmd_passwd}
USAGE = {
    "add":    "add <username> <password>",
    "list":   "list",
    "delete": "delete <username>",
    "passwd": "passwd <username> <newpassword>",
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    cmd = args[0]
    params = args[1:]
    expected = {"add": 2, "list": 0, "delete": 1, "passwd": 2}
    if len(params) != expected[cmd]:
        sys.exit(f"Usage: python create_user.py {USAGE[cmd]}")
    COMMANDS[cmd](*params)
