"""Create the first admin account, or reset a password from the command line.

    python -m scripts.create_admin                      # asks for username and password
    python -m scripts.create_admin --username principal --role super
    python -m scripts.create_admin --username ravi --role editor
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.admin.security import audit, hash_password, password_problem  # noqa: E402
from app.db.models import AdminUser  # noqa: E402
from app.db.session import session_scope  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an admin user (or reset an existing user's password)")
    ap.add_argument("--username")
    ap.add_argument("--role", choices=["super", "editor"], default="super")
    args = ap.parse_args()

    username = (args.username or input("Username: ")).strip().lower()
    if not 3 <= len(username) <= 64:
        sys.exit("Usernames must be 3–64 characters.")
    while True:
        password = getpass.getpass("Password (at least 10 characters, hidden): ")
        problem = password_problem(password)
        if problem:
            print(problem)
            continue
        if getpass.getpass("Repeat password: ") != password:
            print("The passwords do not match.")
            continue
        break

    with session_scope() as db:
        user = db.scalar(select(AdminUser).where(AdminUser.username == username))
        if user:
            user.password_hash, user.failed_logins, user.locked_until, user.active = hash_password(password), 0, None, True
            action = "reset"
        else:
            db.add(AdminUser(username=username, role=args.role, password_hash=hash_password(password)))
            action = "created"
    audit("cli", "add_user" if action == "created" else "reset_password", username, args.role)
    print(f"Admin '{username}' {action}. Sign in at http://localhost:8000/admin")


if __name__ == "__main__":
    main()
