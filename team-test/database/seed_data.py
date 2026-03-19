"""Seed script: inserts sample Users and Projects into the database."""

import hashlib
import sys
from pathlib import Path

# Allow running this script directly from the team-test directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team_test.database.connection import SessionLocal, init_db
from team_test.database.models import Project, User


def _fake_hash(password: str) -> str:
    """Produce a deterministic SHA-256 hex digest of *password*.

    This is intentionally **not** a production-grade password hash — it is
    only used to populate seed data with a recognisable, reproducible value.
    Real endpoints use ``passlib`` (bcrypt) to hash passwords.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


SEED_USERS: list[dict] = [
    {
        "username": "alice",
        "email": "alice@example.com",
        "full_name": "Alice Wonderland",
        "hashed_password": _fake_hash("alice_secret"),
        "is_active": True,
    },
    {
        "username": "bob",
        "email": "bob@example.com",
        "full_name": "Bob Builder",
        "hashed_password": _fake_hash("bob_secret"),
        "is_active": True,
    },
    {
        "username": "charlie",
        "email": "charlie@example.com",
        "full_name": "Charlie Chaplin",
        "hashed_password": _fake_hash("charlie_secret"),
        "is_active": False,
    },
]

SEED_PROJECTS: list[dict] = [
    {
        "name": "Alpha Initiative",
        "description": "First project for Alice",
        "owner_username": "alice",
        "is_active": True,
    },
    {
        "name": "Beta Platform",
        "description": "Alice's second project",
        "owner_username": "alice",
        "is_active": True,
    },
    {
        "name": "Bob's Workshop",
        "description": "Bob's primary project",
        "owner_username": "bob",
        "is_active": True,
    },
    {
        "name": "Legacy System",
        "description": "Archived project by Charlie",
        "owner_username": "charlie",
        "is_active": False,
    },
]


def seed(clear: bool = False) -> None:
    """Populate the database with sample users and projects.

    Parameters
    ----------
    clear:
        When ``True``, delete all existing rows before inserting seed data.
        Defaults to ``False`` so that repeated runs are idempotent (rows
        whose ``username`` / ``name`` already exist are skipped).
    """
    init_db()
    db = SessionLocal()
    try:
        if clear:
            db.query(Project).delete()
            db.query(User).delete()
            db.commit()

        # --- Users ---
        username_to_user: dict[str, User] = {}
        for user_data in SEED_USERS:
            existing = db.query(User).filter_by(username=user_data["username"]).first()
            if existing:
                username_to_user[user_data["username"]] = existing
                continue
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                full_name=user_data["full_name"],
                hashed_password=user_data["hashed_password"],
                is_active=user_data["is_active"],
            )
            db.add(user)
            db.flush()  # populate user.id before referencing it
            username_to_user[user_data["username"]] = user

        db.commit()

        # Refresh to pick up server-assigned timestamps
        for user in username_to_user.values():
            db.refresh(user)

        # --- Projects ---
        for proj_data in SEED_PROJECTS:
            owner = username_to_user.get(proj_data["owner_username"])
            if owner is None:
                print(
                    f"[seed] WARNING: owner '{proj_data['owner_username']}' not found; "
                    f"skipping project '{proj_data['name']}'.",
                    file=sys.stderr,
                )
                continue

            existing = (
                db.query(Project)
                .filter_by(name=proj_data["name"], owner_id=owner.id)
                .first()
            )
            if existing:
                continue

            project = Project(
                name=proj_data["name"],
                description=proj_data["description"],
                is_active=proj_data["is_active"],
                owner_id=owner.id,
            )
            db.add(project)

        db.commit()
        print("[seed] Database seeded successfully.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the team-test database.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing rows before inserting seed data.",
    )
    args = parser.parse_args()
    seed(clear=args.clear)
