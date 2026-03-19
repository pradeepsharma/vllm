# team-test/database/seed_data.py
"""Seed script: inserts sample Users and Projects into the database."""

import hashlib
import sys
from pathlib import Path

# Allow running as a standalone script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team_test.database.connection import SessionLocal, init_db
from team_test.database.models import Project, User


def _fake_hash(password: str) -> str:
    """Deterministic fake hash for seed data (NOT for production use)."""
    return hashlib.sha256(password.encode()).hexdigest()


SEED_USERS = [
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

SEED_PROJECTS = [
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


def seed(db=None) -> None:
    """Insert seed data. Accepts an optional session for testability."""
    close_after = db is None
    if db is None:
        init_db()
        db = SessionLocal()

    try:
        # Skip if data already exists
        if db.query(User).count() > 0:
            print("Seed data already present — skipping.")
            return

        # Insert users
        user_map: dict[str, User] = {}
        for u_data in SEED_USERS:
            user = User(**u_data)
            db.add(user)
            user_map[u_data["username"]] = user

        db.flush()  # Assign IDs without committing

        # Insert projects — copy to avoid mutating the module-level constant
        for p_data in SEED_PROJECTS:
            p_copy = dict(p_data)
            owner_username = p_copy.pop("owner_username")
            project = Project(owner=user_map[owner_username], **p_copy)
            db.add(project)

        db.commit()
        print(f"Seeded {len(SEED_USERS)} users and {len(SEED_PROJECTS)} projects.")

    except Exception:
        db.rollback()
        raise
    finally:
        if close_after:
            db.close()


if __name__ == "__main__":
    seed()
    print("Done.")
