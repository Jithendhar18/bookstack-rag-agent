#!/usr/bin/env python3
"""Script to seed the database with initial roles, permissions, and admin user."""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db.seed import run_seeds


async def main():
    print("Seeding database...")
    await run_seeds()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
