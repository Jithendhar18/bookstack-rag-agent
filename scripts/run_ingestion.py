#!/usr/bin/env python3
"""Script to run a one-off ingestion from BookStack."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db.session import AsyncSessionLocal, init_db
from app.db.seed import run_seeds
from app.ingestion.pipeline import IngestionPipeline


async def main():
    print("Initializing database...")
    await init_db()
    await run_seeds()

    print("Starting ingestion...")
    async with AsyncSessionLocal() as db:
        pipeline = IngestionPipeline(db)
        stats = await pipeline.ingest_pages(tenant_id="default")
        print(f"Ingestion complete: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
