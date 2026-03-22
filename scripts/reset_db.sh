#!/usr/bin/env bash
# Reset database: drop + create + migrate + seed
set -euo pipefail

DB_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${POSTGRES_DB:-bookstack_rag}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5435}"

echo "Dropping database $DB_NAME..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME"

echo "Creating database $DB_NAME..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"

echo "Running migrations..."
cd "$(dirname "$0")/../backend"
alembic upgrade head

echo "Seeding database..."
python -c "import asyncio; from app.db.seed import run_seeds; asyncio.run(run_seeds())"

echo "Done!"
