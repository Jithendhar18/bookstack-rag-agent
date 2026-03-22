#!/usr/bin/env bash
# Reset database + vector store: drop + create + migrate + seed
set -euo pipefail

DB_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${POSTGRES_DB:-bookstack_rag}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5435}"

QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION_NAME:-bookstack_documents}"

echo "Terminating all connections to $DB_NAME..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '$DB_NAME'
AND pid <> pg_backend_pid();" 2>/dev/null || true

sleep 1

echo "Dropping database $DB_NAME..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME"

echo "Creating database $DB_NAME..."
PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"

echo "Resetting Qdrant collection '$QDRANT_COLLECTION'..."
curl -s -X DELETE "http://$QDRANT_HOST:$QDRANT_PORT/collections/$QDRANT_COLLECTION" > /dev/null 2>&1 || echo "  (collection may not exist yet)"

echo "Running migrations..."
cd "$(dirname "$0")/../backend"
source venv/bin/activate
PYTHONPATH=. alembic upgrade head

echo "Seeding database..."
PYTHONPATH=. python -c "import asyncio; from app.db.seed import run_seeds; asyncio.run(run_seeds())"

echo "✓ Database and Qdrant reset complete!"
