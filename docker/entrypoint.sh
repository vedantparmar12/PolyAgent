#!/bin/bash
set -e

# Wait for postgres to be ready
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for PostgreSQL..."
    while ! nc -z postgres 5432; do
        sleep 0.1
    done
    echo "PostgreSQL is ready!"
fi

# Wait for redis to be ready  
if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    while ! nc -z redis 6379; do
        sleep 0.1
    done
    echo "Redis is ready!"
fi

# Run migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m src.db.migrate
fi

# Execute the main command
exec "$@"