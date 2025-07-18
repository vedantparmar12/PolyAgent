# Enhanced Agentic Workflow - Makefile

.PHONY: help build up down logs shell test lint format clean

# Default target
help:
	@echo "Enhanced Agentic Workflow - Available commands:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs"
	@echo "  make shell       - Open shell in API container"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean up containers and volumes"
	@echo ""
	@echo "Development commands:"
	@echo "  make dev         - Start in development mode"
	@echo "  make dev-build   - Build development images"

# Production commands
build:
	docker-compose -f docker/docker-compose.yml build

up:
	docker-compose -f docker/docker-compose.yml up -d

down:
	docker-compose -f docker/docker-compose.yml down

logs:
	docker-compose -f docker/docker-compose.yml logs -f

shell:
	docker-compose -f docker/docker-compose.yml exec api /bin/bash

# Development commands
dev-build:
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml build

dev:
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

dev-logs:
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f

# Testing and quality
test:
	docker-compose -f docker/docker-compose.yml exec api pytest tests/ -v

test-cov:
	docker-compose -f docker/docker-compose.yml exec api pytest tests/ --cov=src --cov-report=html

lint:
	docker-compose -f docker/docker-compose.yml exec api ruff check src/
	docker-compose -f docker/docker-compose.yml exec api mypy src/

format:
	docker-compose -f docker/docker-compose.yml exec api black src/
	docker-compose -f docker/docker-compose.yml exec api isort src/

# Database operations
db-migrate:
	docker-compose -f docker/docker-compose.yml exec api python -m src.db.migrate

db-shell:
	docker-compose -f docker/docker-compose.yml exec postgres psql -U agent -d agentic

# Cleanup
clean:
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f

clean-all:
	docker-compose -f docker/docker-compose.yml down -v --rmi all
	docker system prune -af

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "PgAdmin: http://localhost:5050 (admin@example.com/admin)"
	@echo "Redis Commander: http://localhost:8081"

# Quick start
quickstart: build up
	@echo "Waiting for services to start..."
	@sleep 10
	@echo ""
	@echo "Enhanced Agentic Workflow is ready!"
	@echo ""
	@echo "Access points:"
	@echo "  - API: http://localhost:8000"
	@echo "  - UI: http://localhost:8501"
	@echo "  - MCP: http://localhost:8765"
	@echo ""
	@echo "Run 'make monitor' to open monitoring dashboards"
	@echo "Run 'make logs' to view logs"