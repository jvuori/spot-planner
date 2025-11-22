#!/bin/bash
set -e

echo "Installing dependencies..."
uv sync --dev

echo "Building Rust extension in development mode..."
source .venv/bin/activate
maturin develop --release

echo "Running tests..."
uv run pytest tests/ -v

echo "Testing import..."
uv run python -c "from spot_planner import get_cheapest_periods; print('Import successful')"

echo "All tests passed!"


