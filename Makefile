.PHONY: run serve install clean lint test test-dbt warehouse pipeline validate

# Gold layer has two schema-identical representations:
#   data/gold/ml_features.parquet  — flat parquet for FastAPI runtime (built by CD)
#   warehouse/co2.duckdb mart_ml_features — dbt table for analytics/BI
# See schemas/gold_ml_features.yaml for the canonical schema.

run:
	streamlit run frontend/app.py

serve:
	uvicorn api.main:app --reload --port 8000

install:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

test:
	pytest tests/ --cov=src --cov=api

pipeline:
	python flows/co2_pipeline.py

validate:
	python -m tests.data.ge_validation

warehouse:
	mkdir -p data/warehouse
	cd warehouse/co2_warehouse && dbt deps --profiles-dir ../.. && dbt run --profiles-dir ../..

test-dbt:
	cd warehouse/co2_warehouse && dbt test --profiles-dir ../..
