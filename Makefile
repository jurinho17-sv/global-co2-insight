.PHONY: run install clean lint test

run:
	streamlit run src/app.py

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

test:
	pytest tests/ -v
