.PHONY: install data train serve test lint

install:
	pip install -e ".[dev]"

data:
	python src/data/synthetic_generator.py

train:
	python src/models/train.py

serve:
	uvicorn src.serving.app:app --reload --port 8000

test:
	pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=60

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports
