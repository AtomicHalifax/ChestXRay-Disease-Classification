.PHONY: install test lint serve docker docker-run up samples test-alert

install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,serve,demo]"

test:
	pytest -q

lint:
	ruff check src tests app.py

serve:
	uvicorn chexpert_cls.api:app --reload --port 8000

docker:
	docker build -t chexpert-api .

docker-run:
	docker run --rm -p 8000:8000 chexpert-api

up:
	docker compose up --build

samples:
	python scripts/fetch_samples.py

test-alert:
	sh scripts/send_test_alert.sh
