.DEFAULT_GOAL := help
DATA_URL := https://files.grouplens.org/datasets/movielens/ml-latest.zip

.PHONY: help install download prepare train evaluate pipeline serve lint format test check docker-build docker-run docker-pipeline deploy clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install all dependency groups
	uv sync

download: ## Download MovieLens ml-latest (~335 MB) into data/
	@mkdir -p data
	curl -fL --retry 3 -o data/ml-latest.zip $(DATA_URL)
	unzip -o -q data/ml-latest.zip -d data && rm data/ml-latest.zip

prepare: ## Clean data, build items/genome artifacts
	uv run recommender prepare

train: ## Train the serving ALS model
	uv run recommender train

evaluate: ## Benchmark all models (random + temporal split) -> reports/
	uv run recommender evaluate

pipeline: prepare train evaluate ## Run the full offline pipeline

serve: ## Run the dev server on http://127.0.0.1:5000
	uv run recommender serve --debug

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

check: lint test ## Everything CI runs

docker-build: ## Build the web image (needs artifacts/ from `make pipeline`)
	docker compose build web

docker-run: ## Run the web image on http://localhost:8000
	docker compose up web

docker-pipeline: ## Run the offline pipeline inside Docker
	docker compose run --rm pipeline all

deploy: ## Deploy to the Hugging Face Space (needs `hf auth login` + artifacts)
	uv run python scripts/deploy_space.py

clean: ## Remove caches (keeps data and artifacts)
	rm -rf .pytest_cache .ruff_cache **/__pycache__
