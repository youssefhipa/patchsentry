.PHONY: help install data test lint sweep benchmark bundle verify bench loadtest figures report all serve

# On the author's iCloud-synced Mac, set UV_PROJECT_ENVIRONMENT outside iCloud and PYTHON to match.
PYTHON ?= .venv/bin/python

help: ## Show this help.
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "%-12s %s\n", $$1, $$2}'

install: ## Install all dependencies from the lock file.
	uv sync --extra train --extra dev

data: ## Download and extract the VisA dataset (1.9 GB).
	$(PYTHON) -m patchsentry.data.download --root data/raw/visa

test: ## Run both test groups, separately: FAISS and PyTorch cannot share a macOS process.
	$(PYTHON) -m pytest tests/training -q
	$(PYTHON) -m pytest tests/serving -q

lint: ## Lint, check formatting and type-check.
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy

sweep: ## Tune hyperparameters on the development categories only.
	$(PYTHON) -m patchsentry.pipeline.sweep

benchmark: ## Compare PatchCore with the baselines on all categories.
	$(PYTHON) -m patchsentry.pipeline.benchmark

bundle: ## Export ONNX, fit memory banks, calibrate thresholds.
	$(PYTHON) -m patchsentry.pipeline.build_bundle

verify: ## Evaluate the deployed path (ONNX Runtime and FAISS) on the test set.
	$(PYTHON) -m patchsentry.pipeline.verify_serving

bench: ## Measure backbone and index latency.
	$(PYTHON) -m patchsentry.pipeline.bench_inference --part backbone
	$(PYTHON) -m patchsentry.pipeline.bench_inference --part index

loadtest: ## Compare micro-batching with no batching over HTTP.
	$(PYTHON) -m patchsentry.pipeline.loadtest

figures: ## Draw the README figures from the reports.
	$(PYTHON) -m patchsentry.pipeline.figures

report: ## Write the result tables in README.md from the reports.
	$(PYTHON) -m patchsentry.pipeline.report

all: data sweep benchmark bundle verify bench loadtest figures report ## Reproduce every result.

serve: ## Run the API locally on port 8000.
	$(PYTHON) -m patchsentry.serving
