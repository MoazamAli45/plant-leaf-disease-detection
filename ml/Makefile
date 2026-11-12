PY ?= .venv/bin/python

.PHONY: setup data data-public prepare eda test lint format train-paper-local report

setup:            ## create venv and install everything for Phase 1
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[download,notebook,dev,train]"

data:             ## download NPD (Kaggle token required), PlantDoc and the PlantVillage leaf map
	$(PY) -m leafdx.data.download --data-dir data

data-public:      ## only the sources that need no account
	$(PY) -m leafdx.data.download --data-dir data --only plantdoc leafmap

prepare:          ## index, audit, split; writes manifests/ and ../docs/figures/data_audit/
	$(PY) -m leafdx.data.prepare --config configs/data.yaml

eda:              ## figures into ../docs/figures/eda/
	$(PY) -m leafdx.eda --config configs/data.yaml

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

train-paper-local: ## the two steps100 reproduction runs (CPU-feasible, ~30 min each)
	$(PY) -m leafdx.train.run --config configs/paper/cnn_steps100.yaml
	$(PY) -m leafdx.train.run --config configs/paper/mobilenet_steps100.yaml

report:           ## collect outputs/ into results/ and regenerate reproduction tables
	$(PY) -m leafdx.train.report collect outputs
	$(PY) -m leafdx.train.report reproduction
