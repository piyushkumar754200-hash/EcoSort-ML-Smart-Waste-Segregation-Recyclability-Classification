.PHONY: setup data train eval export dev-backend dev-frontend test lint clean-demo reset-db docker-up

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

setup:
	uv venv --python 3.11 --seed $(VENV)
	uv pip install --no-cache --index-strategy unsafe-best-match --python $(PYTHON) -r backend/requirements.txt -r ml/requirements.txt
	npm install

data:
	$(PYTHON) ml/scripts/download_data.py
	$(PYTHON) ml/scripts/verify_dataset.py
	$(PYTHON) ml/scripts/prepare_dataset.py

train:
	$(PYTHON) ml/scripts/train.py --config ml/configs/efficientnet_b0.yaml
	$(PYTHON) ml/scripts/train.py --config ml/configs/mobilenet_v3_large.yaml

eval:
	$(PYTHON) ml/scripts/evaluate.py

export:
	$(PYTHON) ml/scripts/export_onnx.py

dev-backend:
	PYTHONPATH=backend $(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	npm run dev

test:
	PYTHONPATH=backend $(PYTEST) backend/tests -q --cov=app
	npm run test -- --run
	npm run lint
	npm run build

lint:
	$(RUFF) check backend ml
	npm run lint

clean-demo:
	$(PYTHON) scripts/manage_db.py --remove-demo

reset-db:
	$(PYTHON) scripts/manage_db.py --reset

docker-up:
	docker compose build && docker compose up -d && curl -f http://localhost:8000/api/health
