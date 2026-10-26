# EcoSort ML — Smart Waste Segregation & Recyclability Classification System (T237)

EcoSort ML is a full-stack, production-oriented AI waste classification and recyclability guidance platform. It classifies waste item photographs into **5 canonical material categories** (`plastic`, `paper`, `metal`, `glass`, `organic`) with a calibrated **confidence-gated `"uncertain"` outcome**, maps detections to a startup-validated recyclability taxonomy (`backend/app/core/taxonomy.yaml`), persists every prediction in SQLite/PostgreSQL via SQLAlchemy 2.x + Alembic, and serves an interactive React + TypeScript + Tailwind CSS + Recharts analytics interface.

---

## 1. Key Capabilities & Real Measured Metrics

- **Real Multi-Source Dataset (`5,222` Usable Deduplicated Images):**
  - Combines **UCI RealWaste** (`CC BY 4.0`) and **TrashNet** (`MIT`), verified with Pillow decode checks and `8x8` perceptual hash (`dHash`/`aHash`) deduplication (`8` duplicates removed, `0` cross-split hash leakage).
  - Stratified `70 / 15 / 15` split (`seed = 42`): `glass` (`870`), `metal` (`858`), `organic` (`847`), `paper` (`1,749`), `plastic` (`898`).
- **Dual-Backbone Transfer Learning & ONNX Runtime CPU Serving:**
  - Trained both **`EfficientNet-B0`** (`val_macro_f1 = 0.8570`) and **`MobileNetV3-Large`** (`val_macro_f1 = 0.8654`).
  - Held-out test set (`784` images): **Accuracy `86.48%`**, **Top-3 Accuracy `98.72%`**, **Macro-F1 `0.8586`** (`>= 0.85` target met), **Macro ROC-AUC `0.9781`**, **ECE `0.0474`**.
  - Exported to `artifacts/models/model.onnx` (`opset 17`) with verified PyTorch vs. ONNX softmax parity across 20 test images (`max_abs_diff = 1.79e-06 < 1e-4`).
- **Production FastAPI + SQLAlchemy 2.x + Alembic Backend:**
  - Endpoints: `/api/health`, `/api/model/info`, `/api/predict`, `/api/predict/batch`, `/api/predictions`, `/api/predictions/{id}`, `/api/predictions/{id}/image`, `/api/predictions/{id}/feedback`, `/api/analytics/summary`, `/api/export/csv`.
  - Security: Magic-byte verification (`JPEG`, `PNG`, `WebP`), Pillow pixel decode + EXIF stripping, upload size limit (`413`), unsupported media rejection (`415`), corrupt image rejection (`422`), `503 MODEL_NOT_LOADED` safeguard when weights are absent, and CSV formula injection sanitization (`'`).

---

## 2. Quickstart Commands

### Environment Setup
```bash
# Create Python 3.11 virtual environment and install backend + ML dependencies
uv venv --python 3.11 --seed .venv
uv pip install --no-cache --index-strategy unsafe-best-match --python .venv/bin/python \
  -r backend/requirements.txt -r ml/requirements.txt

# Install frontend dependencies
npm install --legacy-peer-deps
```

### ML Pipeline (Download -> Verify -> Split -> Train -> Evaluate -> Export ONNX -> CLI)
```bash
.venv/bin/python ml/scripts/download_data.py
.venv/bin/python ml/scripts/verify_dataset.py
.venv/bin/python ml/scripts/prepare_dataset.py
.venv/bin/python ml/scripts/train.py --config ml/configs/efficientnet_b0.yaml
.venv/bin/python ml/scripts/train.py --config ml/configs/mobilenet_v3_large.yaml
.venv/bin/python ml/scripts/evaluate.py
.venv/bin/python ml/scripts/export_onnx.py

# Run CLI inference on a single image or directory
.venv/bin/python ml/scripts/predict_cli.py ml/data/raw/realwaste/Glass/Glass_1.jpg --json
```

### Database Migration & Real Inference Seeding
```bash
cd backend && ../.venv/bin/alembic upgrade head && cd ..
# Seed 15 real held-out test set predictions via ONNX Runtime (is_demo=False)
.venv/bin/python scripts/manage_db.py --seed-real 3
# Or seed/purge labeled demo rows (is_demo=True)
.venv/bin/python scripts/manage_db.py --seed-demo 2
.venv/bin/python scripts/manage_db.py --purge-demo
```

### Running the Application
```bash
# Start full-stack dev server (FastAPI on 127.0.0.1:8000 + Vite/Express proxy on 0.0.0.0:3000)
npm run dev
```

### Testing & Linting
```bash
# Backend + ML tests with coverage (13 tests, 87% coverage)
PYTHONPATH=backend .venv/bin/pytest backend/tests -v --cov=app

# Frontend unit & component tests (Vitest + React Testing Library)
npm run test -- --run

# Linters & typecheck
.venv/bin/ruff check backend/app backend/tests ml/scripts scripts
npm run lint
npm run build
```

### Docker Compose (Multi-Stage Production Containers)
```bash
docker compose up --build -d
# Optional PostgreSQL profile:
docker compose --profile postgres up --build -d
```

---

## 3. Documentation Index
- [`docs/WORKSPACE_AUDIT.md`](docs/WORKSPACE_AUDIT.md) — Hardware, OS, Python/Node, and container audit.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture and ML pipeline dataflow.
- [`docs/DATASETS.md`](docs/DATASETS.md) — UCI RealWaste + TrashNet sources, licenses, and label mapping.
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — Held-out test metrics, confusion matrix, calibration, and limitations.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — Architectural and ML engineering decisions.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — Root-cause analysis and fixes for runtime issues.
- [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) — Phase-by-phase verification table and deliverable audit.
