# Architecture & System Design (`docs/ARCHITECTURE.md`)

## 1. High-Level System Architecture

```
+-----------------------------------------------------------------------------------+
|                        Client Browser (React 19 + TypeScript)                     |
|  +-------------+  +-------------+  +-------------+  +-------------+  +---------+  |
|  |  Classify   |  |   History   |  |  Analytics  |  |    Model    |  |  Guide  |  |
|  | (Drop/Cam/  |  | (Filter/CSV |  | (Recharts / |  | (Confusion/ |  | (Taxon. |  |
|  |  Batch/CAM) |  |  /Inspect)  |  |  Real Rows) |  |  ECE/Speed) |  |  Rules) |  |
|  +------+------+  +------+------+  +------+------+  +------+------+  +----+----+  |
|         |                |                |                |              |       |
+---------+----------------+----------------+----------------+--------------+-------+
          | REST / Multipart HTTP (/api/*)
          v
+-----------------------------------------------------------------------------------+
|            Gateway Layer: Port 3000 (server.ts in Dev / Nginx in Prod)            |
+-----------------------------------------+-----------------------------------------+
                                          | Proxy /api/* -> 127.0.0.1:8000
                                          v
+-----------------------------------------------------------------------------------+
|                     FastAPI Application (backend/app/main.py)                     |
|                                                                                   |
|  +--------------------------+   +----------------------------------------------+  |
|  | Security & Validation    |   | Taxonomy Engine (backend/app/core/taxonomy)  |  |
|  | - Magic byte inspection  |   | - Startup YAML schema validation             |  |
|  | - Pillow decode + EXIF   |   | - 5 categories -> Recyclability, Bin,        |  |
|  | - Max dimension <= 4096  |   |   Prep Steps, Disposal Tips, Disclaimer      |  |
|  | - CSV formula escaping   |   +----------------------------------------------+  |
|  +------------+-------------+                                                     |
|               |                                                                   |
|               v                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | ONNX Runtime Inference Service (backend/app/ml/inference.py)                |  |
|  | - Loads artifacts/models/model.onnx + model_card.json                       |  |
|  | - Returns HTTP 503 if model artifact is missing (never fabricates output)   |  |
|  | - ImageNet normalization (224x224) + Post-hoc Temperature Scaling (T)       |  |
|  | - Confidence & margin gating (< 0.50 or margin < 0.12 -> "uncertain")       |  |
|  | - Optional explainability overlay (explain=true -> heatmap_png_base64)      |  |
|  +--------------------------------------+--------------------------------------+  |
|                                         |                                         |
|                                         v                                         |
|  +-----------------------------------------------------------------------------+  |
|  | Persistence Layer: SQLAlchemy 2.x + Alembic (backend/app/models/prediction) |  |
|  | - SQLite WAL mode by default (sqlite:///./data/ecosort.db) or PostgreSQL    |  |
|  | - Stores SHA-256 hash, probabilities JSON, top-k, inference_ms, is_demo     |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

## 2. Offline ML Training & Export Pipeline (`ml/scripts/`)
1. **`download_data.py`**: Downloads **TrashNet** (2,527 images) and **UCI RealWaste** (4,752 images) over HTTPS, normalizes images to `256x256` RGB JPEG on extraction, and writes `ml/data/raw/`.
2. **`verify_dataset.py`**: Validates every image file with Pillow (`verify()` + `convert("RGB")`), removes corrupt or `<32x32` images, computes average perceptual hashes (`8x8` aHash) to eliminate duplicates before splitting, and outputs `artifacts/reports/dataset_report.json`.
3. **`prepare_dataset.py`**: Applies `ml/configs/label_map.yaml` to map source folders into `{glass, metal, organic, paper, plastic}` and performs a deterministic (`seed=42`) stratified 70% train / 15% val / 15% test split (`ml/data/splits/{train,val,test}.csv`) with zero SHA-256 or perceptual hash leakage across splits.
4. **`train.py`**: Trains `EfficientNet-B0` (primary) and `MobileNetV3-Large` (comparison) with two-stage transfer learning (Stage 1: frozen backbone head warmup; Stage 2: full fine-tuning with cosine LR schedule, inverse-frequency class weights, label smoothing `0.05`, and early stopping on validation macro-F1).
5. **`evaluate.py`**: Evaluates checkpoints on the held-out test split, computes accuracy, macro/weighted precision, recall, F1, per-class metrics, confusion matrix, ROC-AUC, PR-AUC, fits post-hoc temperature scaling ($T$) on the validation set to minimize Expected Calibration Error (ECE), benchmarks CPU latency (p50/p95 over 100 runs), and saves `artifacts/metrics/metrics.json` plus failure cases.
6. **`export_onnx.py`**: Exports the winning PyTorch model to `artifacts/models/model.onnx` (`opset 17`), verifies PyTorch vs. ONNX Runtime numerical parity across 20 test images (`max_abs_diff < 1e-4`), and writes `artifacts/models/model_card.json`.
