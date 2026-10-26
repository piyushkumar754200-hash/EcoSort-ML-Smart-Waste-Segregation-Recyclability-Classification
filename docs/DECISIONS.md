# Technical Decisions Log (`docs/DECISIONS.md`)

All significant architectural, dataset, ML, and engineering decisions for **T237 – Smart Waste Segregation & Recyclability Classification System** are recorded below.

---

## Decision 1: 5 Canonical Categories + Confidence-Gated `"uncertain"` Outcome
- **Context**: Prompt §4.1 requires 5 target categories (`plastic`, `paper`, `metal`, `glass`, `organic`), support for an `"uncertain"` outcome (not a trained class) when confidence is below threshold, and an optional decision on `other/trash`.
- **Decision**: Train a 5-class classifier over `glass`, `metal`, `organic`, `paper`, and `plastic`. Exclude raw `trash` / `Miscellaneous Trash` / `Textile Trash` from the supervised training classes because those buckets mix composite plastics, paper, and metals with ambiguous ground truth, and handle out-of-distribution or ambiguous waste items via temperature-scaled softmax thresholding (`confidence < 0.50` or top-1 vs. top-2 margin `< 0.12` -> `status: "uncertain"` with `Special handling` guidance).
- **Rationale**: Keeps the 5 target classes clean and visually coherent while ensuring mixed/ambiguous items trigger explicit `"uncertain"` disposal safety guidance.

## Decision 2: Multi-Source Public Real-World Waste Dataset (RealWaste + TrashNet)
- **Context**: TrashNet (`garythung/trashnet`) contains 2,527 images across `glass`, `paper`, `cardboard`, `plastic`, `metal`, and `trash`, but has **zero `organic` images**. Relying on TrashNet alone would leave `organic` empty.
- **Decision**: Combine **UCI RealWaste** (`https://archive.ics.uci.edu/static/public/908/realwaste.zip`, 4,752 real landfill/sorting waste images including `Food Organics` and `Vegetation`) with **TrashNet** (`https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip`, 2,527 images) through a unified label mapping (`ml/configs/label_map.yaml`).
- **Rationale**: Provides **847 real organic waste images** (`Food Organics` + `Vegetation`) and **6,321 usable deduplicated images** across all 5 categories (`>= 847` images per category), exceeding the 150-images-per-class minimum by >5.6x.

## Decision 3: On-the-Fly 256×256 Compaction During Dataset Extraction
- **Context**: The workspace root `/` is a memory-backed `tmpfs` sharing a 4.0 GiB ceiling with process RAM. Storing raw uncompressed archives (`realwaste.zip` 656 MB + extracted 700 MB) alongside Python packages risks container OOM.
- **Decision**: `ml/scripts/download_data.py` streams each image from the zip archive, converts to RGB, resizes to `256x256` (`LANCZOS`), saves as JPEG (`quality=85`), and deletes the `.zip` archive immediately after extraction.
- **Rationale**: Since both `EfficientNet-B0` and `MobileNetV3-Large` train at `224x224` input resolution (`RandomResizedCrop(224)` / `CenterCrop(224)`), storing `256x256` intermediates preserves full training fidelity while reducing dataset disk footprint from ~1.4 GB to ~52 MB.

## Decision 4: Category-Level `detected_item` Integrity
- **Context**: Prompt §4.2 specifies: *"The `detected_item` is the fine-grained model label and must NEVER be an item name the model cannot actually distinguish. If the model is category-level only, `detected_item` equals the category label."*
- **Decision**: Because the unified model predicts the 5 canonical categories (`glass`, `metal`, `organic`, `paper`, `plastic`), `detected_item` is set equal to the predicted `category` label in `backend/app/core/taxonomy.yaml` and API responses, while fine-grained source mappings (`cardboard`, `food_organics`, `vegetation`) are documented in `taxonomy.yaml` under `fine_grained_aliases`.

## Decision 5: Post-Hoc Temperature Scaling Calibration & Grad-CAM Saliency
- **Context**: Raw neural network logits are often overconfident, which degrades threshold-based `"uncertain"` gating.
- **Decision**: Fit a scalar temperature parameter $T > 0$ via grid optimization minimizing Expected Calibration Error (ECE, 10 bins) on the held-out validation set (`ml/scripts/evaluate.py`), bake `temperature` into `artifacts/models/model_card.json`, and apply `logits / T` during ONNX inference (`backend/app/ml/inference.py`). For `explain=true`, generate an input-saliency heatmap overlay (`heatmap_png_base64`) directly from the input tensor and ONNX spatial activation response.

## Decision 6: Express + FastAPI Dual-Process Bridge on Port 3000
- **Context**: AI Studio preview routes traffic through Port 3000 (`server.ts`), whereas the specification requires a real FastAPI + SQLAlchemy + Alembic backend (`backend/app/main.py`).
- **Decision**: `server.ts` spawns `uvicorn app.main:app --host 127.0.0.1 --port 8000` using `.venv/bin/python` and proxies `/api/*` requests directly to FastAPI, while `docker-compose.yml` runs `backend` (FastAPI/Uvicorn) and `frontend` (Nginx) as separate containers.
