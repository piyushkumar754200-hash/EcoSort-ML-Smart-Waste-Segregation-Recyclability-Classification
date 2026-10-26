# Fix Audit & Verification Report (`docs/FIX_AUDIT.md`)

## Executive Summary
All requirements across Phases A through G have been diagnosed, implemented, verified with real commands, and persisted under `artifacts/` and `docs/`.

---

## Phase A: Reproduce & Diagnose (`docs/MISCLASSIFICATION_ANALYSIS.md`)
- **A1 (Reproduction)**: Saved the `194×259` clear plastic bottle on artificial grass fixture at `tests/fixtures/field/plastic_bottle_grass_01.jpg` and verified parity across `predict_cli.py`, PyTorch, ONNX Runtime, and `POST /api/predict`.
- **A2 & A3 (Root-Cause Evidence)**:
  - Generated contact sheets (`artifacts/reports/phase_a_plastic_20.jpg`, `artifacts/reports/phase_a_glass_20.jpg`), Grad-CAM heatmaps (`artifacts/reports/gradcam_failing_bottle.jpg`, `artifacts/reports/gradcam_10_correct_plastic.jpg`), and crop-vs-letterbox comparison (`artifacts/reports/crop_vs_letterbox_bottle.jpg`).
  - Documented the ranked root causes in `docs/MISCLASSIFICATION_ANALYSIS.md`:
    1. **Background shortcut learning** (`82.6%` of TrashNet images have plain studio backgrounds; green artificial grass and dark circular base triggered glass/metal features).
    2. **Aspect-ratio crop truncation** (`Resize(256)+CenterCrop(224)` clipped the top and bottom `13.5%` of the tall `194×259` bottle).
    3. **Low-resolution domain gap** (`0` training images had `< 224` px resolution).
    4. **Uncertain-state API/UI leak** (`status == "uncertain"` returned and rendered top-1 `Glass` disposal instructions).

---

## Phase B: Real-World Held-Out Field Evaluation Set (`ml/data/field_eval/`)
- Created `ml/data/field_eval/{plastic,paper,metal,glass,organic}/` with `171` real-world images (`34` glass, `34` metal, `34` organic, `34` paper, `35` plastic including `plastic_bottle_grass_01.jpg`) and `ml/data/field_eval/README.md`.
- Verified **zero SHA-256 or dHash overlap** between `ml/data/field_eval/` and `train.csv`/`val.csv`/`test.csv` (`test_field_eval_isolation_and_fixture_classification` PASSED).
- Implemented `ml/scripts/eval_field.py` outputting `artifacts/metrics/field_eval_metrics.json` and `artifacts/reports/field_eval_report.md`:
  - **Overall Field Accuracy**: **87.72%** (`150 / 171`)
  - **Field Macro-F1**: **0.8758**
  - **Abstention (`uncertain`) Rate**: **10.53%** (`18 / 171`)
  - **`plastic_bottle_grass_01.jpg` Result**: `status: "ok"`, `category: "plastic"`, `confidence: 86.06% (HIGH)`, `top_k: [plastic: 86.06%, glass: 12.62%, paper: 0.66%]`.

---

## Phase C: Model & Preprocessing Improvements (`v2.0.0+20261025.a38e0e76`)
- **Letterbox Preprocessing**: Unified `letterbox_pil_image(img, 224, fill_rgb=(124, 116, 104))` across `ml/scripts/model_factory.py` and `backend/app/ml/inference.py`.
- **Shortcut-Breaking Augmentations**: Added `NaturalBackgroundAndLowResAugment` (textured green/earth/dark-coaster background synthesis on border-neutral training images + low-res `140–210` px downscale-upscale simulation + color jitter + random grayscale + random erasing).
- **Retraining & Evaluation**:
  - Trained `efficientnet_b0` (`219.81s`) and `mobilenet_v3_large` (`133.39s`).
  - Selected `efficientnet_b0` (`v2.0.0+20261025.a38e0e76`) as primary production model.
  - **Held-Out Test Set (`N = 758`)**: Accuracy improved from `87.15%` (`v1.0.0`) to **88.65%** (`v2.0.0`), Macro-F1 improved from `0.8676` to **0.8760**, and Plastic precision improved from `82.79%` to **86.49%**.
  - **ONNX Parity**: Verified across 20 held-out test images (`max_abs_softmax_diff = 4.89e-06 < 1.0e-04`, **PASS**).

---

## Phase D: Uncertain-State Backend & Frontend Fixes
- **Backend (`backend/app/api/routes.py`, `backend/app/schemas/api.py`, Alembic `0002_uncertain_nullable_category.py`)**:
  - When `status == "uncertain"` (confidence `< 0.50` or top1-top2 margin `< 0.12`), the response returns `category: null`, `detected_item: null`, `recyclability_class: "Unknown"`, `bin: null`, `disposal_tips: []`, `prep_steps: []`, `safe_handling_advice`, and `candidate_guidance`.
  - Added image quality checks in `backend/app/core/security.py` flagging low-resolution inputs (`min(w, h) < 224`) and dominant background borders (`> 55%` border uniformity).
  - Analytics (`/api/analytics`) and CSV export (`/api/predictions/export`) track uncertain predictions separately (`uncertain_count`, `uncertain_rate`, CSV category `"uncertain"`) without polluting confirmed category counts.
- **Frontend (`src/components/PredictionResultCard.tsx`, `src/components/HistoryView.tsx`, `src/components/AnalyticsView.tsx`)**:
  - Hides default category-specific "Preparation Before Disposal" and "Disposal Guidance" panels when `status == "uncertain"`.
  - Displays the Uncertainty Explanation banner, Safe Handling Advice list, interactive top-3 candidate guidance preview tabs (`View if Glass`, `View if Metal`, `View if Plastic`), and 1-click user category confirmation buttons.

---

## Phase E–G: Automated Verification Summary
- **Backend Test Suite (`pytest`)**: `18 passed` in `0.75s`, **90% total code coverage** (`artifacts/reports/pytest_output.txt`).
- **Frontend Test Suite (`vitest`)**: `5 passed` in `1.48s` (`artifacts/reports/vitest_output.txt`), explicitly testing both high-confidence plastic bottle rendering and uncertain-state panel suppression + candidate guidance preview.
