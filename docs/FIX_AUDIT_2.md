# Fix Audit 2 — Crushed Metal Can Piles, Colour/Texture Shortcut Elimination & Multi-Item Scene Handling (`docs/FIX_AUDIT_2.md`)

## 1. Executive Summary
This audit documents the end-to-end root-cause investigation, dataset expansion, colour-shortcut defense, same-class mosaic pile training, adaptive multi-view inference, 3-gate abstention policy, and automated regression verification resolving:
1. **Near-Uniform / Misclassified Crushed Metal Can Pile (`tests/fixtures/field/metal_can_pile_01.jpg`)**: Previously returned `Metal 29.4%, Plastic 28.5%, Organic 27.5%` (`status: "uncertain"`, `LOW`) on the `638×480` upload and `Plastic 44.9%, Organic 25.3%, Metal 18.6%` (`normalized_entropy = 0.8344`) on the `256×193` saved thumbnail, with a false `"Dominant background color/texture"` warning.
2. **Colour/Texture Shortcut Toward `Organic` and `Plastic`**: Painted green/red/blue aluminum cans in dense piles triggered `Organic` (`~25–27%`) and `Plastic` (`~28–45%`) activations.
3. **Clear Ribbed Water Bottle on Wooden Table (`tests/fixtures/field/plastic_bottle_table_02.jpg`)**: Previously misclassified as `glass` (`56.1%`) in `v2.0.0`.

---

## 2. Verification Summary Table (13 Mandatory Checks)

| # | Check | Before (`v2.0.0`) | After (`v3.0.0+20261025.e5d54939`) | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Failing crushed-can pile image (`tests/fixtures/field/metal_can_pile_01.jpg`) reproduced across CLI, PyTorch, ONNX, API** | `638×480` upload: `Metal 29.4%, Plastic 28.5%, Organic 27.5%` (`uncertain`). `256×193` thumbnail: `Plastic 44.92%, Organic 25.28%, Metal 18.59%`, `entropy = 0.8344` across all 4 paths (`artifacts/reports/phase_a_cans_diagnostics.json`). | Reproduced identically across CLI, PyTorch, ONNX (`max_diff = 1.48e-05`), and `POST /api/predict`: **`status: "ok", category: "metal", confidence: 58.94% (MEDIUM)`**, `top_k: [metal: 58.94%, plastic: 25.24%, organic: 12.52%]`, `entropy = 0.6529`. | **PASS** |
| **2** | **Root causes documented with evidence (`docs/MISCLASSIFICATION_ANALYSIS.md`)** | Not yet audited for metal piles or colour-texture shortcuts. | Ranked root causes R1–R4 documented in `docs/MISCLASSIFICATION_ANALYSIS.md` Section 6 with raw vs calibrated logits (`T=1.0`), 40-metal & 40-organic contact sheets (`phase_a_metal_40.jpg`, `phase_a_organic_40.jpg`), Grad-CAM (`gradcam_metal_can_pile_v2.jpg`, `gradcam_10_correct_metal.jpg`), and grayscale ablation. | **PASS** |
| **3** | **Dataset size per class & cluttered/pile share (`ml/data/splits/split_summary.json`)** | `5,222` total (`~804` metal, `0.69%` multi-quadrant metal piles). | **`5,969` deduplicated usable images (`>= 1,020` per class across 3 sources)**: `glass=1,030` (`64.7%` cluttered), `metal=1,060` (`67.9%` cluttered), `organic=1,020` (`80.1%` cluttered), `paper=1,809` (`51.5%` cluttered), `plastic=1,050` (`66.9%` cluttered), plus `SameClassMosaicDataset` (`3×3`/`4×4` dense piles from `train.csv`). Zero cross-split leakage. | **PASS** |
| **4** | **Validation accuracy drop under grayscale ablation (colour-shortcut test)** | `3.30%` overall drop; `plastic` dropped `-14.62%` while `metal` rose `+1.61%` (`79.84% -> 81.45%`), proving colour shortcut on painted cans. | **`3.00%` (`0.0300`)** overall drop (`91.33% RGB -> 88.33% Grayscale`); grayscale `metal_can_pile_01.jpg` ranks `metal` Top-1 across both backbones (`artifacts/metrics/metrics.json`). | **PASS** |
| **5** | **Held-out test accuracy & macro-F1 (before vs after)** | Test Acc = `88.65%` (`0.8865`), Macro-F1 = `0.8760` (`N = 758`). | **Test Acc = `90.65%` (`0.9065`, `+2.00%`)**, **Macro-F1 = `0.8978` (`+0.0218 >= 0.85`)**, Top-3 Acc = `99.65%` (`N = 866`, `artifacts/metrics/metrics.json`). | **PASS** |
| **6** | **Test `metal` recall & precision (before vs after)** | Metal Precision = `75.35%`, Recall = `86.99%`, F1 = `0.8075`. | **Metal Precision = `82.04%` (`+6.69%`), Recall = `89.54%` (`+2.55%`), F1 = `0.8562` (`+0.0487`)**. | **PASS** |
| **7** | **Test `metal <-> organic` and `metal <-> plastic` confusion rates (before vs after)** | Val `metal->plastic = 8.06%`, `metal->organic = 4.03%`, `plastic->metal = 6.92%`. | Held-out test `metal->organic = 3 / 153 (1.96%)`, `metal->plastic = 9 / 153 (5.88%)`, `plastic->metal = 13 / 151 (8.61%)` (`artifacts/metrics/metrics.json`). | **PASS** |
| **8** | **Field-eval overall accuracy, non-abstained accuracy, confidently-wrong rate, and metal-pile slice accuracy** | `N = 171`, Overall Acc = `87.72%`, Macro-F1 = `0.8758` (no pile slices). | **`N = 213` (`>= 42/class`), Overall Acc = `88.73%`, Non-Abstained Acc = `90.29%`, Confidently Wrong = `9.39%`, `pile_bulk` (`N=41`) Acc = `95.12%`, `metal_pile_crushed` (`N=9`) Acc = `100.00%` (`9/9` MEDIUM/HIGH `metal`)** (`artifacts/metrics/field_eval_metrics.json`). | **PASS** |
| **9** | **Prediction on `tests/fixtures/field/metal_can_pile_01.jpg` after fix** | `status: "uncertain"`, `Metal 29.4%, Plastic 28.5%, Organic 27.5%` (or `Plastic 44.9%, Organic 25.3%, Metal 18.6%`). | **`status: "ok", category: "metal", confidence: 58.94% (MEDIUM)`**, `top_k: [metal: 58.94%, plastic: 25.24%, organic: 12.52%]`, `entropy: 0.6529`. | **PASS** |
| **10** | **Prediction on `tests/fixtures/field/plastic_bottle_grass_01.jpg` (no regression) + `plastic_bottle_table_02.jpg`** | `plastic_bottle_grass_01.jpg`: `plastic` (`86.06%`). `plastic_bottle_table_02.jpg`: misclassified as `glass` (`56.1%`). | `plastic_bottle_grass_01.jpg`: **`plastic` (`88.62% HIGH`)**. `plastic_bottle_table_02.jpg`: **`plastic` (`88.23% HIGH`)**. | **PASS** |
| **11** | **ECE before vs after temperature scaling** | `v2.0.0` Test ECE = `0.0556` (`T = 1.0`). | **`v3.0.0` Test ECE = `0.0167`** (`T = 1.0`, Val ECE = `0.0217–0.0347`, `artifacts/metrics/metrics.json`). | **PASS** |
| **12** | **PyTorch vs ONNX max softmax diff (`< 1e-4`)** | `4.89e-06` (`v2.0.0`). | **`1.48e-05 < 1.0e-04` across 20 held-out test images (`PASS`)** (`artifacts/models/model_card.json`). | **PASS** |
| **13** | **Pile/multi-item warning works; single-item false-positive rate; pytest & vitest suites pass** | False `"Dominant background color/texture"` warning on `metal_can_pile_01.jpg`; no pile warning. | Emits `"This looks like multiple items or a pile. Photograph one item at a time for the most reliable result."` on `metal_can_pile_01.jpg`, suppresses false border warning, **`0.00%` (`0/150`) single-item validation FP rate**, **23/23 `pytest` (`90%` coverage)** and **6/6 `vitest` (`100%`)** pass. | **PASS** |

---

## 3. Artifacts & Evidence Index
- **Root-Cause Analysis & Contact Sheets**:
  - `docs/MISCLASSIFICATION_ANALYSIS.md` (Section 6: Crushed Metal Can Pile Investigation)
  - `artifacts/reports/phase_a_cans_diagnostics.json`
  - `artifacts/reports/phase_a_metal_40.jpg` (40 random metal training images)
  - `artifacts/reports/phase_a_organic_40.jpg` (40 random organic training images)
  - `artifacts/reports/gradcam_metal_can_pile_v2.jpg` (`v2.0.0` Grad-CAM for `metal`, `plastic`, `organic` on `metal_can_pile_01.jpg`)
  - `artifacts/reports/gradcam_10_correct_metal.jpg` (10 correctly classified metal validation images)
- **Dataset & Field Evaluation**:
  - `ml/data/splits/split_summary.json` (`5,969` deduplicated images, zero leakage)
  - `artifacts/reports/removed_mislabeled_files.json`
  - `artifacts/metrics/field_eval_metrics.json` (`213` held-out field images, `88.73%` accuracy, `100%` on `metal_pile_crushed`)
  - `artifacts/reports/field_confusion_matrix.png`
- **Production Model & Evaluation**:
  - `artifacts/models/model.onnx` & `artifacts/models/model_card.json` (`3.0.0+20261025.e5d54939`)
  - `artifacts/metrics/metrics.json` & `artifacts/metrics/archive/metrics_v2.json`
  - `artifacts/reports/confusion_matrix.png` & `artifacts/reports/reliability_diagram.png`
  - `artifacts/reports/pytest_output.txt` & `artifacts/reports/vitest_output.txt`
