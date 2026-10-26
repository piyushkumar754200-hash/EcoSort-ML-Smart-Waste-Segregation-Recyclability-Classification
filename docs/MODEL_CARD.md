# Model Card — EcoSort ML Waste Classifier (`v3.0.0+20261025.e5d54939`)

## 1. Model Details
- **Model Name**: EcoSort ML Waste Classifier
- **Model Version**: `3.0.0+20261025.e5d54939`
- **Selected Architecture**: `efficientnet_b0` (pretrained on ImageNet-1K, two-stage fine-tuned on 5-class waste taxonomy with dual-view shortcut defense + same-class `3×3`/`4×4` mosaic pile training and frozen `BatchNorm2d` statistics)
- **Exported Format**: ONNX opset 17 (`artifacts/models/model.onnx`)
- **Weights SHA-256**: `e5d54939` (full digest in `artifacts/models/model_card.json`)
- **PyTorch vs. ONNX Parity**: Verified across 20 held-out test images (`max_abs_softmax_diff = 1.48e-05 < 1.0e-04`, **PASS**)

## 2. Input & Preprocessing Specification
- **Input Tensor Name**: `input`
- **Shape & Dtype**: `[batch_size, 3, 224, 224]`, `float32`, RGB color space
- **Spatial Preprocessing**: Aspect-ratio-preserving `LetterboxTransform(224, fill_rgb=(124, 116, 104))`
  - Scales `max(width, height)` to `224` using `BILINEAR` interpolation and pads the shorter axis symmetrically with ImageNet mean RGB `(124, 116, 104)`.
- **Normalization**: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`
- **Adaptive Multi-View Inference**:
  - **Single-Item / Low-Entropy Scenes**: 2-view horizontal flip (`0.65 * logits(orig) + 0.35 * logits(hflip)`), temperature `T = 1.0`.
  - **Multi-Item Piles / High-Entropy Scenes (`is_pile` or `normalized_entropy > 0.72`)**: Combines 5 overlapping `46%`-scale sub-crops (`top-left`, `top-right`, `bottom-left`, `bottom-right`, `center`), a full-frame grayscale de-colourised view, and the full-frame RGB view (`0.50 * tiles_logits + 0.35 * gray_logits + 0.15 * full_logits`) so individual crushed cans/containers are evaluated at canonical scale without colour-to-class shortcuts.

## 3. Three-Gate Confidence, Margin & Entropy Abstention Policy
Derived via validation risk-coverage analysis (`artifacts/reports/risk_coverage_curve.json` and `backend/app/config.py`):
- **`CONFIDENCE_HIGH`**: `0.80` (`top1_prob >= 0.80`, `top1 - top2 >= 0.12`, `normalized_entropy <= 0.82` -> `HIGH`, `status: "ok"`)
- **`CONFIDENCE_MIN`**: `0.50` (`0.50 <= top1_prob < 0.80`, `top1 - top2 >= 0.12`, `normalized_entropy <= 0.82` -> `MEDIUM`, `status: "ok"`)
- **`MARGIN_MIN`**: `0.12` (`top1 - top2 < 0.12` -> `LOW`, `status: "uncertain"`)
- **`ENTROPY_MAX`**: `0.82` (`normalized_entropy > 0.82` or `top1 - top3 < 0.10` -> `LOW`, `status: "uncertain"`, `"The model could not distinguish between these materials."`)
- When `status == "uncertain"`, the backend returns `category: null`, `detected_item: null`, `recyclability_class: "Unknown"`, `bin: null`, `disposal_tips: []`, `prep_steps: []`, `safe_handling_advice`, and `candidate_guidance` for the top-3 candidate classes.

## 4. Before / After Held-Out Test Set Evaluation (`v1.0.0` vs. `v2.0.0` vs. `v3.0.0`)
Archived metrics are preserved at `artifacts/metrics/archive/metrics_v1.json` and `artifacts/metrics/archive/metrics_v2.json`. Current `v3.0.0` metrics are at `artifacts/metrics/metrics.json`.

| Metric | v1.0.0 (`CenterCrop`) | v2.0.0 (`Letterbox + 2-View TTA`) | v3.0.0 (`Shortcut Defense + Mosaic + Adaptive Multi-View`) | Delta (`v2 -> v3`) |
| :--- | :---: | :---: | :---: | :---: |
| **Held-Out Test Samples** | 747 | 758 | **866** | +108 |
| **Test Accuracy** | 87.15% (`0.8715`) | 88.65% (`0.8865`) | **90.65%** (`0.9065`) | **+2.00%** |
| **Test Macro-F1** | 0.8676 | 0.8760 | **0.8978** | **+0.0218** |
| **Test Macro-Precision** | 0.8691 | 0.8763 | **0.8982** | **+0.0219** |
| **Test Macro-Recall** | 0.8682 | 0.8795 | **0.8989** | **+0.0194** |
| **Test Weighted-F1** | 0.8716 | 0.8865 | **0.9060** | **+0.0195** |
| **Test Top-3 Accuracy** | 98.53% (`0.9853`) | 97.89% (`0.9789`) | **99.65%** (`0.9965`) | **+1.76%** |
| **Expected Calibration Error (ECE)** | 0.0369 | 0.0556 | **0.0167** | **-0.0389** |
| **Metal F1** | 0.8240 | 0.8075 | **0.8562** | **+0.0487** |
| **Glass F1** | 0.8550 | 0.8617 | **0.8805** | **+0.0188** |
| **Organic F1** | 0.9268 | 0.9719 | **0.9833** | **+0.0114** |
| **Paper F1** | 0.9022 | 0.9391 | **0.9623** | **+0.0232** |
| **Plastic F1** | 0.8300 | 0.8000 | **0.8069** | **+0.0069** |
| **Validation Grayscale Ablation Drop** | N/A | 3.30% (`0.0330`) | **3.00%** (`0.0300`) | **-0.30%** |
| **Single-Item Pile Detector FP Rate** | N/A | N/A | **0.00%** (`0 / 150`) | **0.00%** |

### Per-Class Held-Out Test Metrics (`v3.0.0`, `N = 866`)
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `glass` | 0.8958 | 0.8658 | 0.8805 | 149 |
| `metal` | 0.8204 | 0.8954 | 0.8562 | 153 |
| `organic` | 0.9671 | 1.0000 | 0.9833 | 147 |
| `paper` | 0.9659 | 0.9586 | 0.9623 | 266 |
| `plastic` | 0.8417 | 0.7748 | 0.8069 | 151 |

### Held-Out Test Confusion Matrix (`v3.0.0`, rows = true, cols = predicted)
| True \ Pred | `glass` | `metal` | `organic` | `paper` | `plastic` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`glass`** | **129** | 11 | 0 | 0 | 9 |
| **`metal`** | 2 | **137** | 3 | 2 | 9 |
| **`organic`** | 0 | 0 | **147** | 0 | 0 |
| **`paper`** | 1 | 6 | 0 | **255** | 4 |
| **`plastic`** | 12 | 13 | 2 | 7 | **117** |

## 5. Held-Out Real-World Field Evaluation (`ml/data/field_eval`, `N = 213`)
Evaluated via `ml/scripts/eval_field.py` (`artifacts/metrics/field_eval_metrics.json`):
- **Total Samples**: `213` (`42` glass, `43` metal, `42` organic, `42` paper, `44` plastic; zero SHA-256 or dHash overlap with `train.csv`, `val.csv`, or `test.csv`)
- **Overall Top-1 Accuracy**: **88.73%** (`0.8873`, vs `87.72%` on `171` images in `v2.0.0`)
- **Non-Abstained Accuracy**: **90.29%** (`0.9029`)
- **Confidently Wrong Rate**: **9.39%** (`0.0939`)
- **Abstention (`status == "uncertain"`) Rate**: **3.29%** (`7 / 213`)
- **Performance Slices (`field_eval`)**:
  - **`metal_pile_crushed` (`N = 9`)**: **100.00%** accuracy (`9 / 9` classified as `metal` with `MEDIUM` or `HIGH` confidence, `0.0%` confidently wrong)
  - **`pile_bulk` (`N = 41`)**: **95.12%** accuracy (`95.12%` non-abstained accuracy, `4.88%` confidently wrong)
  - **`single_item` (`N = 172`)**: **87.21%** accuracy (`89.09%` non-abstained accuracy)
  - **`low_res` (`N = 47`)**: **93.62%** accuracy (`95.65%` non-abstained accuracy)
- **User Field Fixtures**:
  1. **`tests/fixtures/field/metal_can_pile_01.jpg` (`256×193` px pile of crushed aluminum cans)**:
     - **Before (`v2.0.0`)**: `status: "uncertain"` (`638×480` upload: `Metal 29.4%, Plastic 28.5%, Organic 27.5%`; `256×193` thumbnail: `Plastic 44.9%, Organic 25.3%, Metal 18.6%`), with false `"Dominant background color/texture"` warning.
     - **After (`v3.0.0`)**: `status: "ok"`, `category: "metal"`, `confidence: 58.94%` (`0.5894`, `MEDIUM`), `top_k: [metal: 58.94%, plastic: 25.24%, organic: 12.52%]`, `entropy: 0.6529`, with honest warning `"This looks like multiple items or a pile. Photograph one item at a time for the most reliable result."` and zero false background warning.
  2. **`tests/fixtures/field/plastic_bottle_grass_01.jpg` (`194×259` px clear bottle on artificial grass)**:
     - **After (`v3.0.0`)**: `status: "ok"`, `category: "plastic"`, `confidence: 88.62%` (`0.8862`, `HIGH`, improved from `86.06%` in `v2.0.0`).
  3. **`tests/fixtures/field/plastic_bottle_table_02.jpg` (`144×256` px clear ribbed bottle on wooden table)**:
     - **Before (`v2.0.0`)**: Misclassified as `glass` (`56.1%`).
     - **After (`v3.0.0`)**: `status: "ok"`, `category: "plastic"`, `confidence: 88.23%` (`0.8823`, `HIGH`).
