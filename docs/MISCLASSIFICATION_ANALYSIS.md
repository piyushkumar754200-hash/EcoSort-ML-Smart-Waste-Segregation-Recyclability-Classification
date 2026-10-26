# Root-Cause Misclassification Analysis (`docs/MISCLASSIFICATION_ANALYSIS.md`)

This document records the empirical root-cause investigations for:
- **Investigation I (v1.0.0 -> v2.0.0)**: Transparent Plastic Water Bottle on Green Artificial Grass (`tests/fixtures/field/plastic_bottle_grass_01.jpg`, `194×259` px)
- **Investigation II (v2.0.0 -> v3.0.0)**: Pile of Crushed Metal Cans (`tests/fixtures/field/metal_can_pile_01.jpg`, `638×480` original / `256×193` saved fixture) + Colour/Texture Shortcut to Organic/Plastic + Clear Ribbed Plastic Bottle on Wooden Table (`tests/fixtures/field/plastic_bottle_table_02.jpg`, `250×444` original / `144×256` saved fixture).

All raw diagnostic numbers are persisted in `artifacts/reports/phase_a_diagnostics.json` and `artifacts/reports/phase_a_cans_diagnostics.json`.

---

## Part II: Crushed Metal Can Pile + Colour/Texture Shortcut Investigation (`v2.0.0` Model)

### A1. Reproduction Across All Four Inference Paths
- **User Live Upload (`images__3_.jpeg`, `638×480` px, recorded in `artifacts/ecosort.db` at `2026-10-25 17:30:36`)**:
  - Model version: `2.0.0+20261025.a38e0e76`
  - Output: `status = "uncertain"`, `confidence_level = "LOW"`, `top_k = [metal: 29.39%, plastic: 28.48%, organic: 27.52%]`
  - Quality warning emitted: `"Dominant background color/texture detected around the edges. Placing the item on a plain, neutral surface improves accuracy on transparent items."`
- **Saved Thumbnail Fixture (`tests/fixtures/field/metal_can_pile_01.jpg`, `256×193` px)**:
  - Evaluated across (a) `ml/scripts/predict_cli.py`, (b) PyTorch `selected_best.pt`, (c) ONNX Runtime `model.onnx`, and (d) `POST /api/predict`:
  - All four paths agree to 4 decimal places (`all_four_agree = true`):
    - `plastic`: `0.4489` (`44.89%`)
    - `organic`: `0.2527` (`25.27%`)
    - `metal`: `0.1863` (`18.63%`)
    - `paper`: `0.0607` (`6.07%`)
    - `glass`: `0.0514` (`5.14%`)
    - `status`: `"uncertain"` (`category: null`)

### A2. Raw Logits vs. Temperature-Calibrated Probabilities & Entropy
Measured on `tests/fixtures/field/metal_can_pile_01.jpg` (`artifacts/reports/phase_a_cans_diagnostics.json`):

| Class | Raw Logit (Orig) | Raw Logit (HFlip) | Raw Logit (2-View TTA) | Softmax Before Temp (`T=1.0`) | Softmax After Temp (`T=1.0`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`glass`** | `-1.8944` | `-1.6151` | `-1.7966` | `0.0514` | `0.0514` |
| **`metal`** | `-0.5034` | `-0.5221` | `-0.5099` | `0.1863` | `0.1863` |
| **`organic`** | `-0.2004` | `-0.2134` | `-0.2050` | `0.2527` | `0.2527` |
| **`paper`** | `-1.7635` | `-1.3866` | `-1.6316` | `0.0607` | `0.0607` |
| **`plastic`** | `+0.5115` | `+0.1061` | `+0.3696` | `0.4489` | `0.4489` |

- **Shannon Entropy ($H$)**: `1.3429 nats` (out of maximum possible $\ln 5 = 1.6094\text{ nats}$)
- **Normalized Entropy ($H / \ln 5$)**: **`0.8344`** (`83.4%` of maximum uniform entropy; on the `638×480` upload `29.4% / 28.5% / 27.5%`, normalized entropy is **`0.891`**).
- **Finding**: Because `temperature = 1.0`, temperature scaling is **not** what flattened the distribution—the **raw logits themselves are near-uniform and unseparated** (`metal = -0.51`, `organic = -0.21`, `plastic = +0.37`), proving that the `v2.0.0` feature extractor genuinely confuses a dense multi-coloured pile of crushed cans with `plastic` and `organic`.

### A3. Training-Data Audit for `metal` vs. `organic`
Audited across the `3,536` training images in `ml/data/splits/train.csv`:
- **Per-Class Training Counts**:
  - `paper`: `1,200` (`614` RealWaste, `586` TrashNet)
  - `plastic`: `605` (`299` RealWaste, `306` TrashNet)
  - `glass`: `585` (`278` RealWaste, `307` TrashNet)
  - `metal`: `577` (`299` RealWaste, `278` TrashNet)
  - `organic`: `569` (`569` RealWaste, `0` TrashNet)
- **Structural Breakdown of the `577` `metal` Training Images**:
  - `(a) Single clean can on plain background`: **`297` (`51.47%`)**
  - `(b) Crushed/deformed single cans`: **`0` (`0.0%`)** explicitly high-edge crushed cans in TrashNet; RealWaste has deformed single tins/cans grouped under general single items
  - `(c) Multi-item piles / scrap / bulk metal filling frame`: **`4` (`0.69%`)** (`min_quadrant_edge >= 0.22` and `edge_density >= 0.28`)
  - `(d) Foil / tins / lids / other single metal items`: **`276` (`47.83%`)**
- **Visual Feature Comparison (`metal` vs. `organic` vs. `metal_can_pile_01.jpg`)**:
  - Mean edge density (`train.csv` `metal`): `0.0716`
  - Mean edge density (`train.csv` `organic`): `0.0912` (**27.4% higher** than `metal` because `organic` consists of multi-piece food waste and tangled vegetation filling the frame)
  - Edge density of `metal_can_pile_01.jpg`: **`0.4546`** (**6.35× higher** than average `metal` training images, with `min_quadrant_edge = 0.4085` across all 4 quadrants)
  - Colourfulness of `metal_can_pile_01.jpg`: **`54.47`** (**2.65× higher** than `mean_metal_colourfulness = 20.52` due to bright green Heineken/Sprite cans, red Coca-Cola cans, blue Pepsi cans, and gold cans)
- **Contact Sheets Saved**:
  - `artifacts/reports/phase_a_metal_40.jpg` (40 random training metal images: almost exclusively single isolated cans/tins)
  - `artifacts/reports/phase_a_organic_40.jpg` (40 random training organic images: dense, multi-piece, colourful food scraps and green foliage)

### A4. Shortcut Check: Grad-CAM & Validation Colour-Ablation Test
- **Grad-CAM Visual Evidence**:
  - `artifacts/reports/gradcam_metal_can_pile_v2.jpg`: Shows that on `metal_can_pile_01.jpg`, the `organic` class activation focuses directly on the **bright green crushed cans** (mistaking green-painted aluminum with white/silver crumpled highlights for green vegetation/produce), while the `plastic` activation focuses on the **glossy red and blue printed cylindrical bodies**, and `metal` only weakly activates on two silver can tops!
  - `artifacts/reports/gradcam_10_correct_metal.jpg`: Shows that on single metal cans, the model relies on a single isolated cylindrical silhouette against a uniform background.
- **Validation Colour-Ablation Test (`N = 758` validation images, RGB vs. 3-channel Grayscale)**:
  - **Overall Validation RGB Accuracy**: `85.88%` (`0.8588`)
  - **Overall Validation Grayscale Accuracy**: `80.87%` (`0.8087`, overall drop `-5.01%`)

| Category | Support | RGB Precision | RGB Recall | RGB F1 | Grayscale Recall | Grayscale F1 | Recall Drop (RGB -> Gray) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`glass`** | 125 | 0.8560 | 0.8560 | 0.8560 | 0.8320 | 0.7790 | `+2.40%` |
| **`metal`** | 124 | 0.6875 | 0.7984 | 0.7388 | **0.8145** | 0.7189 | **`-1.61%` (IMPROVES!)** |
| **`organic`** | 122 | 0.9213 | 0.9590 | 0.9398 | 0.9672 | 0.9365 | `-0.82%` |
| **`paper`** | 257 | 0.9603 | 0.9416 | 0.9509 | 0.8677 | 0.9010 | `+7.39%` |
| **`plastic`** | 130 | 0.7818 | 0.6615 | 0.7167 | **0.5154** | 0.6063 | **`+14.62%` (LARGE DROP)** |

- **Critical Insight from Grayscale Ablation**:
  - Removing colour **increases `metal` recall by `+1.61%`** (`79.84% -> 81.45%`) while `plastic` recall drops by `14.62%`! This proves that bright saturated colours on painted metal cans actively bias the `v2.0.0` model toward `plastic` and `organic`, whereas in grayscale the model attends to metallic rims, pull-tabs, and specular crease geometry.

### A5. Validation Confusion Matrix & Confusion Rates (`N = 758`)
Rows = true class, Columns = predicted class (`glass, metal, organic, paper, plastic`):
- `glass` (`125`): `[107, 15, 0, 0, 3]`
- `metal` (`124`): `[5, 99, 2, 3, 15]`
- `organic` (`122`): `[1, 0, 117, 0, 4]`
- `paper` (`257`): `[2, 6, 5, 242, 2]`
- `plastic` (`130`): `[10, 24, 3, 7, 86]`
- **Key Pairwise Confusion Rates on Validation**:
  - `metal -> plastic`: **`12.10%`** (`15 / 124`)
  - `plastic -> metal`: **`18.46%`** (`24 / 130`)
  - `plastic -> glass`: **`7.69%`** (`10 / 130`)
  - `metal -> glass`: **`4.03%`** (`5 / 124`)
  - `metal -> organic`: **`1.61%`** (`2 / 124` on single-item validation, rising to `25.3%–27.5%` probability on full-frame multi-can piles).

### A6. Ranked Root Causes (with Evidence)
1. **Severe Scarcity of Multi-Object / Crushed-Can Pile Scenes in Training (`0.69%` of Metal)**:
   - `99.3%` of `metal` training images depict a single isolated object (`mean_edge_density = 0.0716`), whereas `metal_can_pile_01.jpg` contains ~20 crushed cans filling all 4 quadrants (`edge_density = 0.4546`, `min_quadrant_edge = 0.4085`). When resized to `224×224`, each can is ~35 px wide, outside the scale distribution learned from single-can images.
2. **Colour/Clutter Shortcut Linking Green/Red/Blue Clutter to `Organic` and `Plastic`**:
   - Grad-CAM (`artifacts/reports/gradcam_metal_can_pile_v2.jpg`) and grayscale ablation (`+1.61%` higher metal recall when colour is stripped; `14.62%` colour dependency on plastic) prove the model uses saturated green patches as a shortcut for `organic` (vegetation) and saturated red/blue glossy patches as a shortcut for `plastic`.
3. **Single-Scale Global Inference Loses Local Can Geometry on Dense Piles**:
   - Evaluating only the downscaled `224×224` full-frame image blurs the pull-tabs, double-seamed aluminum rims, and sharp metallic creases that distinguish crushed aluminum cans from plastic wrappers or organic waste. Cropping `2×2` overlapping quadrants restores 2× spatial resolution on individual cans.
4. **Refinement Needed in Border Background Warning vs. Multi-Item Pile Warning**:
   - `compute_quality_warnings` triggered `"Dominant background color/texture detected around the edges"` whenever `mean_border_sat > 0.36`, even when the entire frame (both border and interior) was packed with colourful cans (`border_uniformity = 0.0593`, `min_quadrant_edge = 0.4085`). A full-frame pile should trigger the **Multi-Item / Pile Warning**, whereas the **Dominant Background Border Warning** should only fire when the border is uniform/low-edge relative to a localized central object.
5. **Clear Ribbed Plastic Bottle (`plastic_bottle_table_02.jpg`) Predicted as Glass (`56.2%` Glass vs. `38.4%` Plastic)**:
   - Tall clear PET water bottles with horizontal structural ribs and specular reflections on indoor wooden tables (`images__1_.jpeg`) still exhibit a narrow margin (`56.2%` vs `38.4%`, margin `0.178`) toward `glass` because training lacked enough clear ribbed/labeled PET bottles on indoor wood/cluttered backgrounds.

---

## Part I: Archive of Phase A Investigation I (`v1.0.0` Transparent Plastic Bottle on Grass)
- See `artifacts/reports/phase_a_diagnostics.json`, `artifacts/reports/crop_vs_letterbox_bottle.jpg`, and `artifacts/reports/gradcam_failing_bottle.jpg` for the original `v1.0.0` investigation (`Resize(256)+CenterCrop(224)` truncation + artificial turf shortcut), which was resolved in `v2.0.0` (`plastic_bottle_grass_01.jpg` classified as `plastic` with `86.06%` confidence).
