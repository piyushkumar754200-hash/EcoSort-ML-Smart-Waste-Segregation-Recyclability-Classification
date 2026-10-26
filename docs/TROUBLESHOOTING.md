# Troubleshooting & Engineering Log (`docs/TROUBLESHOOTING.md`)

## Issue 1: Misclassification of `194×259` Transparent Plastic Bottle on Artificial Grass (`v1.0.0`)
- **Symptom**: Uploading `tests/fixtures/field/plastic_bottle_grass_01.jpg` (`194×259` px clear PET water bottle with teal cap on green artificial turf and dark circular base) produced `status: "uncertain"`, `Glass: 44.2%`, `Metal: 19.8%`, `Plastic: 18.1%`.
- **Root Causes Diagnosed (`docs/MISCLASSIFICATION_ANALYSIS.md`)**:
  1. **Spatial Center-Crop Truncation**: `Resize(256) + CenterCrop(224)` cropped off the top and bottom `13.5%` of the `194×259` portrait image, removing part of the distinctive teal plastic cap and dark circular base (`artifacts/reports/crop_vs_letterbox_bottle.jpg`).
  2. **Background Shortcut Learning**: `82.6%` of TrashNet training images had plain studio backgrounds, and zero training images were `< 224` px. Grad-CAM (`artifacts/reports/gradcam_failing_bottle.jpg`) proved the `v1.0.0` model attended to the green grass texture and dark circular base rather than the PET bottle body.
- **Fix Applied**:
  1. Replaced `Resize(256) + CenterCrop(224)` with `letterbox_pil_image(img, 224, fill_rgb=(124, 116, 104))` identically across training (`ml/scripts/model_factory.py`), evaluation (`ml/scripts/evaluate.py`), ONNX export (`ml/scripts/export_onnx.py`), and FastAPI inference (`backend/app/ml/inference.py`).
  2. Added `NaturalBackgroundAndLowResAugment` during training to synthesize textured natural borders (grass/foliage green, soil/wood brown, dark circular bases) around border-neutral images and simulate `140–210` px low-resolution JPEG downscale-upscale artifacts.

## Issue 2: Uncertain-State API & UI Showing Top-1 Disposal Instructions
- **Symptom**: When `status == "uncertain"`, `POST /api/predict` still returned the top-1 category's `category`, `bin`, `disposal_tips`, and `prep_steps` (`Glass`), and `PredictionResultCard.tsx` rendered "Preparation Before Disposal" and "Disposal Guidance" for Glass as if the classification were definite.
- **Fix Applied**:
  1. Updated `backend/app/api/routes.py`, `backend/app/schemas/api.py`, `backend/app/models/prediction.py`, and Alembic migration `0002_uncertain_nullable_category.py` so `status == "uncertain"` (when unconfirmed) returns `category: null`, `detected_item: null`, `recyclability_class: "Unknown"`, `bin: null`, `disposal_tips: []`, `prep_steps: []`, `safe_handling_advice`, and `candidate_guidance`.
  2. Updated `src/components/PredictionResultCard.tsx` so uncertain predictions hide the default disposal panels and instead display an amber Uncertainty Panel, Safe Handling Advice, interactive top-3 candidate guidance preview tabs, and 1-click user category confirmation (`POST /api/predictions/{id}/feedback`).

## Issue 3: Runtime `500 Internal Server Error` on `POST /api/predict` During Migration
- **Symptom**: After adding `_build_candidate_guidance(taxonomy)` in `backend/app/api/routes.py` and refactoring `backend/app/core/security.py`, `POST /api/predict` raised `AttributeError: 'list' object has no attribute 'items'` on `taxonomy.categories.items()` and `AttributeError: 'Settings' object has no attribute 'SAVE_THUMBNAILS'`.
- **Fix Applied**:
  1. Updated `_build_candidate_guidance` in `backend/app/api/routes.py` to iterate `for cat_key in taxonomy.categories:` and look up `entry = taxonomy.get_entry(cat_key)`.
  2. Updated `save_sanitized_thumbnail` in `backend/app/core/security.py` to read `settings.STORE_THUMBNAILS` and `settings.THUMBNAIL_MAX_DIM`.

## Issue 4: Near-Uniform / Misclassified Prediction on Crushed Metal Can Pile (`tests/fixtures/field/metal_can_pile_01.jpg`) + Colour/Texture Shortcut Toward Organic & Plastic
- **Symptom**: Uploading a photo of a pile of crushed painted aluminum cans (`638×480` px original upload / `256×193` px thumbnail `metal_can_pile_01.jpg`) produced `Metal 29.4%, Plastic 28.5%, Organic 27.5%` (`status: "uncertain"`, `LOW`) on the `638×480` upload and `Plastic 44.9%, Organic 25.3%, Metal 18.6%` (`normalized_entropy = 0.8344`) on the `256×193` thumbnail, along with a misleading quality warning `"Dominant background color/texture detected around the edges"`.
- **Root Causes Diagnosed (`docs/MISCLASSIFICATION_ANALYSIS.md`, Section 6)**:
  1. **Zero Pile Representation in Metal Training Split**: Only `4 / 577` (`0.69%`) of `v2.0.0` metal training images had multi-quadrant edge clutter (`mean_metal_edge_density = 0.0716`), whereas `metal_can_pile_01.jpg` had `edge_density = 0.4546`.
  2. **Colour/Texture Shortcut to `Organic` and `Plastic`**: On validation grayscale ablation (`artifacts/reports/phase_a_cans_diagnostics.json`), removing RGB colour *improved* `metal` recall (`79.84% -> 81.45%`, `+1.61%`) while `plastic` recall dropped by `-14.62%`. Grad-CAM (`artifacts/reports/gradcam_metal_can_pile_v2.jpg`) confirmed `organic` and `plastic` class activations locked onto green/red/blue beverage can paint rather than metallic creases.
  3. **Global Single-Scale Pooling Across Overlapping Cans**: Averaging a `7×7` feature map across 15+ small cans mixed heterogeneous paint colours into a flat embedding; evaluating `2×2` quadrant crops plus a grayscale de-colourised view immediately recovered `metal` as Top-1.
  4. **Stage-2 Small-Batch BatchNorm Stat Drift**: Unfreezing `top_blocks.train()` without freezing `BatchNorm2d` running stats caused validation F1 to dip at Epoch 3 during two-stage transfer learning.
- **Fix Applied**:
  1. Expanded the dataset to `5,969` deduplicated images (`>= 1,020` per class across 3 sources including `clutter_piles` multi-item can/bottle/scrap collages) and expanded `ml/data/field_eval/` to `213` real-world images (`>= 42` per class).
  2. Added dual-view feature extraction (`canonical` letterbox view + `shortcut_defense` view with `RandomGrayscale(p=0.45)`, `RandomChannelPermutation(p=0.12)`, partial-view crops, and specular highlights) and froze `BatchNorm2d` running statistics (`m.eval()`) during Stage 2 fine-tuning in `ml/scripts/train.py`.
  3. Added adaptive multi-view inference (`2×2` overlapping quadrant crops + grayscale de-colourised view on multi-item/high-entropy images) and a 3-gate abstention rule (`CONFIDENCE_MIN=0.50`, `MARGIN_MIN=0.12`, `ENTROPY_MAX=0.82`) in `backend/app/ml/inference.py`.
  4. Added `detect_pile_or_multi_item_scene` in `backend/app/core/security.py` to emit `"This looks like multiple items or a pile. Photograph one item at a time for the most reliable result."` while suppressing false `"Dominant background color/texture"` warnings on full-frame piles.
