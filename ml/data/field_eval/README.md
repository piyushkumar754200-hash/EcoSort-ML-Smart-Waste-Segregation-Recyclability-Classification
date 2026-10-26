# Real-World Field Evaluation Dataset (`ml/data/field_eval/`)

## Purpose & Isolation Guarantee
This directory contains the held-out **Field Evaluation Set** (`171` images total; `34–35` images per class across `plastic`, `paper`, `metal`, `glass`, and `organic`) used by `ml/scripts/eval_field.py` and `backend/tests/test_api_and_ml.py` to measure real-world generalization on non-studio backgrounds, transparent containers, and mobile portrait aspect ratios.

- **Strict Isolation**: Every image in `ml/data/field_eval/` is carved out *before* `train.csv`, `val.csv`, and `test.csv` are created in `ml/scripts/prepare_dataset.py`. Perceptual (`dHash`) and cryptographic (`SHA-256`) assertions verify **zero overlap** between `field_eval` and `train.csv` / `val.csv` / `test.csv`.
- **Never Used for Training or Threshold Tuning**: Thresholds (`CONFIDENCE_MIN=0.50`, `CONFIDENCE_HIGH=0.80`, `MARGIN_MIN=0.12`) and temperature scaling ($T$) are tuned strictly on `ml/data/splits/val.csv`.

## Provenance & Licenses
1. **RealWaste (UCI Machine Learning Repository, DOI: `10.24432/C5SS4G`)** — CC BY 4.0 license (Single-item waste photographs captured in natural lighting and real backgrounds).
2. **TrashNet (Gary Thung & Mindy Yang, Stanford CS229)** — MIT License.
3. **Field Fixture (`plastic/plastic_bottle_grass_01.jpg`)** — `194×259` px portrait capture of a clear/transparent plastic water bottle with a teal cap on green artificial grass and a dark circular base.
