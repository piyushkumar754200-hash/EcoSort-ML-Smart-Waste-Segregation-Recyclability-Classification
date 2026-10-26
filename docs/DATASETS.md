# Dataset Documentation & Provenance (`docs/DATASETS.md`)

## 1. Data Sources & Licenses
EcoSort ML combines three open-licensed waste classification sources plus a strictly isolated held-out field evaluation split (`ml/data/field_eval/`):

1. **TrashNet** (`garythung/trashnet`, HuggingFace / GitHub mirror)
   - **License**: MIT License
   - **Raw Categories**: `glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`
   - **Characteristics**: Studio-lit single-item images (`512×384` px) predominantly on white/neutral posterboard backgrounds.
2. **RealWaste** (UCI Machine Learning Repository / GitHub `samkirkiles/realwaste`)
   - **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
   - **Raw Categories**: `Cardboard`, `Food Organics`, `Glass`, `Metal`, `Miscellaneous Trash`, `Paper`, `Plastic`, `Textile Trash`, `Vegetation`
   - **Characteristics**: Real-world waste items photographed at a recovery facility with natural lighting, varied angles, and non-uniform backgrounds.
3. **Clutter & Multi-Item Piles (`ml/data/raw/clutter_piles`)**
   - **License**: CC BY 4.0 / MIT (derived multi-object collages, crushed cans, stacked scrap metal, clear/tinted glass collections, organic piles, paper bundles, and clear/crushed plastic piles with zero source overlap with `val`, `test`, or `field_eval`)
   - **Raw Categories**: `glass`, `metal`, `organic`, `paper`, `plastic`
   - **Characteristics**: Multi-item piles (`2×2` and `3×2` overlapping compositions), crushed/deformed cans, crumpled foil/tins, and cluttered natural backgrounds designed to eliminate single-object studio bias and colour-to-class shortcuts.

## 2. Canonical 5-Class Label Mapping (`ml/configs/label_map.yaml`)
Audited in `docs/MISCLASSIFICATION_ANALYSIS.md`:
- **`plastic`**: `trashnet:plastic`, `realwaste:Plastic`, `clutter_piles:plastic`
- **`paper`**: `trashnet:paper`, `trashnet:cardboard`, `realwaste:Paper`, `realwaste:Cardboard`, `clutter_piles:paper`
- **`metal`**: `trashnet:metal`, `realwaste:Metal`, `clutter_piles:metal`
- **`glass`**: `trashnet:glass`, `realwaste:Glass`, `clutter_piles:glass`
- **`organic`**: `realwaste:Food Organics`, `realwaste:Vegetation`, `clutter_piles:organic`
- **Excluded ambiguous composite classes**: `trashnet:trash`, `realwaste:Miscellaneous Trash`, `realwaste:Textile Trash` (logged in `artifacts/reports/dataset_verification.json` and `artifacts/reports/removed_mislabeled_files.json`).

## 3. Deduplication, Per-Class Counts & Cluttered Share (`ml/data/splits/split_summary.json`)
`ml/scripts/verify_dataset.py` and `ml/scripts/prepare_dataset.py` perform SHA-256 and 64-bit difference perceptual hashing (`dHash`) verification across all raw images before splitting:
- **Total Usable Deduplicated Images**: `5,969` (`>= 1,000` per class across all 5 classes)
  - **`glass`**: `1,030` (`64.66%` cluttered/natural background share)
  - **`metal`**: `1,060` (`67.92%` cluttered/natural background share)
  - **`organic`**: `1,020` (`80.10%` cluttered/natural background share)
  - **`paper`**: `1,809` (`51.52%` cluttered/natural background share)
  - **`plastic`**: `1,050` (`66.86%` cluttered/natural background share)
- **Held-Out Real-World Field Evaluation Set (`ml/data/field_eval/`)**: `213` images (`42` glass, `43` metal, `42` organic, `42` paper, `44` plastic, including `tests/fixtures/field/metal_can_pile_01.jpg`, `tests/fixtures/field/plastic_bottle_grass_01.jpg`, and `tests/fixtures/field/plastic_bottle_table_02.jpg`) carved out **before** train/val/test splitting.
- **Training Split (`ml/data/splits/train.csv`)**: `4,028` images (`691` glass, `712` metal, `684` organic, `1,236` paper, `705` plastic)
- **Validation Split (`ml/data/splits/val.csv`)**: `865` images (`148` glass, `153` metal, `147` organic, `265` paper, `152` plastic)
- **Held-Out Test Split (`ml/data/splits/test.csv`)**: `866` images (`149` glass, `153` metal, `147` organic, `266` paper, `151` plastic)
- **Leakage Verification**: Zero SHA-256 or dHash overlap across `train.csv`, `val.csv`, `test.csv`, and `ml/data/field_eval/` (`zero_cross_split_hash_leakage: true`, `zero_field_eval_leakage: true`).
