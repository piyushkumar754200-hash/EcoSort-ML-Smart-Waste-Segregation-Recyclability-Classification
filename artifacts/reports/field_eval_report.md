# Real-World Field Evaluation Report (`artifacts/reports/field_eval_report.md`)

- **Model Version:** `3.0.0+20261025.988a7aab`
- **Field Evaluation Samples:** `213` (`>= 42` per class across 5 classes, strictly disjoint from train/val/test)
- **Overall Top-1 Accuracy:** `85.92%` (`0.8592`)
- **Non-Abstained (`status == 'ok'`) Accuracy:** `88.61%` (`0.8861`)
- **Confidently Wrong Rate:** `10.80%` (`0.1080`)
- **Macro-F1 Score:** `0.8593`
- **Abstention (`status == 'uncertain'`) Rate:** `5.16%`

## 1. Per-Class Field Performance

| Category | Accuracy | Precision | Recall | F1-Score | Support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **glass** | `0.8333` | `0.8750` | `0.8333` | `0.8537` | `42` |
| **metal** | `0.8372` | `0.8571` | `0.8372` | `0.8471` | `43` |
| **organic** | `0.9762` | `0.9762` | `0.9762` | `0.9762` | `42` |
| **paper** | `0.9524` | `0.8511` | `0.9524` | `0.8989` | `42` |
| **plastic** | `0.7045` | `0.7381` | `0.7045` | `0.7209` | `44` |

## 2. Field Evaluation Slices (`pile_bulk`, `single_item`, `low_res`, `metal_pile_crushed`)

| Slice | Count | Accuracy | Non-Abstained Accuracy | Abstention Rate | Confidently Wrong Rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **pile_bulk** | `41` | `0.9024` | `0.9459` | `0.0976` | `0.0488` |
| **single_item** | `172` | `0.8488` | `0.8727` | `0.0407` | `0.1221` |
| **low_res** | `47` | `0.8936` | `0.9302` | `0.0851` | `0.0638` |
| **metal_pile_crushed** | `9` | `1.0000` | `1.0000` | `0.1111` | `0.0000` |

## Crushed Metal Can Pile (`metal_can_pile_01.jpg`)

- **Predicted Top-1:** `metal`
- **Status:** `uncertain`
- **Returned Category:** `None`
- **Confidence:** `42.57%` (`LOW`)
- **Normalized Entropy:** `0.8517`
- **Top-3 Candidates:** `[{"label": "metal", "category": "metal", "probability": 0.4257}, {"label": "organic", "category": "organic", "probability": 0.2511}, {"label": "plastic", "category": "plastic", "probability": 0.1858}]`

## Transparent Plastic Bottle on Grass (`plastic_bottle_grass_01.jpg`)

- **Predicted Top-1:** `plastic`
- **Status:** `ok`
- **Returned Category:** `plastic`
- **Confidence:** `76.20%` (`MEDIUM`)
- **Normalized Entropy:** `0.5122`
- **Top-3 Candidates:** `[{"label": "plastic", "category": "plastic", "probability": 0.762}, {"label": "metal", "category": "metal", "probability": 0.1207}, {"label": "organic", "category": "organic", "probability": 0.0627}]`

## Clear Ribbed Plastic Bottle on Table (`plastic_bottle_table_02.jpg`)

- **Predicted Top-1:** `plastic`
- **Status:** `ok`
- **Returned Category:** `plastic`
- **Confidence:** `76.78%` (`MEDIUM`)
- **Normalized Entropy:** `0.3640`
- **Top-3 Candidates:** `[{"label": "plastic", "category": "plastic", "probability": 0.7678}, {"label": "glass", "category": "glass", "probability": 0.2233}, {"label": "organic", "category": "organic", "probability": 0.0066}]`