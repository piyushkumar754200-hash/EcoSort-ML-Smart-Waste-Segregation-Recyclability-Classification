# Held-Out Test Set Evaluation Report (`artifacts/reports/evaluation_report.md`)

- **Selected Model:** `mobilenet_v3_large_waste5` (`mobilenet_v3_large`)
- **Held-Out Test Set Size:** `784` deduplicated images
- **Test Accuracy:** `86.48%` (`0.8648`)
- **Test Top-3 Accuracy:** `98.72%` (`0.9872`)
- **Test Macro-F1:** `0.8586` (Target >= 0.85: `PASS`)
- **Test Weighted-F1:** `0.8681`
- **Expected Calibration Error (ECE):** `0.0474` (Temperature $T = 1.0000$)

## 1. Backbone Comparison (Validation Macro-F1 Selection)

| Checkpoint | Backbone | Val Macro-F1 | Calibration Temp $T$ | Training Wall-Clock (s) |
| :--- | :--- | ---: | ---: | ---: |
| `efficientnet_b0_best.pt` | `efficientnet_b0` | `0.8570` | `1.0000` | `221.8s` |
| `mobilenet_v3_large_best.pt` | `mobilenet_v3_large` | `0.8654` | `1.0000` | `146.7s` |

## 2. Per-Class Test Performance

| Category | Precision | Recall | F1-Score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **glass** | `0.8678` | `0.8015` | `0.8333` | `131` |
| **metal** | `0.7698` | `0.8359` | `0.8015` | `128` |
| **organic** | `0.9760` | `0.9606` | `0.9683` | `127` |
| **paper** | `0.9785` | `0.8669` | `0.9194` | `263` |
| **plastic** | `0.6988` | `0.8593` | `0.7708` | `135` |

## 3. Per-Source Test Performance

| Dataset Source | Test Images | Accuracy | Macro-F1 |
| :--- | ---: | ---: | ---: |
| `realwaste` | `460` | `0.8674` | `0.8483` |
| `trashnet` | `324` | `0.8611` | `0.8452` |