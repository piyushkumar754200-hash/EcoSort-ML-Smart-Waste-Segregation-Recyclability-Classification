# Held-Out Test Set Evaluation Report (`artifacts/reports/evaluation_report.md`)

- **Selected Model:** `efficientnet_b0_waste5` (`efficientnet_b0`)
- **Preprocessing:** `Letterbox (224x224, fill=[124,116,104]) + 2-View Horizontal-Flip TTA`
- **Held-Out Test Set Size:** `758` deduplicated images
- **Test Accuracy:** `88.65%` (`0.8865`)
- **Test Top-3 Accuracy:** `97.89%` (`0.9789`)
- **Test Macro-F1:** `0.8760` (Target >= 0.85: `PASS`)
- **Test Weighted-F1:** `0.8865`
- **Expected Calibration Error (ECE):** `0.0556` (Temperature $T = 1.0000$)
- **Transparent Container Slice (Plastic vs Glass) Accuracy:** `80.39%` (`plastic->glass`: `10`, `glass->plastic`: `8`)

## 1. Backbone Comparison (Validation Macro-F1 Selection)

| Checkpoint | Backbone | Val Macro-F1 | Calibration Temp $T$ | Training Wall-Clock (s) |
| :--- | :--- | ---: | ---: | ---: |
| `efficientnet_b0_best.pt` | `efficientnet_b0` | `0.8367` | `1.0000` | `219.8s` |
| `mobilenet_v3_large_best.pt` | `mobilenet_v3_large` | `0.8811` | `1.0000` | `133.4s` |

## 2. Per-Class Test Performance

| Category | Precision | Recall | F1-Score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **glass** | `0.8583` | `0.8651` | `0.8617` | `126` |
| **metal** | `0.7535` | `0.8699` | `0.8075` | `123` |
| **organic** | `0.9528` | `0.9918` | `0.9719` | `122` |
| **paper** | `0.9522` | `0.9264` | `0.9391` | `258` |
| **plastic** | `0.8649` | `0.7442` | `0.8000` | `129` |

## 3. Per-Source Test Performance

| Dataset Source | Test Images | Accuracy | Macro-F1 |
| :--- | ---: | ---: | ---: |
| `realwaste` | `442` | `0.8801` | `0.8510` |
| `trashnet` | `316` | `0.8956` | `0.7094` |