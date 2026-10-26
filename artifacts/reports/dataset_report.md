# Dataset Verification & Coverage Report (`artifacts/reports/dataset_report.md`)

- **Overall Verdict:** `PASS` (All 5 categories >= 1,000 usable deduplicated images, >= 25% cluttered/natural/multi-item share)
- **Total Raw Files Scanned:** `5977`
- **Near-Duplicates Removed (64-bit Perceptual `dHash` across all sources):** `8`
- **Total Clean, Deduplicated Usable Images:** `5969`

## 1. Five-Category Coverage Table

| Target Category | Clean Usable Images | RealWaste | TrashNet | Clutter/Piles | Cluttered/Natural Share | >= 1000 Target | Contact Sheet |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| **glass** | `1030` | `420` | `450` | `160` | `69.9%` | `PASS` | `artifacts/reports/contact_sheet_glass.jpg` |
| **metal** | `1060` | `450` | `408` | `202` | `64.3%` | `PASS` | `artifacts/reports/contact_sheet_metal.jpg` |
| **organic** | `1020` | `847` | `0` | `173` | `68.2%` | `PASS` | `artifacts/reports/contact_sheet_organic.jpg` |
| **paper** | `1809` | `900` | `849` | `60` | `80.3%` | `PASS` | `artifacts/reports/contact_sheet_paper.jpg` |
| **plastic** | `1050` | `450` | `448` | `152` | `68.4%` | `PASS` | `artifacts/reports/contact_sheet_plastic.jpg` |