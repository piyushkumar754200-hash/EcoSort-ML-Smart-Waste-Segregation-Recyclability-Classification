# Human-in-the-Loop Feedback Staging Directory

This directory (`artifacts/feedback_staging/`) contains user-confirmed waste images and labels exported by `ml/scripts/export_feedback_dataset.py`.

## Safety & Governance Rules
1. **Never Auto-Merged**: Files in this staging directory are **never** automatically copied into `ml/data/raw/` or `ml/data/splits/`.
2. **Manual Review Workflow**:
   - Inspect each image under `artifacts/feedback_staging/{plastic,paper,metal,glass,organic}/` and verify the `user_confirmed_category` in `feedback_manifest.json`.
   - Reject blurry, multi-material, or mislabeled submissions.
   - Move approved single-material images into `ml/data/raw/realwaste/<Category>/`, then re-run:
     ```bash
     python ml/scripts/verify_dataset.py
     python ml/scripts/prepare_dataset.py
     ```
     so perceptual-hash (`dHash`) deduplication and leak-free split generation are enforced before retraining.
