"""Export user-confirmed feedback images and labels into a staging folder for human review (Phase D3).

CRITICAL SAFETY INVARIANT:
- Only exports predictions where `corrected_category` is non-null AND `STORE_THUMBNAILS=true` AND a saved thumbnail exists.
- Exports into `artifacts/feedback_staging/{category}/` + `artifacts/feedback_staging/feedback_manifest.json`.
- NEVER auto-merges into `ml/data/raw/` or `ml/data/splits/` or training/analytics ground truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.config import settings  # noqa: E402

DEFAULT_STAGING_DIR = BASE_DIR / "artifacts" / "feedback_staging"


def export_confirmed_feedback(
    db_path: Path | None = None,
    thumb_dir: Path | None = None,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    store_thumbnails_enabled: bool | None = None,
) -> Dict[str, Any]:
    if store_thumbnails_enabled is None:
        store_thumbnails_enabled = bool(settings.STORE_THUMBNAILS)

    if db_path is None:
        raw_url = settings.DATABASE_URL
        db_file = raw_url.replace("sqlite:///", "")
        db_path = Path(db_file)

    if thumb_dir is None:
        thumb_dir = Path(settings.THUMBNAIL_DIR)

    staging_dir.mkdir(parents=True, exist_ok=True)

    exported_records: List[Dict[str, Any]] = []
    skipped_no_consent_or_thumb = 0

    if not store_thumbnails_enabled:
        summary = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "store_thumbnails_enabled": False,
            "exported_count": 0,
            "skipped_count": 0,
            "staging_dir": str(staging_dir),
            "auto_merged_into_training": False,
            "records": [],
            "note": "STORE_THUMBNAILS is disabled; no images exported.",
        }
        (staging_dir / "feedback_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    if not db_path.exists():
        summary = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "store_thumbnails_enabled": True,
            "exported_count": 0,
            "skipped_count": 0,
            "staging_dir": str(staging_dir),
            "auto_merged_into_training": False,
            "records": [],
            "note": f"Database not found at {db_path}.",
        }
        (staging_dir / "feedback_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, original_filename, file_sha256, status, predicted_top1, category,
                   corrected_category, confidence, model_version, thumbnail_path, is_demo, created_at
            FROM predictions
            WHERE corrected_category IS NOT NULL
              AND corrected_category != ''
              AND is_demo = 0
            ORDER BY created_at ASC
            """
        ).fetchall()
    finally:
        conn.close()

    for r in rows:
        thumb_name = r["thumbnail_path"]
        corr_cat = str(r["corrected_category"]).strip().lower()
        if not thumb_name or corr_cat not in {"plastic", "paper", "metal", "glass", "organic"}:
            skipped_no_consent_or_thumb += 1
            continue

        src_thumb = thumb_dir / thumb_name
        if not src_thumb.exists():
            skipped_no_consent_or_thumb += 1
            continue

        cat_dir = staging_dir / corr_cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        dst_name = f"feedback_{r['id'][:8]}_{corr_cat}.jpg"
        dst_path = cat_dir / dst_name
        shutil.copy2(src_thumb, dst_path)

        exported_records.append(
            {
                "prediction_id": r["id"],
                "original_filename": r["original_filename"],
                "file_sha256": r["file_sha256"],
                "model_status": r["status"],
                "model_predicted_top1": r["predicted_top1"],
                "user_confirmed_category": corr_cat,
                "model_confidence": float(r["confidence"]),
                "model_version": r["model_version"],
                "staged_image_path": str(dst_path.relative_to(staging_dir.parent.parent if staging_dir.is_relative_to(BASE_DIR) else staging_dir)),
                "created_at": r["created_at"],
                "review_status": "PENDING_HUMAN_REVIEW",
            }
        )

    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "store_thumbnails_enabled": True,
        "exported_count": len(exported_records),
        "skipped_count": skipped_no_consent_or_thumb,
        "staging_dir": str(staging_dir),
        "auto_merged_into_training": False,
        "records": exported_records,
    }
    (staging_dir / "feedback_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_text = """# Human-in-the-Loop Feedback Staging Directory

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
"""
    (staging_dir / "README.md").write_text(readme_text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export user-confirmed feedback images into staging folder for review.")
    parser.add_argument("--staging-dir", type=str, default=str(DEFAULT_STAGING_DIR))
    args = parser.parse_args()

    res = export_confirmed_feedback(staging_dir=Path(args.staging_dir))
    print(
        f"[export_feedback] Exported={res['exported_count']} | Skipped={res['skipped_count']} | "
        f"Staging={res['staging_dir']} (auto_merged_into_training={res['auto_merged_into_training']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
