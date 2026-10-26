"""Database management utility for resetting DB, removing demo rows, or seeding real predictions."""
import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "backend"))

from sqlalchemy import delete, select  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.security import save_sanitized_thumbnail, validate_and_sanitize_image  # noqa: E402
from app.core.taxonomy import load_taxonomy  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.ml.inference import classifier_service  # noqa: E402
from app.models.prediction import Prediction  # noqa: E402


def remove_demo_rows() -> int:
    with SessionLocal() as db:
        rows = db.execute(select(Prediction).where(Prediction.is_demo.is_(True))).scalars().all()
        count = len(rows)
        for r in rows:
            if r.thumbnail_path:
                p = Path(settings.THUMBNAIL_DIR) / Path(r.thumbnail_path).name
                if p.exists():
                    p.unlink(missing_ok=True)
            db.delete(r)
        db.commit()
    print(f"[manage_db] Removed {count} demo rows (is_demo=True).")
    return count


def reset_database() -> int:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        rows = db.execute(select(Prediction)).scalars().all()
        count = len(rows)
        for r in rows:
            if r.thumbnail_path:
                p = Path(settings.THUMBNAIL_DIR) / Path(r.thumbnail_path).name
                if p.exists():
                    p.unlink(missing_ok=True)
        db.execute(delete(Prediction))
        db.commit()
    print(f"[manage_db] Reset database: deleted {count} total prediction rows.")
    return count


def seed_real_predictions_from_test_split(per_class: int = 3, is_demo: bool = False) -> int:
    import csv

    Base.metadata.create_all(bind=engine)
    if not classifier_service.ensure_loaded():
        print("[manage_db] Cannot seed predictions: ONNX model is not loaded yet.")
        return 0

    test_csv = BASE_DIR / "ml" / "data" / "splits" / "test.csv"
    if not test_csv.exists():
        print(f"[manage_db] Split manifest not found at {test_csv}")
        return 0

    taxonomy = load_taxonomy(settings.TAXONOMY_YAML_PATH)
    by_cat = {}
    with test_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["label"]
            by_cat.setdefault(cat, [])
            if len(by_cat[cat]) < per_class:
                by_cat[cat].append(row)

    inserted = 0
    with SessionLocal() as db:
        for cat, items in sorted(by_cat.items()):
            for item in items:
                img_path = BASE_DIR / item["path"]
                if not img_path.exists():
                    continue
                raw = img_path.read_bytes()
                val = validate_and_sanitize_image(raw, original_filename=img_path.name)
                inf = classifier_service.predict_sync(val.clean_rgb_image)
                entry = taxonomy.get_entry(inf.top_label)
                thumb_name = save_sanitized_thumbnail(val.clean_rgb_image)

                rec = Prediction(
                    original_filename=val.sanitized_filename,
                    file_sha256=val.file_sha256,
                    detected_item=entry.detected_item,
                    category=entry.category,
                    recyclability_class=entry.recyclability_class,
                    bin=entry.bin,
                    confidence=inf.confidence,
                    confidence_level=inf.confidence_level,
                    status=inf.status,
                    top_k=inf.top_k,
                    quality_warnings=val.quality_warnings,
                    model_version=inf.model_version,
                    inference_ms=inf.inference_ms,
                    image_width=val.width,
                    image_height=val.height,
                    thumbnail_path=thumb_name,
                    is_demo=is_demo,
                )
                db.add(rec)
                inserted += 1
        db.commit()
    print(f"[manage_db] Ran real ONNX inference and inserted {inserted} rows (is_demo={is_demo}).")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="EcoSort ML Database Management CLI")
    parser.add_argument("--remove-demo", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--seed-real", type=int, default=0)
    parser.add_argument("--as-demo", action="store_true")
    args = parser.parse_args()

    if args.reset:
        reset_database()
    if args.remove_demo:
        remove_demo_rows()
    if args.seed_real > 0:
        seed_real_predictions_from_test_split(per_class=args.seed_real, is_demo=args.as_demo)


if __name__ == "__main__":
    main()
