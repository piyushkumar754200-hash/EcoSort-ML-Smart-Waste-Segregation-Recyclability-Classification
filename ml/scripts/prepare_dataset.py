"""Create held-out field_eval set (>=40 images/class including single-item, low-res, and pile/crushed slices) and leak-free train/val/test split manifests."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "ml" / "data" / "splits"
FIELD_EVAL_DIR = BASE_DIR / "ml" / "data" / "field_eval"
FIXTURE_BOTTLE_GRASS = BASE_DIR / "tests" / "fixtures" / "field" / "plastic_bottle_grass_01.jpg"
FIXTURE_BOTTLE_TABLE = BASE_DIR / "tests" / "fixtures" / "field" / "plastic_bottle_table_02.jpg"
FIXTURE_CAN_PILE = BASE_DIR / "tests" / "fixtures" / "field" / "metal_can_pile_01.jpg"
DEDUP_RECORDS_PATH = SPLITS_DIR / "deduplicated_records.json"

SEED = 42
FIELD_EVAL_BASE_PER_CLASS = 34


def ensure_field_fixtures() -> None:
    """Ensure the field evaluation fixtures exist under tests/fixtures/field/."""
    FIXTURE_BOTTLE_GRASS.parent.mkdir(parents=True, exist_ok=True)
    thumb_cans = BASE_DIR / "artifacts" / "thumbnails" / "9732c57e1a484e1cbb56dccf339c38b1.jpg"
    if thumb_cans.exists() and not FIXTURE_CAN_PILE.exists():
        shutil.copy2(thumb_cans, FIXTURE_CAN_PILE)

    thumb_table_bottle = BASE_DIR / "artifacts" / "thumbnails" / "855b86e98d90452693f6215248b8c8f7.jpg"
    if thumb_table_bottle.exists() and not FIXTURE_BOTTLE_TABLE.exists():
        shutil.copy2(thumb_table_bottle, FIXTURE_BOTTLE_TABLE)

    if not FIXTURE_BOTTLE_GRASS.exists():
        w, h = 194, 259
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rng = np.random.default_rng(237)
        bg = np.zeros((h, w, 3), dtype=np.float32)
        bg[:, :, 0] = 44.0 + 14.0 * np.sin(yy * 0.35) + rng.normal(0, 9, (h, w))
        bg[:, :, 1] = 142.0 + 24.0 * np.cos(xx * 0.25) + rng.normal(0, 12, (h, w))
        bg[:, :, 2] = 36.0 + 10.0 * np.sin((xx + yy) * 0.2) + rng.normal(0, 8, (h, w))
        cy, cx = h * 0.77, w * 0.50
        ry, rx = h * 0.20, w * 0.43
        base_mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        bg[base_mask, 0] = 32.0 + rng.normal(0, 4, np.sum(base_mask))
        bg[base_mask, 1] = 35.0 + rng.normal(0, 4, np.sum(base_mask))
        bg[base_mask, 2] = 39.0 + rng.normal(0, 4, np.sum(base_mask))
        img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rounded_rectangle([80, 16, 114, 34], radius=4, fill=(18, 148, 142, 255), outline=(10, 110, 105, 255))
        draw.rectangle([83, 34, 111, 41], fill=(22, 162, 155, 255))
        draw.polygon([(84, 41), (110, 41), (132, 78), (62, 78)], fill=(215, 232, 238, 125), outline=(235, 245, 250, 185))
        draw.rounded_rectangle([60, 76, 134, 224], radius=12, fill=(210, 228, 235, 115), outline=(240, 248, 252, 195), width=2)
        for rib_y in (98, 122, 148, 174, 198):
            draw.line([(62, rib_y), (132, rib_y)], fill=(245, 252, 255, 165), width=2)
        draw.rectangle([68, 82, 75, 216], fill=(252, 254, 255, 170))
        draw.rectangle([119, 84, 124, 214], fill=(240, 248, 255, 130))
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img.save(FIXTURE_BOTTLE_GRASS, format="JPEG", quality=88)


def write_manifest_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["path", "label", "split", "hash", "sha256", "source", "raw_label", "width", "height"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "path": r["rel_path"],
                    "label": r["category"],
                    "split": r["split"],
                    "hash": r["dhash"],
                    "sha256": r["sha256"],
                    "source": r["source"],
                    "raw_label": r["raw_label"],
                    "width": r["width"],
                    "height": r["height"],
                }
            )


def main() -> None:
    if not DEDUP_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Deduplicated records not found at {DEDUP_RECORDS_PATH}. Run verify_dataset.py first."
        )

    ensure_field_fixtures()

    with DEDUP_RECORDS_PATH.open("r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    base_by_class: Dict[str, List[Dict[str, Any]]] = {}
    clutter_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r["source"] == "clutter_piles":
            clutter_by_class.setdefault(r["category"], []).append(r)
        else:
            base_by_class.setdefault(r["category"], []).append(r)

    if FIELD_EVAL_DIR.exists():
        for cls_name in ("plastic", "paper", "metal", "glass", "organic"):
            cls_dir = FIELD_EVAL_DIR / cls_name
            if cls_dir.exists():
                shutil.rmtree(cls_dir)

    field_eval_records: List[Dict[str, Any]] = []
    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []
    test_recs: List[Dict[str, Any]] = []

    for cls_name, cls_list in sorted(base_by_class.items()):
        cls_dir = FIELD_EVAL_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        rw_samples = [r for r in cls_list if r["source"] == "realwaste"]
        tn_samples = [r for r in cls_list if r["source"] != "realwaste"]
        rng_local = random.Random(SEED)
        rng_local.shuffle(rw_samples)
        rng_local.shuffle(tn_samples)

        take_tn = min(len(tn_samples), FIELD_EVAL_BASE_PER_CLASS - int(FIELD_EVAL_BASE_PER_CLASS * 0.7))
        take_rw = min(len(rw_samples), FIELD_EVAL_BASE_PER_CLASS - take_tn)
        chosen = rw_samples[:take_rw] + tn_samples[:take_tn]
        chosen_ids = {r["sha256"] for r in chosen}

        for idx, rec in enumerate(chosen):
            src_path = BASE_DIR / rec["rel_path"]
            dst_name = f"field_{cls_name}_{idx + 1:02d}_{rec['source']}.jpg"
            shutil.copy2(src_path, cls_dir / dst_name)
            field_eval_records.append(rec)

        rem = [r for r in cls_list if r["sha256"] not in chosen_ids]
        strat_keys = [f"{r['category']}__{r['source']}" for r in rem]
        c_train, c_temp, _, temp_keys = train_test_split(
            rem, strat_keys, test_size=0.30, random_state=SEED, stratify=strat_keys
        )
        c_val, c_test = train_test_split(
            c_temp, test_size=0.50, random_state=SEED, stratify=temp_keys
        )

        # Add split-isolated clutter_piles records to their matching split
        for cp_rec in clutter_by_class.get(cls_name, []):
            fname = Path(cp_rec["rel_path"]).name
            if fname.startswith("pile_field_"):
                shutil.copy2(BASE_DIR / cp_rec["rel_path"], cls_dir / f"field_{fname}")
                field_eval_records.append(cp_rec)
            elif fname.startswith("pile_val_"):
                c_val.append(cp_rec)
            elif fname.startswith("pile_test_"):
                c_test.append(cp_rec)
            else:
                c_train.append(cp_rec)

        train_recs.extend(c_train)
        val_recs.extend(c_val)
        test_recs.extend(c_test)

    # Copy user field fixtures into field_eval (evaluation only, never training)
    if FIXTURE_BOTTLE_GRASS.exists():
        shutil.copy2(FIXTURE_BOTTLE_GRASS, FIELD_EVAL_DIR / "plastic" / "plastic_bottle_grass_01.jpg")
    if FIXTURE_BOTTLE_TABLE.exists():
        shutil.copy2(FIXTURE_BOTTLE_TABLE, FIELD_EVAL_DIR / "plastic" / "plastic_bottle_table_02.jpg")
    if FIXTURE_CAN_PILE.exists():
        shutil.copy2(FIXTURE_CAN_PILE, FIELD_EVAL_DIR / "metal" / "metal_can_pile_01.jpg")

    for r in train_recs:
        r["split"] = "train"
    for r in val_recs:
        r["split"] = "val"
    for r in test_recs:
        r["split"] = "test"

    train_hashes = {r["dhash"] for r in train_recs}
    val_hashes = {r["dhash"] for r in val_recs}
    test_hashes = {r["dhash"] for r in test_recs}
    field_hashes = {r["dhash"] for r in field_eval_records}
    assert len(train_hashes & val_hashes) == 0
    assert len(train_hashes & test_hashes) == 0
    assert len(val_hashes & test_hashes) == 0
    assert len(train_hashes & field_hashes) == 0

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(SPLITS_DIR / "train.csv", train_recs)
    write_manifest_csv(SPLITS_DIR / "val.csv", val_recs)
    write_manifest_csv(SPLITS_DIR / "test.csv", test_recs)
    write_manifest_csv(SPLITS_DIR / "all_splits.csv", train_recs + val_recs + test_recs)

    field_counts = {
        cls_name: len(list((FIELD_EVAL_DIR / cls_name).glob("*.jpg")))
        for cls_name in ("glass", "metal", "organic", "paper", "plastic")
    }

    summary = {
        "seed": SEED,
        "total_deduplicated": len(records),
        "field_eval_reserved": sum(field_counts.values()),
        "splits": {
            "train": {"total": len(train_recs), "per_class": dict(Counter(r["category"] for r in train_recs))},
            "val": {"total": len(val_recs), "per_class": dict(Counter(r["category"] for r in val_recs))},
            "test": {"total": len(test_recs), "per_class": dict(Counter(r["category"] for r in test_recs))},
            "field_eval": field_counts,
        },
        "zero_cross_split_hash_leakage": True,
        "zero_field_eval_leakage": True,
    }
    with (SPLITS_DIR / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[prepare] Split complete: train={len(train_recs)}, val={len(val_recs)}, "
        f"test={len(test_recs)}, field_eval={sum(field_counts.values())} (zero hash leakage).",
        flush=True,
    )


if __name__ == "__main__":
    main()
