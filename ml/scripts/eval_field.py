"""Evaluate the exported ONNX waste classifier on the held-out real-world field evaluation set (ml/data/field_eval/).

Reports:
- Per-class accuracy, precision, recall, F1, support
- Confusion matrix
- Abstention (uncertain) rate, non-abstained accuracy, and confidently wrong rate
- Slice metrics: 'pile_bulk', 'single_item', 'low_res', and 'metal_pile_crushed'
- Specific fixture outcomes: metal_can_pile_01.jpg, plastic_bottle_grass_01.jpg, plastic_bottle_table_02.jpg
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.core.security import detect_pile_or_multi_item_scene  # noqa: E402
from app.ml.inference import classifier_service  # noqa: E402

FIELD_EVAL_DIR = BASE_DIR / "ml" / "data" / "field_eval"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"
CLASSES = ["glass", "metal", "organic", "paper", "plastic"]


def compute_slice_summary(slice_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not slice_records:
        return {
            "count": 0,
            "accuracy": 0.0,
            "non_abstained_accuracy": 0.0,
            "abstention_rate": 0.0,
            "confidently_wrong_rate": 0.0,
        }
    total = len(slice_records)
    correct = sum(1 for r in slice_records if r["predicted_top1"] == r["true_category"])
    non_abstained = [r for r in slice_records if r["status"] == "ok"]
    non_abs_correct = sum(1 for r in non_abstained if r["predicted_top1"] == r["true_category"])
    conf_wrong = sum(1 for r in non_abstained if r["predicted_top1"] != r["true_category"])
    return {
        "count": total,
        "accuracy": round(correct / total, 4),
        "non_abstained_accuracy": round(non_abs_correct / max(1, len(non_abstained)), 4),
        "abstention_rate": round((total - len(non_abstained)) / total, 4),
        "confidently_wrong_rate": round(conf_wrong / total, 4),
    }


def main() -> None:
    if not classifier_service.ensure_loaded():
        raise RuntimeError("ONNX model is not loaded. Run export_onnx.py first.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    statuses: List[str] = []
    records: List[Dict[str, Any]] = []

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = FIELD_EVAL_DIR / cls_name
        if not cls_dir.exists():
            continue
        files = sorted(
            p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for p in files:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
            is_pile_img, _ = detect_pile_or_multi_item_scene(rgb)
            is_pile_slice = bool(is_pile_img or "pile" in p.name.lower())
            is_low_res = bool(min(w, h) < 224)

            res = classifier_service.predict(rgb)
            pred_idx = CLASSES.index(res.category)
            y_true.append(cls_idx)
            y_pred.append(pred_idx)
            confidences.append(res.confidence)
            statuses.append(res.status)
            records.append(
                {
                    "path": str(p.relative_to(BASE_DIR)),
                    "filename": p.name,
                    "true_category": cls_name,
                    "predicted_top1": res.category,
                    "returned_category": None if res.status == "uncertain" else res.category,
                    "status": res.status,
                    "confidence": res.confidence,
                    "confidence_level": res.confidence_level,
                    "entropy": res.entropy,
                    "is_pile_slice": is_pile_slice,
                    "is_low_res": is_low_res,
                    "width": w,
                    "height": h,
                    "top_k": res.top_k,
                }
            )

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)
    acc = float(accuracy_score(y_true_np, y_pred_np))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=list(range(len(CLASSES))), zero_division=0
    )

    per_class = []
    for idx, cname in enumerate(CLASSES):
        cls_mask = y_true_np == idx
        cls_acc = float(np.mean(y_pred_np[cls_mask] == idx)) if np.sum(cls_mask) > 0 else 0.0
        per_class.append(
            {
                "category": cname,
                "accuracy": round(cls_acc, 4),
                "precision": round(float(p_cls[idx]), 4),
                "recall": round(float(r_cls[idx]), 4),
                "f1": round(float(f1_cls[idx]), 4),
                "support": int(sup_cls[idx]),
            }
        )

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(CLASSES))))
    uncertain_rate = float(np.mean([s == "uncertain" for s in statuses])) if statuses else 0.0
    non_abstained_recs = [r for r in records if r["status"] == "ok"]
    non_abstained_acc = (
        round(sum(1 for r in non_abstained_recs if r["predicted_top1"] == r["true_category"]) / len(non_abstained_recs), 4)
        if non_abstained_recs
        else 0.0
    )
    confidently_wrong_rate = round(
        sum(1 for r in non_abstained_recs if r["predicted_top1"] != r["true_category"]) / max(1, len(records)),
        4,
    )

    # Slice evaluations
    pile_bulk_slice = [r for r in records if r["is_pile_slice"]]
    single_item_slice = [r for r in records if not r["is_pile_slice"]]
    low_res_slice = [r for r in records if r["is_low_res"]]
    metal_pile_slice = [r for r in records if r["true_category"] == "metal" and r["is_pile_slice"]]
    metal_pile_med_high_ok = sum(
        1
        for r in metal_pile_slice
        if r["predicted_top1"] == "metal" and r["status"] == "ok" and r["confidence_level"] in ("MEDIUM", "HIGH")
    )

    slices = {
        "pile_bulk": compute_slice_summary(pile_bulk_slice),
        "single_item": compute_slice_summary(single_item_slice),
        "low_res": compute_slice_summary(low_res_slice),
        "metal_pile_crushed": {
            **compute_slice_summary(metal_pile_slice),
            "classified_as_metal_medium_or_high_rate": round(
                metal_pile_med_high_ok / max(1, len(metal_pile_slice)), 4
            ),
        },
    }

    bottle_grass_rec = next((r for r in records if r["filename"] == "plastic_bottle_grass_01.jpg"), None)
    bottle_table_rec = next((r for r in records if r["filename"] == "plastic_bottle_table_02.jpg"), None)
    can_pile_rec = next((r for r in records if r["filename"] == "metal_can_pile_01.jpg"), None)

    payload = {
        "model_version": classifier_service.model_version,
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "non_abstained_accuracy": non_abstained_acc,
        "confidently_wrong_rate": confidently_wrong_rate,
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "slices": slices,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASSES,
        "plastic_bottle_grass_01": bottle_grass_rec,
        "plastic_bottle_table_02": bottle_table_rec,
        "metal_can_pile_01": can_pile_rec,
    }

    out_json = METRICS_DIR / "field_eval_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Real-World Field Evaluation Report (`artifacts/reports/field_eval_report.md`)",
        "",
        f"- **Model Version:** `{classifier_service.model_version}`",
        f"- **Field Evaluation Samples:** `{len(y_true)}` (`>= 42` per class across 5 classes, strictly disjoint from train/val/test)",
        f"- **Overall Top-1 Accuracy:** `{acc * 100:.2f}%` (`{acc:.4f}`)",
        f"- **Non-Abstained (`status == 'ok'`) Accuracy:** `{non_abstained_acc * 100:.2f}%` (`{non_abstained_acc:.4f}`)",
        f"- **Confidently Wrong Rate:** `{confidently_wrong_rate * 100:.2f}%` (`{confidently_wrong_rate:.4f}`)",
        f"- **Macro-F1 Score:** `{macro_f1:.4f}`",
        f"- **Abstention (`status == 'uncertain'`) Rate:** `{uncertain_rate * 100:.2f}%`",
        "",
        "## 1. Per-Class Field Performance",
        "",
        "| Category | Accuracy | Precision | Recall | F1-Score | Support |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pc in per_class:
        md.append(
            f"| **{pc['category']}** | `{pc['accuracy']:.4f}` | `{pc['precision']:.4f}` | "
            f"`{pc['recall']:.4f}` | `{pc['f1']:.4f}` | `{pc['support']}` |"
        )

    md.extend(
        [
            "",
            "## 2. Field Evaluation Slices (`pile_bulk`, `single_item`, `low_res`, `metal_pile_crushed`)",
            "",
            "| Slice | Count | Accuracy | Non-Abstained Accuracy | Abstention Rate | Confidently Wrong Rate |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s_name, s_data in slices.items():
        md.append(
            f"| **{s_name}** | `{s_data['count']}` | `{s_data['accuracy']:.4f}` | "
            f"`{s_data['non_abstained_accuracy']:.4f}` | `{s_data['abstention_rate']:.4f}` | `{s_data['confidently_wrong_rate']:.4f}` |"
        )

    for title, fix_rec in (
        ("Crushed Metal Can Pile (`metal_can_pile_01.jpg`)", can_pile_rec),
        ("Transparent Plastic Bottle on Grass (`plastic_bottle_grass_01.jpg`)", bottle_grass_rec),
        ("Clear Ribbed Plastic Bottle on Table (`plastic_bottle_table_02.jpg`)", bottle_table_rec),
    ):
        if fix_rec:
            md.extend(
                [
                    "",
                    f"## {title}",
                    "",
                    f"- **Predicted Top-1:** `{fix_rec['predicted_top1']}`",
                    f"- **Status:** `{fix_rec['status']}`",
                    f"- **Returned Category:** `{fix_rec['returned_category']}`",
                    f"- **Confidence:** `{fix_rec['confidence'] * 100:.2f}%` (`{fix_rec['confidence_level']}`)",
                    f"- **Normalized Entropy:** `{fix_rec['entropy']:.4f}`",
                    f"- **Top-3 Candidates:** `{json.dumps(fix_rec['top_k'])}`",
                ]
            )

    (REPORTS_DIR / "field_eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"[eval_field] Samples={len(y_true)} | Accuracy={acc:.4f} | NonAbstainedAcc={non_abstained_acc:.4f} | "
        f"ConfWrong={confidently_wrong_rate:.4f} | UncertainRate={uncertain_rate:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
