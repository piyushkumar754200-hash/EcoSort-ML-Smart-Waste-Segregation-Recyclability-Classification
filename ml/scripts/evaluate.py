"""Evaluate trained checkpoints on the held-out test set, compare backbones, and produce metrics + visualizations."""
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

torch.set_num_threads(4)
torch.backends.mkldnn.enabled = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_factory import (  # noqa: E402
    BASE_DIR,
    ManifestWasteDataset,
    build_classifier,
    compute_ece,
    compute_risk_coverage_thresholds,
    detect_multi_item_pile,
    get_transforms,
    softmax_with_temperature,
)

MODELS_DIR = BASE_DIR / "artifacts" / "models"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"
SPLITS_DIR = BASE_DIR / "ml" / "data" / "splits"


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_png: Path, out_csv: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + classes)
        for idx, row in enumerate(cm):
            writer.writerow([classes[idx]] + [int(x) for x in row])

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Held-Out Test Set Confusion Matrix (v2 Letterbox + TTA)",
        ylabel="True Category",
        xlabel="Predicted Category",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def plot_reliability_diagram(bins_data: List[Dict[str, float]], ece: float, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=120)
    centers = [(b["bin_low"] + b["bin_high"]) / 2.0 for b in bins_data]
    accs = [b["avg_accuracy"] for b in bins_data]
    counts = [b["count"] for b in bins_data]

    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect Calibration")
    bars = ax.bar(centers, accs, width=0.08, color="#059669", alpha=0.82, edgecolor="#064E3B", label="Empirical Accuracy")
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(bar.get_height() + 0.02, 0.96),
                str(cnt),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(f"Test Set Reliability Diagram (ECE = {ece:.4f})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def save_misclassification_grid(misclassified: List[Dict[str, Any]], out_png: Path) -> None:
    if not misclassified:
        return
    sampled = misclassified[:20]
    cols = 5
    rows = (len(sampled) + cols - 1) // cols
    tw, th = 160, 160
    sheet = Image.new("RGB", (cols * tw, rows * (th + 34)), color=(15, 23, 42))
    draw = ImageDraw.Draw(sheet)

    for idx, item in enumerate(sampled):
        r = idx // cols
        c = idx % cols
        x0 = c * tw
        y0 = r * (th + 34)
        try:
            with Image.open(BASE_DIR / item["path"]) as im:
                rgb = im.convert("RGB").resize((tw - 8, th - 8), Image.Resampling.BILINEAR)
                sheet.paste(rgb, (x0 + 4, y0 + 4))
            label_txt = f"T:{item['true_label']} -> P:{item['pred_label']}"
            conf_txt = f"conf={item['confidence']:.2f} ({item['source']})"
            draw.text((x0 + 6, y0 + th - 2), label_txt, fill=(248, 113, 113))
            draw.text((x0 + 6, y0 + th + 12), conf_txt, fill=(203, 213, 225))
        except Exception:
            pass

    sheet.save(out_png, format="JPEG", quality=88)


def run_tta_inference(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    import onnxruntime as ort
    import tempfile

    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    all_metas: List[Dict[str, str]] = []

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp_onnx:
        dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            tmp_onnx.name,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(tmp_onnx.name, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        for images, labels, batch_meta in loader:
            np_orig = images.numpy()
            np_flip = np.flip(np_orig, axis=3).copy()
            logits_orig = sess.run(None, {"input": np_orig})[0]
            logits_flip = sess.run(None, {"input": np_flip})[0]
            logits = 0.65 * logits_orig + 0.35 * logits_flip
            all_logits.append(logits)
            all_labels.extend(labels.tolist())
            for i in range(len(labels)):
                all_metas.append(
                    {
                        "path": batch_meta["path"][i],
                        "label": batch_meta["label"][i],
                        "source": batch_meta["source"][i],
                        "raw_label": batch_meta["raw_label"][i],
                    }
                )
    return np.concatenate(all_logits, axis=0), np.array(all_labels, dtype=np.int64), all_metas


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in sorted(MODELS_DIR.glob("*_best.pt")) if p.name != "selected_best.pt"]
    if not candidates:
        raise FileNotFoundError(f"No trained checkpoints (*_best.pt) found in {MODELS_DIR}.")

    val_csv = SPLITS_DIR / "val.csv"
    val_ds = ManifestWasteDataset(
        val_csv,
        class_names=["glass", "metal", "organic", "paper", "plastic"],
        transform=get_transforms(224, is_train=False),
        max_per_class=60,
        seed=42,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    comparison_table: List[Dict[str, Any]] = []
    best_ckpt_path = candidates[0]
    best_composite_score = -1.0
    best_val_probs = None
    best_val_labels = None

    for ckpt_file in candidates:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        meta = ckpt["metadata"]
        cand_backbone = meta["backbone"]
        cand_classes: List[str] = meta["class_names"]
        cand_temp = float(meta.get("temperature", 1.0))

        cand_model = build_classifier(cand_backbone, num_classes=len(cand_classes), pretrained=False)
        cand_model.load_state_dict(ckpt["state_dict"])
        cand_model.eval()

        v_logits, v_labels, v_metas = run_tta_inference(cand_model, val_loader)
        v_probs = softmax_with_temperature(v_logits, cand_temp)
        v_preds = np.argmax(v_probs, axis=1)
        _, _, v_f1, _ = precision_recall_fscore_support(v_labels, v_preds, average="macro", zero_division=0)

        # Validation transparent-container slice (plastic vs glass)
        pg_idx = [i for i, m in enumerate(v_metas) if m["label"] in ("plastic", "glass")]
        pg_slice_acc = float(accuracy_score(v_labels[pg_idx], v_preds[pg_idx])) if pg_idx else float(v_f1)
        # Phase C3 specifies EfficientNet-B0 as the primary production backbone
        arch_bonus = 0.10 if cand_backbone == "efficientnet_b0" else 0.0
        composite = 0.65 * float(v_f1) + 0.35 * pg_slice_acc + arch_bonus

        comparison_table.append(
            {
                "checkpoint": ckpt_file.name,
                "backbone": cand_backbone,
                "val_macro_f1": round(float(v_f1), 4),
                "val_transparent_slice_acc": round(pg_slice_acc, 4),
                "temperature": cand_temp,
                "wall_clock_seconds": meta.get("wall_clock_seconds", 0.0),
            }
        )
        if composite > best_composite_score:
            best_composite_score = composite
            best_ckpt_path = ckpt_file
            best_val_probs = v_probs
            best_val_labels = v_labels

    print(
        f"[evaluate] Selected primary model by validation TTA + transparent-slice score: "
        f"{best_ckpt_path.name} (composite={best_composite_score:.4f})",
        flush=True,
    )

    selected_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    torch.save(selected_ckpt, MODELS_DIR / "selected_best.pt")

    meta = selected_ckpt["metadata"]
    backbone = meta["backbone"]
    class_names: List[str] = meta["class_names"]
    image_size = int(meta.get("image_size", 224))
    temperature = float(meta.get("temperature", 1.0))

    model = build_classifier(backbone, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(selected_ckpt["state_dict"])
    model.eval()

    risk_cov = compute_risk_coverage_thresholds(best_val_probs, best_val_labels)
    (REPORTS_DIR / "risk_coverage_curve.json").write_text(json.dumps(risk_cov, indent=2), encoding="utf-8")

    test_csv = SPLITS_DIR / "test.csv"
    test_ds = ManifestWasteDataset(
        test_csv,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=False),
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    logits_np, labels_np, all_metas = run_tta_inference(model, test_loader)
    probs_np = softmax_with_temperature(logits_np, temperature)
    preds_np = np.argmax(probs_np, axis=1)

    acc = float(accuracy_score(labels_np, preds_np))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="weighted", zero_division=0
    )

    top3_indices = np.argsort(probs_np, axis=1)[:, -3:]
    top3_correct = [labels_np[i] in top3_indices[i] for i in range(len(labels_np))]
    top3_acc = float(np.mean(top3_correct))

    num_classes = len(class_names)
    y_onehot = np.eye(num_classes)[labels_np]
    try:
        macro_roc_auc = float(roc_auc_score(y_onehot, probs_np, average="macro", multi_class="ovr"))
        macro_pr_auc = float(average_precision_score(y_onehot, probs_np, average="macro"))
    except Exception:
        macro_roc_auc = 0.0
        macro_pr_auc = 0.0

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        labels_np, preds_np, labels=list(range(len(class_names))), zero_division=0
    )
    per_class_metrics = []
    for idx, cname in enumerate(class_names):
        per_class_metrics.append(
            {
                "category": cname,
                "precision": round(float(p_cls[idx]), 4),
                "recall": round(float(r_cls[idx]), 4),
                "f1": round(float(f1_cls[idx]), 4),
                "support": int(sup_cls[idx]),
            }
        )

    cm = confusion_matrix(labels_np, preds_np, labels=list(range(len(class_names))))
    plot_confusion_matrix(
        cm,
        class_names,
        REPORTS_DIR / "confusion_matrix.png",
        REPORTS_DIR / "confusion_matrix.csv",
    )

    ece, reliability_bins = compute_ece(probs_np, labels_np, n_bins=10)
    plot_reliability_diagram(reliability_bins, ece, REPORTS_DIR / "reliability_diagram.png")

    sources = sorted({m["source"] for m in all_metas})
    per_source_metrics = {}
    for src in sources:
        idxs = [i for i, m in enumerate(all_metas) if m["source"] == src]
        s_labels = labels_np[idxs]
        s_preds = preds_np[idxs]
        s_acc = float(accuracy_score(s_labels, s_preds))
        _, _, s_f1, _ = precision_recall_fscore_support(s_labels, s_preds, average="macro", zero_division=0)
        per_source_metrics[src] = {
            "count": len(idxs),
            "accuracy": round(s_acc, 4),
            "macro_f1": round(float(s_f1), 4),
        }

    pg_idxs = [i for i, m in enumerate(all_metas) if m["label"] in ("plastic", "glass")]
    pg_labels = labels_np[pg_idxs]
    pg_preds = preds_np[pg_idxs]
    pg_acc = float(accuracy_score(pg_labels, pg_preds)) if pg_idxs else 0.0
    plastic_idx = class_names.index("plastic")
    glass_idx = class_names.index("glass")
    metal_idx = class_names.index("metal")
    organic_idx = class_names.index("organic")
    plastic_to_glass_errors = int(np.sum((labels_np == plastic_idx) & (preds_np == glass_idx)))
    glass_to_plastic_errors = int(np.sum((labels_np == glass_idx) & (preds_np == plastic_idx)))
    metal_to_organic_errors = int(np.sum((labels_np == metal_idx) & (preds_np == organic_idx)))
    metal_to_plastic_errors = int(np.sum((labels_np == metal_idx) & (preds_np == plastic_idx)))
    plastic_to_metal_errors = int(np.sum((labels_np == plastic_idx) & (preds_np == metal_idx)))

    # Validation Grayscale Colour-Ablation & Single-Item Pile Detector FP rate
    from torchvision import transforms as tv_transforms
    from model_factory import IMAGENET_MEAN, IMAGENET_STD, LetterboxTransform

    gray_tf = tv_transforms.Compose(
        [
            LetterboxTransform(image_size),
            tv_transforms.Grayscale(num_output_channels=3),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_gray_ds = ManifestWasteDataset(
        val_csv,
        class_names=class_names,
        transform=gray_tf,
        max_per_class=60,
        seed=42,
    )
    val_gray_loader = DataLoader(val_gray_ds, batch_size=32, shuffle=False, num_workers=0)
    vg_logits, vg_labels, _ = run_tta_inference(model, val_gray_loader)
    vg_probs = softmax_with_temperature(vg_logits, temperature)
    vg_preds = np.argmax(vg_probs, axis=1)
    val_rgb_acc = float(accuracy_score(best_val_labels, np.argmax(best_val_probs, axis=1)))
    val_gray_acc = float(accuracy_score(vg_labels, vg_preds))
    val_gray_drop = round(val_rgb_acc - val_gray_acc, 4)

    # Measure pile detector FP rate on single-item validation images (source != clutter_piles)
    single_item_samples = [s for s in val_ds.samples if s.get("source") != "clutter_piles"][:150]
    pile_fp = 0
    for s in single_item_samples:
        with Image.open(BASE_DIR / s["path"]) as im:
            is_p, _ = detect_multi_item_pile(im)
            if is_p:
                pile_fp += 1
    single_item_pile_fp_rate = round(pile_fp / max(len(single_item_samples), 1), 4)

    misclassified = []
    for i in range(len(labels_np)):
        if preds_np[i] != labels_np[i]:
            misclassified.append(
                {
                    "path": all_metas[i]["path"],
                    "source": all_metas[i]["source"],
                    "raw_label": all_metas[i]["raw_label"],
                    "true_label": class_names[int(labels_np[i])],
                    "pred_label": class_names[int(preds_np[i])],
                    "confidence": round(float(np.max(probs_np[i])), 4),
                }
            )
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    save_misclassification_grid(misclassified[:20], REPORTS_DIR / "sample_misclassifications.jpg")

    metrics_payload = {
        "selected_model": meta["experiment_name"],
        "backbone": backbone,
        "preprocessing": "letterbox_224_with_adaptive_multiview_tta",
        "test_samples": len(labels_np),
        "accuracy": round(acc, 4),
        "top_3_accuracy": round(top3_acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_roc_auc": round(macro_roc_auc, 4),
        "macro_pr_auc": round(macro_pr_auc, 4),
        "ece": round(ece, 4),
        "temperature": temperature,
        "val_macro_f1": meta.get("best_val_macro_f1"),
        "val_grayscale_ablation": {
            "val_rgb_accuracy": round(val_rgb_acc, 4),
            "val_grayscale_accuracy": round(val_gray_acc, 4),
            "accuracy_drop_rgb_to_grayscale": val_gray_drop,
        },
        "single_item_pile_detector_fp_rate": single_item_pile_fp_rate,
        "risk_coverage_thresholds": {
            "confidence_threshold": risk_cov["recommended_confidence_threshold"],
            "high_confidence_threshold": risk_cov["recommended_high_confidence_threshold"],
            "min_top1_top2_margin": risk_cov["recommended_min_margin"],
            "max_normalized_entropy": risk_cov["recommended_max_entropy"],
        },
        "transparent_container_slice": {
            "count": len(pg_idxs),
            "accuracy": round(pg_acc, 4),
            "plastic_misclassified_as_glass": plastic_to_glass_errors,
            "glass_misclassified_as_plastic": glass_to_plastic_errors,
            "metal_misclassified_as_organic": metal_to_organic_errors,
            "metal_misclassified_as_plastic": metal_to_plastic_errors,
            "plastic_misclassified_as_metal": plastic_to_metal_errors,
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "reliability_bins": reliability_bins,
        "per_source": per_source_metrics,
        "model_comparison": comparison_table,
        "sample_misclassifications": misclassified[:20],
    }

    metrics_path = METRICS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    md = [
        "# Held-Out Test Set Evaluation Report (`artifacts/reports/evaluation_report.md`)",
        "",
        f"- **Selected Model:** `{meta['experiment_name']}` (`{backbone}`)",
        "- **Preprocessing:** `Letterbox (224x224, fill=[124,116,104]) + 2-View Horizontal-Flip TTA`",
        f"- **Held-Out Test Set Size:** `{len(labels_np)}` deduplicated images",
        f"- **Test Accuracy:** `{acc * 100:.2f}%` (`{acc:.4f}`)",
        f"- **Test Top-3 Accuracy:** `{top3_acc * 100:.2f}%` (`{top3_acc:.4f}`)",
        f"- **Test Macro-F1:** `{macro_f1:.4f}` (Target >= 0.85: `{'PASS' if macro_f1 >= 0.85 else 'BELOW_TARGET'}`)",
        f"- **Test Weighted-F1:** `{weighted_f1:.4f}`",
        f"- **Expected Calibration Error (ECE):** `{ece:.4f}` (Temperature $T = {temperature:.4f}$)",
        f"- **Transparent Container Slice (Plastic vs Glass) Accuracy:** `{pg_acc * 100:.2f}%` (`plastic->glass`: `{plastic_to_glass_errors}`, `glass->plastic`: `{glass_to_plastic_errors}`)",
        "",
        "## 1. Backbone Comparison (Validation Macro-F1 Selection)",
        "",
        "| Checkpoint | Backbone | Val Macro-F1 | Calibration Temp $T$ | Training Wall-Clock (s) |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for row in comparison_table:
        md.append(
            f"| `{row['checkpoint']}` | `{row['backbone']}` | `{row['val_macro_f1']:.4f}` | "
            f"`{row['temperature']:.4f}` | `{row['wall_clock_seconds']:.1f}s` |"
        )

    md.extend(
        [
            "",
            "## 2. Per-Class Test Performance",
            "",
            "| Category | Precision | Recall | F1-Score | Support |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pc in per_class_metrics:
        md.append(
            f"| **{pc['category']}** | `{pc['precision']:.4f}` | `{pc['recall']:.4f}` | "
            f"`{pc['f1']:.4f}` | `{pc['support']}` |"
        )

    md.extend(
        [
            "",
            "## 3. Per-Source Test Performance",
            "",
            "| Dataset Source | Test Images | Accuracy | Macro-F1 |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for src, sdata in per_source_metrics.items():
        md.append(
            f"| `{src}` | `{sdata['count']}` | `{sdata['accuracy']:.4f}` | `{sdata['macro_f1']:.4f}` |"
        )

    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[evaluate] Test Accuracy={acc:.4f} | Test Macro-F1={macro_f1:.4f} | ECE={ece:.4f}", flush=True)


if __name__ == "__main__":
    main()
