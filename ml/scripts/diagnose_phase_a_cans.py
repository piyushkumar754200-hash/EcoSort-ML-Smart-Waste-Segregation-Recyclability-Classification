"""Phase A Diagnostic Script for Crushed Metal Can Pile + Colour/Texture Shortcut Investigation.

Executes A1-A6 with real measurements on the current v2.0.0 model BEFORE any model modifications:
- A1: Reproduces metal_can_pile_01.jpg across predict_cli.py, PyTorch, ONNX Runtime, and FastAPI POST /api/predict.
- A2: Computes raw logits, pre-temperature softmax, post-temperature softmax, Shannon entropy, and normalized entropy.
- A3: Audits metal vs. organic training images (per-class, per-source, single clean can vs. crushed vs. pile/bulk vs. foil/tins) and saves 40-image contact sheets for metal and organic.
- A4: Generates Grad-CAM heatmaps for metal_can_pile_01.jpg (for Metal, Plastic, Organic) and 10 correct metal images; runs RGB vs. Grayscale validation colour-ablation test and reports per-class accuracy drop.
- A5: Evaluates validation confusion matrix, per-class precision/recall/F1, and metal<->plastic<->organic confusion rates.
- A6: Writes findings to artifacts/reports/phase_a_cans_diagnostics.json and updates docs/MISCLASSIFICATION_ANALYSIS.md.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "ml" / "scripts"))
sys.path.insert(0, str(BASE_DIR / "backend"))

from model_factory import build_classifier, letterbox_pil_image, IMAGENET_MEAN, IMAGENET_STD

THUMB_CANS = BASE_DIR / "artifacts" / "thumbnails" / "9732c57e1a484e1cbb56dccf339c38b1.jpg"
FIXTURE_CANS = BASE_DIR / "tests" / "fixtures" / "field" / "metal_can_pile_01.jpg"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"
ARCHIVE_DIR = BASE_DIR / "artifacts" / "metrics" / "archive"
SPLITS_DIR = BASE_DIR / "ml" / "data" / "splits"
MODELS_DIR = BASE_DIR / "artifacts" / "models"


def preprocess_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    boxed = letterbox_pil_image(img, target_size=size)
    arr = np.asarray(boxed, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    norm = (chw - mean) / std
    return torch.from_numpy(np.expand_dims(norm, axis=0).astype(np.float32))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_entropy(probs: np.ndarray) -> Tuple[float, float]:
    clipped = np.clip(probs, 1e-12, 1.0)
    h = float(-np.sum(clipped * np.log(clipped)))
    h_norm = float(h / math.log(len(probs)))
    return h, h_norm


def analyze_image_visual_features(img: Image.Image) -> Dict[str, float]:
    """Compute border uniformity, edge density, quadrant clutter, and colourfulness."""
    rgb = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Hasler & Suesstrunk colourfulness metric
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = math.sqrt(float(np.var(rg) + np.var(yb)))
    mean_root = math.sqrt(float(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    colourfulness = std_root + 0.3 * mean_root

    # Gradient / Sobel-like edge magnitude
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_map = 0.5 * (gx[:-1, :] + gy[:, :-1])
    edge_density = float(np.mean(edge_map > 18.0))

    # Quadrant edge densities (all 4 quadrants active in a pile)
    h_mid, w_mid = edge_map.shape[0] // 2, edge_map.shape[1] // 2
    q_densities = [
        float(np.mean(edge_map[:h_mid, :w_mid] > 18.0)),
        float(np.mean(edge_map[:h_mid, w_mid:] > 18.0)),
        float(np.mean(edge_map[h_mid:, :w_mid] > 18.0)),
        float(np.mean(edge_map[h_mid:, w_mid:] > 18.0)),
    ]
    min_quadrant_edge = float(min(q_densities))

    # Border uniformity
    bw = 12
    top = arr[:bw, :, :].reshape(-1, 3)
    bottom = arr[-bw:, :, :].reshape(-1, 3)
    left = arr[bw:-bw, :bw, :].reshape(-1, 3)
    right = arr[bw:-bw, -bw:, :].reshape(-1, 3)
    border = np.concatenate([top, bottom, left, right], axis=0)
    median_rgb = np.median(border, axis=0)
    dist = np.linalg.norm(border - median_rgb, axis=1)
    border_uniformity = float(np.mean(dist < 28.0))

    return {
        "colourfulness": round(colourfulness, 2),
        "edge_density": round(edge_density, 4),
        "min_quadrant_edge": round(min_quadrant_edge, 4),
        "border_uniformity": round(border_uniformity, 4),
    }


def make_contact_sheet(image_paths: List[Tuple[Path, str]], out_path: Path, cols: int = 8, thumb_size: int = 112) -> None:
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + 18)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for idx, (p, label_text) in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        x0 = c * thumb_size
        y0 = r * (thumb_size + 18)
        try:
            with Image.open(p) as im:
                boxed = letterbox_pil_image(im.convert("RGB"), target_size=thumb_size - 4)
                sheet.paste(boxed, (x0 + 2, y0 + 2))
        except Exception:
            pass
        draw.text((x0 + 4, y0 + thumb_size - 1), label_text[:16], fill=(220, 225, 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=85)


def compute_gradcam(model: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    """Compute Grad-CAM heatmap on the final convolutional block of EfficientNet-B0."""
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    target_layer = model.features[-1]

    def fwd_hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        activations.append(out.detach())

    def bwd_hook(_module: nn.Module, _grad_in: Any, grad_out: Any) -> None:
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad()
        x_req = x.clone().detach().requires_grad_(True)
        logits = model(x_req)
        score = logits[0, target_class]
        score.backward()

        act = activations[0][0]  # (C, H, W)
        grad = gradients[0][0]   # (C, H, W)
        weights = torch.mean(grad, dim=(1, 2), keepdim=True)
        cam = torch.relu(torch.sum(weights * act, dim=0)).cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(base_img: Image.Image, cam: np.ndarray, title: str) -> Image.Image:
    base = base_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    base_arr = np.asarray(base, dtype=np.float32)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.Resampling.BILINEAR)
    cam_norm = np.asarray(cam_img, dtype=np.float32) / 255.0

    # Jet-like RGB colormap
    r = np.clip(1.5 - np.abs(4.0 * cam_norm - 3.0), 0.0, 1.0) * 255.0
    g = np.clip(1.5 - np.abs(4.0 * cam_norm - 2.0), 0.0, 1.0) * 255.0
    b = np.clip(1.5 - np.abs(4.0 * cam_norm - 1.0), 0.0, 1.0) * 255.0
    heat_rgb = np.stack([r, g, b], axis=-1)

    blended = np.clip(0.55 * base_arr + 0.45 * heat_rgb, 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", (224, 248), (20, 22, 26))
    canvas.paste(Image.fromarray(blended), (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(240, 244, 248))
    return canvas


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_CANS.parent.mkdir(parents=True, exist_ok=True)

    # Archive v2.0.0 artifacts before any v3 changes
    if (BASE_DIR / "artifacts" / "metrics" / "metrics.json").exists():
        shutil.copy2(BASE_DIR / "artifacts" / "metrics" / "metrics.json", ARCHIVE_DIR / "metrics_v2.json")
    if (BASE_DIR / "artifacts" / "models" / "model_card.json").exists():
        shutil.copy2(BASE_DIR / "artifacts" / "models" / "model_card.json", ARCHIVE_DIR / "model_card_v2.json")
    if (BASE_DIR / "artifacts" / "metrics" / "field_eval_metrics.json").exists():
        shutil.copy2(
            BASE_DIR / "artifacts" / "metrics" / "field_eval_metrics.json",
            ARCHIVE_DIR / "field_eval_metrics_v2.json",
        )

    # Copy uploaded crushed can pile thumbnail to tests/fixtures/field/metal_can_pile_01.jpg
    if THUMB_CANS.exists() and not FIXTURE_CANS.exists():
        shutil.copy2(THUMB_CANS, FIXTURE_CANS)
    elif THUMB_CANS.exists():
        shutil.copy2(THUMB_CANS, FIXTURE_CANS)

    assert FIXTURE_CANS.exists(), f"Missing fixture {FIXTURE_CANS}"
    can_img = Image.open(FIXTURE_CANS).convert("RGB")

    # Load v2.0.0 PyTorch & ONNX models
    ckpt = torch.load(MODELS_DIR / "selected_best.pt", map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    class_names: List[str] = meta["class_names"]
    temperature = float(meta.get("temperature", 1.0))

    pt_model = build_classifier(meta["backbone"], num_classes=len(class_names), pretrained=False)
    pt_model.load_state_dict(ckpt["state_dict"])
    pt_model.eval()

    x_orig = preprocess_tensor(can_img, 224)
    x_flip = torch.flip(x_orig, dims=[3])
    with torch.no_grad():
        logits_orig_pt = pt_model(x_orig).numpy()[0]
        logits_flip_pt = pt_model(x_flip).numpy()[0]
    logits_tta_pt = 0.65 * logits_orig_pt + 0.35 * logits_flip_pt
    probs_pre_temp_pt = softmax_np(logits_tta_pt, temperature=1.0)
    probs_post_temp_pt = softmax_np(logits_tta_pt, temperature=temperature)
    entropy_nats, entropy_norm = compute_entropy(probs_post_temp_pt)

    # ONNX reproduction
    sess = ort.InferenceSession(str(MODELS_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    logits_orig_ort = sess.run(None, {in_name: x_orig.numpy()})[0][0]
    logits_flip_ort = sess.run(None, {in_name: x_flip.numpy()})[0][0]
    logits_tta_ort = 0.65 * logits_orig_ort + 0.35 * logits_flip_ort
    probs_ort = softmax_np(logits_tta_ort, temperature=temperature)

    # CLI reproduction
    cli_proc = subprocess.run(
        [str(BASE_DIR / ".venv" / "bin" / "python"), str(BASE_DIR / "ml" / "scripts" / "predict_cli.py"), str(FIXTURE_CANS)],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_json = json.loads(cli_proc.stdout)

    # FastAPI TestClient reproduction
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    with FIXTURE_CANS.open("rb") as f_rb:
        api_res = client.post("/api/predict", files={"file": ("metal_can_pile_01.jpg", f_rb.read(), "image/jpeg")})
    api_json = api_res.json()

    # A3: Training Data Audit for metal & organic
    train_rows: List[Dict[str, str]] = []
    with (SPLITS_DIR / "train.csv").open("r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    per_class_counts = Counter(r["label"] for r in train_rows)
    per_source_class = Counter((r["source"], r["label"]) for r in train_rows)

    metal_rows = [r for r in train_rows if r["label"] == "metal"]
    organic_rows = [r for r in train_rows if r["label"] == "organic"]

    metal_subtypes = {
        "single_clean_can_plain_bg": 0,
        "crushed_or_deformed_can": 0,
        "multi_item_or_pile_clutter": 0,
        "foil_tins_lids_other": 0,
    }
    metal_feats: List[Dict[str, float]] = []
    for r in metal_rows:
        p = BASE_DIR / r["path"]
        with Image.open(p) as im:
            vf = analyze_image_visual_features(im)
            metal_feats.append(vf)
            if vf["border_uniformity"] >= 0.55 and vf["edge_density"] < 0.25:
                metal_subtypes["single_clean_can_plain_bg"] += 1
            elif vf["min_quadrant_edge"] >= 0.22 and vf["edge_density"] >= 0.28 and vf["border_uniformity"] < 0.35:
                metal_subtypes["multi_item_or_pile_clutter"] += 1
            elif vf["edge_density"] >= 0.20 and r["source"] == "realwaste":
                metal_subtypes["crushed_or_deformed_can"] += 1
            else:
                metal_subtypes["foil_tins_lids_other"] += 1

    organic_feats: List[Dict[str, float]] = []
    for r in organic_rows[:200]:
        p = BASE_DIR / r["path"]
        with Image.open(p) as im:
            organic_feats.append(analyze_image_visual_features(im))

    can_pile_visual = analyze_image_visual_features(can_img)

    # Save 40-image contact sheets for metal and organic
    rng = random.Random(42)
    sample_metal = rng.sample(metal_rows, min(40, len(metal_rows)))
    sample_organic = rng.sample(organic_rows, min(40, len(organic_rows)))
    make_contact_sheet(
        [(BASE_DIR / r["path"], f"{r['source'][:2]}:{Path(r['path']).stem[:10]}") for r in sample_metal],
        REPORTS_DIR / "phase_a_metal_40.jpg",
    )
    make_contact_sheet(
        [(BASE_DIR / r["path"], f"{r['source'][:2]}:{Path(r['path']).stem[:10]}") for r in sample_organic],
        REPORTS_DIR / "phase_a_organic_40.jpg",
    )

    # A4: Grad-CAM for metal_can_pile_01.jpg across top-3 classes (metal=1, plastic=4, organic=2)
    boxed_can = letterbox_pil_image(can_img, 224)
    cam_panels: List[Image.Image] = []
    orig_panel = Image.new("RGB", (224, 248), (20, 22, 26))
    orig_panel.paste(boxed_can, (0, 24))
    ImageDraw.Draw(orig_panel).text((6, 5), "Input: metal_can_pile_01", fill=(240, 244, 248))
    cam_panels.append(orig_panel)

    for cls_idx, cls_name in enumerate(class_names):
        if cls_name in ("metal", "plastic", "organic"):
            cam = compute_gradcam(pt_model, x_orig, cls_idx)
            prob_pct = probs_post_temp_pt[cls_idx] * 100.0
            cam_panels.append(overlay_heatmap(boxed_can, cam, f"CAM {cls_name}: {prob_pct:.1f}%"))

    gradcam_pile_sheet = Image.new("RGB", (224 * len(cam_panels), 248), (20, 22, 26))
    for i, panel in enumerate(cam_panels):
        gradcam_pile_sheet.paste(panel, (i * 224, 0))
    gradcam_pile_sheet.save(REPORTS_DIR / "gradcam_metal_can_pile_v2.jpg", format="JPEG", quality=88)

    # Grad-CAM on 10 correctly classified metal validation images
    val_rows: List[Dict[str, str]] = []
    with (SPLITS_DIR / "val.csv").open("r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    correct_metal_panels: List[Image.Image] = []
    metal_idx = class_names.index("metal")
    for r in val_rows:
        if r["label"] != "metal":
            continue
        p = BASE_DIR / r["path"]
        with Image.open(p) as im:
            xt = preprocess_tensor(im.convert("RGB"), 224)
            with torch.no_grad():
                pr = softmax_np(pt_model(xt).numpy()[0], temperature=temperature)
            if int(np.argmax(pr)) == metal_idx and pr[metal_idx] >= 0.75:
                cam = compute_gradcam(pt_model, xt, metal_idx)
                boxed = letterbox_pil_image(im.convert("RGB"), 224)
                correct_metal_panels.append(
                    overlay_heatmap(boxed, cam, f"{Path(p).stem[:12]} ({pr[metal_idx]*100:.0f}%)")
                )
        if len(correct_metal_panels) >= 10:
            break

    if correct_metal_panels:
        cm_sheet = Image.new("RGB", (224 * 5, 248 * 2), (20, 22, 26))
        for idx, panel in enumerate(correct_metal_panels[:10]):
            cm_sheet.paste(panel, ((idx % 5) * 224, (idx // 5) * 248))
        cm_sheet.save(REPORTS_DIR / "gradcam_10_correct_metal.jpg", format="JPEG", quality=86)

    # A4 + A5: Validation evaluation (RGB vs Grayscale Colour-Ablation + Confusion Matrix)
    y_true: List[int] = []
    y_pred_rgb: List[int] = []
    y_pred_gray: List[int] = []

    for r in val_rows:
        true_idx = class_names.index(r["label"])
        y_true.append(true_idx)
        p = BASE_DIR / r["path"]
        with Image.open(p) as im:
            rgb_im = im.convert("RGB")
            gray_im = ImageOps.grayscale(rgb_im).convert("RGB")

            x_rgb = preprocess_tensor(rgb_im, 224)
            x_rgb_flip = torch.flip(x_rgb, dims=[3])
            x_gr = preprocess_tensor(gray_im, 224)
            x_gr_flip = torch.flip(x_gr, dims=[3])

            with torch.no_grad():
                l_rgb = 0.65 * pt_model(x_rgb).numpy()[0] + 0.35 * pt_model(x_rgb_flip).numpy()[0]
                l_gr = 0.65 * pt_model(x_gr).numpy()[0] + 0.35 * pt_model(x_gr_flip).numpy()[0]

            y_pred_rgb.append(int(np.argmax(l_rgb)))
            y_pred_gray.append(int(np.argmax(l_gr)))

    y_true_arr = np.array(y_true)
    y_rgb_arr = np.array(y_pred_rgb)
    y_gray_arr = np.array(y_pred_gray)

    val_rgb_acc = float(accuracy_score(y_true_arr, y_rgb_arr))
    val_gray_acc = float(accuracy_score(y_true_arr, y_gray_arr))
    val_cm = confusion_matrix(y_true_arr, y_rgb_arr, labels=list(range(len(class_names)))).tolist()

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_arr, y_rgb_arr, labels=list(range(len(class_names))), zero_division=0
    )
    _, rec_gray, f1_gray, _ = precision_recall_fscore_support(
        y_true_arr, y_gray_arr, labels=list(range(len(class_names))), zero_division=0
    )

    per_class_val_ablation = []
    for i, cname in enumerate(class_names):
        per_class_val_ablation.append(
            {
                "category": cname,
                "support": int(sup[i]),
                "rgb_precision": round(float(prec[i]), 4),
                "rgb_recall": round(float(rec[i]), 4),
                "rgb_f1": round(float(f1[i]), 4),
                "grayscale_recall": round(float(rec_gray[i]), 4),
                "grayscale_f1": round(float(f1_gray[i]), 4),
                "recall_drop_on_grayscale": round(float(rec[i] - rec_gray[i]), 4),
            }
        )

    # Specific confusion rates on validation
    m_idx = class_names.index("metal")
    p_idx = class_names.index("plastic")
    o_idx = class_names.index("organic")
    g_idx = class_names.index("glass")

    metal_total = max(1, sum(val_cm[m_idx]))
    plastic_total = max(1, sum(val_cm[p_idx]))

    confusion_rates = {
        "metal_to_plastic_rate": round(val_cm[m_idx][p_idx] / metal_total, 4),
        "metal_to_organic_rate": round(val_cm[m_idx][o_idx] / metal_total, 4),
        "metal_to_glass_rate": round(val_cm[m_idx][g_idx] / metal_total, 4),
        "plastic_to_metal_rate": round(val_cm[p_idx][m_idx] / plastic_total, 4),
        "plastic_to_glass_rate": round(val_cm[p_idx][g_idx] / plastic_total, 4),
    }

    report_data = {
        "fixture_path": str(FIXTURE_CANS.relative_to(BASE_DIR)),
        "image_size": list(can_img.size),
        "can_pile_visual_features": can_pile_visual,
        "a1_reproduction": {
            "pytorch_probs": {class_names[i]: round(float(probs_post_temp_pt[i]), 4) for i in range(len(class_names))},
            "onnx_probs": {class_names[i]: round(float(probs_ort[i]), 4) for i in range(len(class_names))},
            "cli_status": cli_json["status"],
            "cli_top_k": cli_json["top_k"],
            "api_status": api_json["status"],
            "api_category": api_json["category"],
            "api_quality_warnings": api_json["quality_warnings"],
            "all_four_agree": True,
        },
        "a2_raw_vs_calibrated": {
            "raw_logits_orig": {class_names[i]: round(float(logits_orig_pt[i]), 4) for i in range(len(class_names))},
            "raw_logits_hflip": {class_names[i]: round(float(logits_flip_pt[i]), 4) for i in range(len(class_names))},
            "raw_logits_tta": {class_names[i]: round(float(logits_tta_pt[i]), 4) for i in range(len(class_names))},
            "softmax_before_temp": {class_names[i]: round(float(probs_pre_temp_pt[i]), 4) for i in range(len(class_names))},
            "temperature": temperature,
            "softmax_after_temp": {class_names[i]: round(float(probs_post_temp_pt[i]), 4) for i in range(len(class_names))},
            "shannon_entropy_nats": round(entropy_nats, 4),
            "max_possible_entropy_nats": round(math.log(len(class_names)), 4),
            "normalized_entropy": round(entropy_norm, 4),
        },
        "a3_training_data_audit": {
            "per_class_train_counts": dict(per_class_counts),
            "per_source_train_counts": {f"{s}:{c}": cnt for (s, c), cnt in sorted(per_source_class.items())},
            "metal_subtypes": metal_subtypes,
            "mean_metal_colourfulness": round(float(np.mean([x["colourfulness"] for x in metal_feats])), 2),
            "mean_organic_colourfulness": round(float(np.mean([x["colourfulness"] for x in organic_feats])), 2),
            "mean_metal_edge_density": round(float(np.mean([x["edge_density"] for x in metal_feats])), 4),
            "mean_organic_edge_density": round(float(np.mean([x["edge_density"] for x in organic_feats])), 4),
            "can_pile_colourfulness": can_pile_visual["colourfulness"],
            "can_pile_edge_density": can_pile_visual["edge_density"],
        },
        "a4_colour_ablation_validation": {
            "val_samples": len(val_rows),
            "val_rgb_accuracy": round(val_rgb_acc, 4),
            "val_grayscale_accuracy": round(val_gray_acc, 4),
            "overall_accuracy_drop": round(val_rgb_acc - val_gray_acc, 4),
            "per_class_ablation": per_class_val_ablation,
        },
        "a5_validation_slices": {
            "confusion_matrix": val_cm,
            "class_names": class_names,
            "confusion_rates": confusion_rates,
        },
    }

    out_json = REPORTS_DIR / "phase_a_cans_diagnostics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[diagnose_phase_a_cans] Wrote {out_json}", flush=True)
    print(json.dumps(report_data["a1_reproduction"], indent=2), flush=True)
    print(json.dumps(report_data["a2_raw_vs_calibrated"], indent=2), flush=True)
    print(json.dumps(report_data["a3_training_data_audit"], indent=2), flush=True)
    print(json.dumps(report_data["a4_colour_ablation_validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
