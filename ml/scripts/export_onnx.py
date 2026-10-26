"""Export selected PyTorch checkpoint to ONNX, verify with onnx.checker, test 20-image softmax parity (< 1e-4), and write model_card.json."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import List

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_factory import (  # noqa: E402
    BASE_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ManifestWasteDataset,
    build_classifier,
    get_transforms,
    softmax_with_temperature,
)

MODELS_DIR = BASE_DIR / "artifacts" / "models"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics" / "metrics.json"
SPLITS_DIR = BASE_DIR / "ml" / "data" / "splits"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected_pt = MODELS_DIR / "selected_best.pt"
    if not selected_pt.exists():
        candidates = sorted(MODELS_DIR.glob("*_best.pt"))
        if not candidates:
            raise FileNotFoundError("No trained PyTorch checkpoint found to export.")
        selected_pt = candidates[0]

    ckpt = torch.load(selected_pt, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    backbone = meta["backbone"]
    class_names: List[str] = meta["class_names"]
    image_size = int(meta.get("image_size", 224))
    temperature = float(meta.get("temperature", 1.0))

    model = build_classifier(backbone, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = MODELS_DIR / "model.onnx"
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    print(f"[export] Exporting {backbone} to {onnx_path} (opset 17)...", flush=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[export] onnx.checker.check_model PASSED.", flush=True)

    onnx_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    model_version = f"3.0.0+{date_tag}.{onnx_sha256[:8]}"

    test_csv = SPLITS_DIR / "test.csv"
    test_ds = ManifestWasteDataset(
        test_csv,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=False),
    )

    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diffs: List[float] = []
    num_parity_images = min(20, len(test_ds))

    with torch.no_grad():
        for i in range(num_parity_images):
            tensor_img, _, _ = test_ds[i]
            batch_t = tensor_img.unsqueeze(0)

            pt_logits = model(batch_t).numpy()
            ort_logits = ort_sess.run(None, {"input": batch_t.numpy()})[0]

            pt_probs = softmax_with_temperature(pt_logits, temperature)
            ort_probs = softmax_with_temperature(ort_logits, temperature)

            diff = float(np.max(np.abs(pt_probs - ort_probs)))
            max_diffs.append(diff)

    overall_max_diff = float(np.max(max_diffs)) if max_diffs else 0.0
    parity_passed = overall_max_diff < 1e-4
    print(
        f"[parity] Checked {num_parity_images} test images: max_abs_diff={overall_max_diff:.2e} "
        f"(< 1e-4 required) -> {'PASS' if parity_passed else 'FAIL'}",
        flush=True,
    )
    if not parity_passed:
        raise AssertionError(f"ONNX parity check failed: max_abs_diff={overall_max_diff} >= 1e-4")

    metrics_summary = {}
    risk_thresholds = {
        "confidence_threshold": 0.50,
        "high_confidence_threshold": 0.80,
        "min_top1_top2_margin": 0.12,
        "max_normalized_entropy": 0.82,
    }
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            m = json.load(f)
            metrics_summary = {
                "test_accuracy": m.get("accuracy"),
                "test_macro_f1": m.get("macro_f1"),
                "test_top_3_accuracy": m.get("top_3_accuracy"),
                "test_ece": m.get("ece"),
                "transparent_container_slice": m.get("transparent_container_slice"),
            }
            if m.get("risk_coverage_thresholds"):
                risk_thresholds = m["risk_coverage_thresholds"]

    model_card = {
        "model_name": "EcoSort ML Waste Classifier",
        "model_version": model_version,
        "architecture": backbone,
        "is_smoke_test": bool(meta.get("is_smoke_test", False)),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "weights_sha256": onnx_sha256,
        "input_spec": {
            "name": "input",
            "shape": ["batch_size", 3, image_size, image_size],
            "dtype": "float32",
            "color_space": "RGB",
            "range": "[0.0, 1.0] normalized by ImageNet mean/std",
        },
        "input_size": image_size,
        "preprocessing": {
            "method": "letterbox",
            "target_size": image_size,
            "fill_rgb": [124, 116, 104],
            "tta": "adaptive multi-view (2-view hflip + 2x2 overlapping tiles & grayscale de-colourised view on cluttered/pile scenes)",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "thresholds": risk_thresholds,
        "class_names": class_names,
        "classes": class_names,
        "temperature": temperature,
        "parity_check": {
            "num_images_tested": num_parity_images,
            "max_abs_softmax_diff": overall_max_diff,
            "threshold": 1e-4,
            "passed": parity_passed,
        },
        "training_metadata": {
            "best_epoch": meta.get("best_epoch"),
            "best_val_macro_f1": meta.get("best_val_macro_f1"),
            "wall_clock_seconds": meta.get("wall_clock_seconds"),
            "git_commit": meta.get("git_commit"),
            "manifest_sha256": meta.get("manifest_sha256"),
        },
        "metrics_summary": metrics_summary,
    }

    card_path = MODELS_DIR / "model_card.json"
    with card_path.open("w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    classes_path = MODELS_DIR / "classes.json"
    with classes_path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"[export] Wrote model card to {card_path} (version={model_version})", flush=True)


if __name__ == "__main__":
    main()
