"""CLI tool for single-image or directory waste classification using the exported ONNX model."""
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "backend"))

from app.config import settings  # noqa: E402
from app.core.security import validate_and_sanitize_image  # noqa: E402
from app.core.taxonomy import load_taxonomy  # noqa: E402
from app.ml.inference import classifier_service  # noqa: E402


def predict_file(path: Path) -> Dict[str, Any]:
    raw_bytes = path.read_bytes()
    val = validate_and_sanitize_image(raw_bytes, original_filename=path.name)
    res = classifier_service.predict(val.clean_rgb_image)
    taxonomy = load_taxonomy(settings.TAXONOMY_YAML_PATH)

    if res.status == "uncertain":
        return {
            "file": str(path),
            "status": "uncertain",
            "category": None,
            "predicted_top1": res.category,
            "detected_item": None,
            "recyclability_class": "Unknown",
            "bin": None,
            "confidence": res.confidence,
            "confidence_level": res.confidence_level,
            "top_k": res.top_k,
            "probabilities": res.probabilities,
            "disposal_tips": [],
            "prep_steps": [],
            "safe_handling_advice": [
                "Do not place an unverified item into a recycling or compost bin.",
                "Inspect the item manually for a resin code (#1-#7) or material label, or retake the photo on a plain background.",
            ],
            "uncertainty_reason": res.uncertainty_reason,
            "entropy": res.entropy,
            "is_near_uniform": res.is_near_uniform,
            "quality_warnings": val.quality_warnings,
            "inference_ms": res.inference_ms,
            "model_version": res.model_version,
        }

    entry = taxonomy.get_entry(res.category)
    return {
        "file": str(path),
        "status": res.status,
        "category": entry.category,
        "predicted_top1": res.category,
        "detected_item": entry.detected_item,
        "recyclability_class": entry.recyclability_class,
        "bin": entry.bin,
        "confidence": res.confidence,
        "confidence_level": res.confidence_level,
        "top_k": res.top_k,
        "probabilities": res.probabilities,
        "disposal_tips": entry.disposal_tips,
        "prep_steps": entry.prep_steps,
        "safe_handling_advice": [],
        "entropy": res.entropy,
        "is_near_uniform": res.is_near_uniform,
        "quality_warnings": val.quality_warnings,
        "inference_ms": res.inference_ms,
        "model_version": res.model_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EcoSort ML CLI Predictor")
    parser.add_argument("target", type=str, help="Path to an image file or directory of images")
    args = parser.parse_args()

    if not classifier_service.ensure_loaded():
        print(json.dumps({"error": "MODEL_NOT_LOADED", "message": "Model not loaded"}), file=sys.stderr)
        sys.exit(2)

    target_path = Path(args.target)
    if target_path.is_dir():
        files = sorted(
            p for p in target_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        results: List[Dict[str, Any]] = [predict_file(p) for p in files]
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(predict_file(target_path), indent=2))


if __name__ == "__main__":
    main()
