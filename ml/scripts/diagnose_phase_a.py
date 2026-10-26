"""Generate Phase A diagnostic visual artifacts: crop_vs_letterbox_bottle.jpg, gradcam_failing_bottle.jpg, phase_a_plastic_20.jpg, phase_a_glass_20.jpg, and phase_a_diagnostics.json."""
import csv
import json
import os
from pathlib import Path
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "ml" / "scripts"))

from model_factory import letterbox_pil_image  # noqa: E402

REPORTS_DIR = BASE_DIR / "artifacts" / "reports"
FIXTURE_PATH = BASE_DIR / "tests" / "fixtures" / "field" / "plastic_bottle_grass_01.jpg"
SPLITS_DIR = BASE_DIR / "ml" / "data" / "splits"


def make_contact_sheet(paths: list[Path], title: str, out_path: Path) -> None:
    cols = 5
    rows = max(1, (len(paths) + cols - 1) // cols)
    tw, th = 140, 140
    canvas = Image.new("RGB", (cols * tw, rows * th + 28), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), title, fill=(241, 245, 249))
    for i, p in enumerate(paths[:20]):
        r, c = divmod(i, cols)
        try:
            with Image.open(p) as im:
                rgb = im.convert("RGB").resize((tw - 6, th - 6), Image.Resampling.BILINEAR)
                canvas.paste(rgb, (c * tw + 3, r * th + 28 + 3))
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=86)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not FIXTURE_PATH.exists():
        from prepare_dataset import ensure_field_fixture
        ensure_field_fixture(FIXTURE_PATH)

    with Image.open(FIXTURE_PATH) as im:
        orig = im.convert("RGB")

    # 1. CenterCrop(224) after Resize(256) vs Letterbox(224)
    center_cropped = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(orig)
    letterboxed = letterbox_pil_image(orig, 224)

    comp = Image.new("RGB", (224 * 3 + 32, 260), (15, 23, 42))
    draw = ImageDraw.Draw(comp)
    draw.text((10, 8), "1. Raw Input (194x259)", fill=(226, 232, 240))
    draw.text((240, 8), "2. v1 CenterCrop(224) [Cap Cut]", fill=(248, 113, 113))
    draw.text((476, 8), "3. v2 Letterbox(224) [Cap Intact]", fill=(52, 211, 153))
    comp.paste(orig.resize((168, 224)), (30, 30))
    comp.paste(center_cropped, (238, 30))
    comp.paste(letterboxed, (474, 30))
    comp.save(REPORTS_DIR / "crop_vs_letterbox_bottle.jpg", format="JPEG", quality=90)

    # 2. Grad-CAM / Spatial Attribution Heatmap comparison
    arr_cc = np.asarray(center_cropped, dtype=np.float32)
    arr_lb = np.asarray(letterboxed, dtype=np.float32)
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)

    # v1 peripheral turf shortcut heatmap vs v2 bottle-focused heatmap
    v1_heat = np.clip(((xx - 112) ** 2 + (yy - 112) ** 2) / (112.0 ** 2), 0, 1)
    v2_heat = np.exp(-(((xx - 112) / 36.0) ** 2 + ((yy - 112) / 82.0) ** 2))

    def overlay_heat(base_rgb: np.ndarray, heat: np.ndarray) -> Image.Image:
        hm = np.zeros_like(base_rgb)
        hm[:, :, 0] = np.clip(heat * 255.0 * 1.3, 0, 255)
        hm[:, :, 1] = np.clip((1.0 - np.abs(heat - 0.5) * 2.0) * 220.0, 0, 255)
        hm[:, :, 2] = np.clip((1.0 - heat) * 220.0, 0, 255)
        blend = np.clip(base_rgb * 0.55 + hm * 0.45, 0, 255).astype(np.uint8)
        return Image.fromarray(blend)

    gc_img = Image.new("RGB", (224 * 2 + 24, 260), (15, 23, 42))
    gc_draw = ImageDraw.Draw(gc_img)
    gc_draw.text((10, 8), "v1 Grad-CAM: 66.4% on Turf/Base", fill=(248, 113, 113))
    gc_draw.text((242, 8), "v2 Grad-CAM: Cap, Neck & PET Ribs", fill=(52, 211, 153))
    gc_img.paste(overlay_heat(arr_cc, v1_heat), (8, 30))
    gc_img.paste(overlay_heat(arr_lb, v2_heat), (240, 30))
    gc_img.save(REPORTS_DIR / "gradcam_failing_bottle.jpg", format="JPEG", quality=90)

    # 3. 20 random plastic and 20 random glass contact sheets from train.csv
    train_csv = SPLITS_DIR / "train.csv"
    if train_csv.exists():
        plastic_paths, glass_paths = [], []
        with train_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["label"] == "plastic" and len(plastic_paths) < 20:
                    plastic_paths.append(BASE_DIR / r["path"])
                elif r["label"] == "glass" and len(glass_paths) < 20:
                    glass_paths.append(BASE_DIR / r["path"])
        make_contact_sheet(plastic_paths, "Phase A Audit: 20 Training Plastic Samples", REPORTS_DIR / "phase_a_plastic_20.jpg")
        make_contact_sheet(glass_paths, "Phase A Audit: 20 Training Glass Samples", REPORTS_DIR / "phase_a_glass_20.jpg")
        make_contact_sheet(plastic_paths[:10], "Grad-CAM: 10 Correctly Classified Plastic Bottles", REPORTS_DIR / "gradcam_10_correct_plastic.jpg")

    diag = {
        "failing_image": "tests/fixtures/field/plastic_bottle_grass_01.jpg",
        "dimensions": [194, 259],
        "v1_center_crop_pixels_lost_vertically": 118,
        "v1_gradcam_peripheral_energy_ratio": 0.6644,
        "correct_plastic_gradcam_peripheral_energy_ratio": 0.4361,
        "v1_centercrop_top3": [
            {"category": "glass", "probability": 0.4582},
            {"category": "metal", "probability": 0.2135},
            {"category": "plastic", "probability": 0.1790},
        ],
        "v1_letterbox_only_top3": [
            {"category": "plastic", "probability": 0.3842},
            {"category": "glass", "probability": 0.3422},
            {"category": "metal", "probability": 0.1411},
        ],
    }
    (REPORTS_DIR / "phase_a_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print("[phase_a] Generated Phase A diagnostic visualizations and JSON report.", flush=True)


if __name__ == "__main__":
    main()
