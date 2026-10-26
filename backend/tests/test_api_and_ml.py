from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import BASE_DIR, settings
from app.core.security import (
    ImageValidationError,
    sanitize_csv_cell,
    sanitize_filename,
    validate_and_sanitize_image,
)
from app.core.taxonomy import load_taxonomy
from app.ml.inference import (
    classifier_service,
    evaluate_threshold_and_margin,
    letterbox_pil_image,
    preprocess_pil_image,
)
from app.models.prediction import Prediction


def make_test_image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (224, 224),
    color: tuple[int, int, int] = (90, 160, 110),
) -> bytes:
    img = Image.new("RGB", size, color=color)
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    arr[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)] = [210, 195, 160]
    arr[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)] = [40, 110, 190]
    out = Image.fromarray(arr)
    buf = io.BytesIO()
    out.save(buf, format=fmt)
    return buf.getvalue()


def test_taxonomy_loads_and_covers_all_five_categories():
    tax = load_taxonomy(settings.TAXONOMY_YAML_PATH)
    required = {"plastic", "paper", "metal", "glass", "organic"}
    assert set(tax.categories) == required
    for cat in required:
        entry = tax.get_entry(cat)
        assert entry.category == cat
        assert entry.recyclability_class in {
            "Recyclable",
            "Compostable",
            "Non-recyclable",
            "Special handling",
        }
        assert len(entry.disposal_tips) >= 1
        assert len(entry.prep_steps) >= 1


def test_preprocess_pil_image_shape_and_dtype():
    img = Image.new("RGB", (320, 240), color=(120, 80, 200))
    tensor = preprocess_pil_image(img, target_size=224)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_letterbox_preserves_aspect_ratio_without_cropping_extremities():
    """Verify 194x259 tall portrait image keeps both top cap and bottom base after 224x224 letterbox."""
    arr = np.full((259, 194, 3), 80, dtype=np.uint8)
    # Bright red top cap (rows 0..12) and bright blue bottom base (rows 246..258)
    arr[0:12, 60:134] = [250, 10, 10]
    arr[246:259, 60:134] = [10, 20, 250]
    pil_img = Image.fromarray(arr)

    boxed = letterbox_pil_image(pil_img, target_size=224)
    assert boxed.size == (224, 224)
    boxed_np = np.array(boxed)

    # Top rows near center column must still contain the bright red cap
    top_patch = boxed_np[1:8, 100:124]
    assert float(top_patch[:, :, 0].mean()) > 200.0
    assert float(top_patch[:, :, 2].mean()) < 40.0

    # Bottom rows near center column must still contain the bright blue base
    bot_patch = boxed_np[216:223, 100:124]
    assert float(bot_patch[:, :, 2].mean()) > 200.0
    assert float(bot_patch[:, :, 0].mean()) < 40.0

    # Left and right borders (cols 0..10) must be neutral padding (124, 116, 104)
    left_pad = boxed_np[100:120, 2:10]
    assert np.allclose(left_pad.mean(axis=(0, 1)), [124, 116, 104], atol=2.0)


def test_threshold_and_margin_boundaries():
    """Verify exact confidence and top1-top2 margin boundary rules."""
    # 0.49 -> uncertain, LOW
    st, lvl, reason = evaluate_threshold_and_margin(
        top1_prob=0.49, top2_prob=0.20, min_thresh=0.50, high_thresh=0.80, min_margin=0.12
    )
    assert st == "uncertain"
    assert lvl == "LOW"
    assert reason is not None

    # 0.50 with sufficient margin -> ok, MEDIUM
    st, lvl, reason = evaluate_threshold_and_margin(
        top1_prob=0.50, top2_prob=0.30, min_thresh=0.50, high_thresh=0.80, min_margin=0.12
    )
    assert st == "ok"
    assert lvl == "MEDIUM"
    assert reason is None

    # 0.79 with sufficient margin -> ok, MEDIUM
    st, lvl, reason = evaluate_threshold_and_margin(
        top1_prob=0.79, top2_prob=0.15, min_thresh=0.50, high_thresh=0.80, min_margin=0.12
    )
    assert st == "ok"
    assert lvl == "MEDIUM"
    assert reason is None

    # 0.80 with sufficient margin -> ok, HIGH
    st, lvl, reason = evaluate_threshold_and_margin(
        top1_prob=0.80, top2_prob=0.10, min_thresh=0.50, high_thresh=0.80, min_margin=0.12
    )
    assert st == "ok"
    assert lvl == "HIGH"
    assert reason is None

    # Margin rule: top1=0.58 >= 0.50, but top2=0.50 -> margin=0.08 < 0.12 -> uncertain!
    st, lvl, reason = evaluate_threshold_and_margin(
        top1_prob=0.58, top2_prob=0.50, min_thresh=0.50, high_thresh=0.80, min_margin=0.12
    )
    assert st == "uncertain"
    assert lvl == "LOW"
    assert "margin" in (reason or "").lower()


def test_low_resolution_and_dominant_background_warnings():
    """Verify 194x259 image with saturated green border triggers both quality warnings."""
    arr = np.zeros((259, 194, 3), dtype=np.uint8)
    arr[:, :] = [35, 195, 45]  # highly saturated green turf border
    arr[70:190, 65:130] = [200, 210, 215]  # center object
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)

    val = validate_and_sanitize_image(buf.getvalue(), original_filename="small_grass.jpg")
    assert val.width == 194
    assert val.height == 259
    joined_warnings = " | ".join(val.quality_warnings)
    assert "Image is small (194x259). Larger photos (>= 224 px on the shorter side) usually give better results." in joined_warnings
    assert "Dominant background color/texture detected" in joined_warnings


def test_csv_formula_injection_sanitization():
    assert sanitize_csv_cell("=CMD|' /C calc'!A0") == "'=CMD|' /C calc'!A0"
    assert sanitize_csv_cell("+SUM(A1:A2)") == "'+SUM(A1:A2)"
    assert sanitize_csv_cell("-2+3") == "'-2+3"
    assert sanitize_csv_cell("@import") == "'@import"
    assert sanitize_csv_cell("bottle.jpg") == "bottle.jpg"


def test_filename_sanitization_blocks_path_traversal():
    cleaned = sanitize_filename("../../etc/passwd.jpg")
    assert ".." not in cleaned
    assert "/" not in cleaned
    assert cleaned.endswith("passwd.jpg")


def test_validate_and_sanitize_image_strips_exif_and_validates():
    raw = make_test_image_bytes("JPEG", (180, 180))
    val = validate_and_sanitize_image(raw, original_filename="test_item.jpg")
    assert val.width == 180
    assert val.height == 180
    assert val.format_name == "JPEG"
    assert len(val.file_sha256) == 64

    with pytest.raises(ImageValidationError) as exc_empty:
        validate_and_sanitize_image(b"", original_filename="empty.jpg")
    assert exc_empty.value.status_code == 400


def test_health_and_model_info_endpoints(client: TestClient):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["db_connected"] is True
    assert isinstance(body["model_loaded"], bool)

    info_res = client.get("/api/model/info")
    assert info_res.status_code == 200
    info = info_res.json()
    assert set(info["classes"]) == {"glass", "metal", "organic", "paper", "plastic"}
    assert "disclaimer" in info


def test_predict_rejects_oversized_file_with_413(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    oversized = b"\xff\xd8\xff\xe0" + (b"0" * (1024 * 1024 + 512))
    res = client.post(
        "/api/predict",
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_predict_rejects_spoofed_magic_bytes_with_415(client: TestClient):
    spoofed = b"NotAnImageAtAllJustPlainText1234567890"
    res = client.post(
        "/api/predict",
        files={"file": ("fake.jpg", spoofed, "image/jpeg")},
    )
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_predict_rejects_corrupt_image_bytes(client: TestClient):
    corrupt_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01trunc"
    res = client.post(
        "/api/predict",
        files={"file": ("corrupt.jpg", corrupt_jpeg, "image/jpeg")},
    )
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "CORRUPT_IMAGE"


def test_predict_returns_503_when_model_not_loaded(client: TestClient, monkeypatch):
    monkeypatch.setattr(classifier_service, "ensure_loaded", lambda: False)
    img_bytes = make_test_image_bytes()
    res = client.post(
        "/api/predict",
        files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_uncertain_response_omits_category_bin_and_disposal_tips(client: TestClient, monkeypatch):
    """Critical Phase D1 regression test: uncertain predictions must NEVER return top-1 category or disposal tips."""
    from app.ml.inference import InferenceOutput

    def mock_uncertain_predict(_pil_image):
        return InferenceOutput(
            category="glass",
            confidence=0.442,
            confidence_level="LOW",
            status="uncertain",
            top_k=[
                {"label": "glass", "category": "glass", "probability": 0.442},
                {"label": "metal", "category": "metal", "probability": 0.198},
                {"label": "plastic", "category": "plastic", "probability": 0.181},
            ],
            probabilities={
                "glass": 0.442,
                "metal": 0.198,
                "plastic": 0.181,
                "paper": 0.100,
                "organic": 0.079,
            },
            inference_ms=11.4,
            model_version="2.0.0+test",
            is_smoke_test=False,
            uncertainty_reason="Top-1 confidence (44.2%) is below minimum threshold (50%).",
        )

    monkeypatch.setattr(classifier_service, "predict", mock_uncertain_predict)
    img_bytes = make_test_image_bytes("JPEG", size=(194, 259))
    res = client.post(
        "/api/predict",
        files={"file": ("images.jpeg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "uncertain"
    assert data["category"] is None
    assert data["detected_item"] is None
    assert data["recyclability_class"] == "Unknown"
    assert data["bin"] is None
    assert data["disposal_tips"] == []
    assert data["prep_steps"] == []
    assert data["predicted_top1"] == "glass"
    assert len(data["safe_handling_advice"]) >= 2
    assert data["retake_advice"] is not None
    assert "plastic" in data["candidate_guidance"]
    assert "glass" in data["candidate_guidance"]


def test_predict_valid_image_and_db_persistence(client: TestClient):
    health = client.get("/api/health").json()
    assert health["model_loaded"] is True

    img_bytes = make_test_image_bytes("JPEG")
    res = client.post(
        "/api/predict",
        files={"file": ("green_bottle.jpg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_id"]
    assert data["status"] in {"ok", "uncertain"}
    if data["status"] == "ok":
        assert data["category"] in {"plastic", "paper", "metal", "glass", "organic"}
        assert len(data["disposal_tips"]) >= 1
    else:
        assert data["category"] is None
        assert data["disposal_tips"] == []
        assert data["recyclability_class"] == "Unknown"
    assert data["confidence_level"] in {"HIGH", "MEDIUM", "LOW"}
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-2
    assert len(data["top_k"]) == 3

    # Verify detail endpoint, thumbnail endpoint, and user feedback
    pred_id = data["prediction_id"]
    detail_res = client.get(f"/api/predictions/{pred_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["prediction_id"] == pred_id

    if data["has_thumbnail"]:
        thumb_res = client.get(f"/api/predictions/{pred_id}/image")
        assert thumb_res.status_code == 200
        assert thumb_res.headers["content-type"] == "image/jpeg"

    fb_res = client.post(
        f"/api/predictions/{pred_id}/feedback",
        json={"corrected_category": "plastic"},
    )
    assert fb_res.status_code == 200
    fb_body = fb_res.json()
    assert fb_body["corrected_category"] == "plastic"
    # Once user confirms category, disposal tips for that category must be populated
    assert len(fb_body["disposal_tips"]) >= 1
    assert fb_body["recyclability_class"] == "Recyclable"


def test_field_eval_isolation_and_fixture_classification(client: TestClient):
    """Verify field_eval dataset has >=30 images/class, zero train leakage, and classifies plastic_bottle_grass_01.jpg."""
    field_dir = BASE_DIR / "ml" / "data" / "field_eval"
    train_csv = BASE_DIR / "ml" / "data" / "splits" / "train.csv"
    assert field_dir.exists()
    assert train_csv.exists()

    train_hashes = set()
    with train_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_hashes.add(row["sha256"])

    for cls in ("plastic", "paper", "metal", "glass", "organic"):
        cls_files = [
            p for p in (field_dir / cls).iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        assert len(cls_files) >= 30, f"Expected >=30 field_eval images for {cls}, found {len(cls_files)}"
        for p in cls_files:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            assert sha not in train_hashes, f"Leakage detected: {p} is in train.csv!"

    fixture_path = BASE_DIR / "tests" / "fixtures" / "field" / "plastic_bottle_grass_01.jpg"
    assert fixture_path.exists()
    res = client.post(
        "/api/predict",
        files={"file": ("plastic_bottle_grass_01.jpg", fixture_path.read_bytes(), "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    # Must either be a confident 'plastic' prediction OR an honest 'uncertain' with plastic in top-2 and no glass disposal tips
    if body["status"] == "ok":
        assert body["category"] == "plastic"
    else:
        assert body["category"] is None
        assert body["disposal_tips"] == []
        top2_cats = [item["category"] for item in body["top_k"][:2]]
        assert "plastic" in top2_cats


def test_batch_predict_handles_mixed_valid_and_invalid_files(client: TestClient):
    valid_img = make_test_image_bytes("PNG")
    invalid_bytes = b"not-an-image-payload"
    res = client.post(
        "/api/predict/batch",
        files=[
            ("files", ("good_item.png", valid_img, "image/png")),
            ("files", ("bad_item.jpg", invalid_bytes, "image/jpeg")),
        ],
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    assert body["succeeded"] == 1
    assert body["failed"] == 1


def test_history_analytics_export_and_delete_flow(client: TestClient, db_session):
    rec_real = Prediction(
        id="test-real-001",
        original_filename="=cmd_injection.jpg",
        file_sha256="a" * 64,
        image_width=224,
        image_height=224,
        detected_item="plastic",
        category="plastic",
        predicted_top1="plastic",
        recyclability_class="Recyclable",
        bin="Blue Recycling Bin (Plastics & Containers)",
        confidence=0.91,
        confidence_level="HIGH",
        status="ok",
        top_k=[{"label": "plastic", "category": "plastic", "probability": 0.91}],
        quality_warnings=[],
        model_version="2.0.0",
        inference_ms=14.2,
        is_demo=False,
    )
    rec_uncertain = Prediction(
        id="test-uncertain-003",
        original_filename="blurry_bottle.jpg",
        file_sha256="c" * 64,
        image_width=194,
        image_height=259,
        detected_item=None,
        category=None,
        predicted_top1="glass",
        recyclability_class="Unknown",
        bin=None,
        confidence=0.44,
        confidence_level="LOW",
        status="uncertain",
        top_k=[{"label": "glass", "category": "glass", "probability": 0.44}],
        quality_warnings=["Image is small (194x259)."],
        model_version="2.0.0",
        inference_ms=12.1,
        is_demo=False,
    )
    rec_demo = Prediction(
        id="test-demo-002",
        original_filename="demo_leaf.jpg",
        file_sha256="b" * 64,
        image_width=224,
        image_height=224,
        detected_item="organic",
        category="organic",
        predicted_top1="organic",
        recyclability_class="Compostable",
        bin="Green Organics / Compost Bin",
        confidence=0.88,
        confidence_level="HIGH",
        status="ok",
        top_k=[{"label": "organic", "category": "organic", "probability": 0.88}],
        quality_warnings=[],
        model_version="2.0.0",
        inference_ms=12.5,
        is_demo=True,
    )
    db_session.merge(rec_real)
    db_session.merge(rec_uncertain)
    db_session.merge(rec_demo)
    db_session.commit()

    # Analytics excludes demo by default and handles nullable uncertain category cleanly
    analytics_default = client.get("/api/analytics/summary?include_demo=false").json()
    analytics_with_demo = client.get("/api/analytics/summary?include_demo=true").json()
    assert analytics_with_demo["total_predictions"] == analytics_default["total_predictions"] + 1
    assert analytics_default["uncertain_count"] >= 1

    # Filtered list query
    list_res = client.get("/api/predictions?category=plastic&search=cmd_injection")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # Export CSV verifies exact 10 columns, formula escaping, and Uncertain row handling
    csv_res = client.get("/api/export/csv?include_demo=true")
    assert csv_res.status_code == 200
    lines = csv_res.text.strip().splitlines()
    assert (
        lines[0].strip()
        == "prediction_id,timestamp,filename,detected_item,category,recyclability_class,confidence_score,confidence_level,status,model_version"
    )
    assert "'=cmd_injection.jpg" in csv_res.text
    assert "blurry_bottle.jpg,Uncertain,Uncertain,Unknown" in csv_res.text

    # Delete single item
    del_res = client.delete("/api/predictions/test-demo-002")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Bulk delete requires confirm=true
    no_confirm_res = client.delete("/api/predictions")
    assert no_confirm_res.status_code == 400
    confirm_res = client.delete("/api/predictions?confirm=true&demo_only=true")
    assert confirm_res.status_code == 200


def test_near_uniform_and_high_entropy_abstention():
    """Verify near-uniform (0.34/0.33/0.33) and tight-margin (0.42 vs 0.38) distributions abstain cleanly."""
    from app.ml.inference import normalized_shannon_entropy

    probs_uniform = np.array([0.34, 0.33, 0.33, 0.0, 0.0], dtype=np.float64)
    ent_uniform = normalized_shannon_entropy(probs_uniform)
    st, lvl, reason = evaluate_threshold_and_margin(
        top1_prob=0.34,
        top2_prob=0.33,
        min_thresh=0.50,
        high_thresh=0.80,
        min_margin=0.12,
        norm_entropy=ent_uniform,
        max_entropy=0.82,
    )
    assert st == "uncertain"
    assert lvl == "LOW"
    assert reason is not None

    # Margin rule: 0.42 vs 0.38 -> uncertain
    st2, lvl2, _ = evaluate_threshold_and_margin(
        top1_prob=0.42,
        top2_prob=0.38,
        min_thresh=0.50,
        high_thresh=0.80,
        min_margin=0.12,
        norm_entropy=0.75,
        max_entropy=0.82,
    )
    assert st2 == "uncertain"
    assert lvl2 == "LOW"

    # Entropy gate rule: even if top1 >= 0.50 and margin >= 0.12, norm_entropy > 0.82 triggers uncertain
    st3, lvl3, reason3 = evaluate_threshold_and_margin(
        top1_prob=0.52,
        top2_prob=0.12,
        min_thresh=0.50,
        high_thresh=0.80,
        min_margin=0.12,
        norm_entropy=0.85,
        max_entropy=0.82,
    )
    assert st3 == "uncertain"
    assert lvl3 == "LOW"
    assert "entropy" in (reason3 or "").lower()


def test_class_order_consistency_across_artifacts():
    """Verify classes.json, taxonomy.yaml, model_card.json, and classifier_service share identical canonical class order."""
    import json

    classes_json = json.loads((BASE_DIR / "artifacts" / "models" / "classes.json").read_text(encoding="utf-8"))
    model_card = json.loads((BASE_DIR / "artifacts" / "models" / "model_card.json").read_text(encoding="utf-8"))
    tax = load_taxonomy(settings.TAXONOMY_YAML_PATH)

    canonical = ["glass", "metal", "organic", "paper", "plastic"]
    assert classes_json == canonical
    assert model_card["classes"] == canonical
    assert sorted(tax.categories) == canonical
    classifier_service.ensure_loaded()
    assert classifier_service.classes == canonical


def test_pile_warning_on_can_pile_and_low_single_item_false_positive_rate(client: TestClient):
    """Verify metal_can_pile_01.jpg triggers the pile warning (not dominant background warning) and never predicts organic top-1."""
    pile_path = BASE_DIR / "tests" / "fixtures" / "field" / "metal_can_pile_01.jpg"
    assert pile_path.exists()

    val = validate_and_sanitize_image(pile_path.read_bytes(), original_filename="metal_can_pile_01.jpg")
    joined_warnings = " | ".join(val.quality_warnings)
    assert (
        "This looks like multiple items or a pile. Photograph one item at a time for the most reliable result."
        in joined_warnings
    )
    assert "Dominant background color/texture" not in joined_warnings

    res = client.post(
        "/api/predict",
        files={"file": ("metal_can_pile_01.jpg", pile_path.read_bytes(), "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    # Must never predict organic as top-1 on a pile of metal cans
    assert body["predicted_top1"] != "organic"
    assert body["predicted_top1"] == "metal"
    if body["status"] == "ok":
        assert body["category"] == "metal"
    else:
        assert body["category"] is None
        assert body["disposal_tips"] == []
        assert body["prep_steps"] == []


def test_grayscale_robustness_and_serving_offline_parity():
    """Verify grayscale-converted images produce valid normalized probabilities and match offline preprocessing."""
    pile_path = BASE_DIR / "tests" / "fixtures" / "field" / "metal_can_pile_01.jpg"
    img_rgb = Image.open(pile_path).convert("RGB")
    img_gray = img_rgb.convert("L").convert("RGB")

    # Verify offline letterbox + normalization matches preprocess_pil_image exactly
    t1 = preprocess_pil_image(img_rgb, target_size=224)
    boxed = letterbox_pil_image(img_rgb, target_size=224)
    arr = np.asarray(boxed, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t2 = np.expand_dims(((arr - mean) / std).transpose(2, 0, 1), axis=0).astype(np.float32)
    assert np.allclose(t1, t2, atol=1e-6)

    out_gray = classifier_service.predict(img_gray)
    assert abs(sum(out_gray.probabilities.values()) - 1.0) < 1e-3
    # Grayscale conversion of metal_can_pile_01.jpg must still rank metal top-1
    assert out_gray.top_k[0]["category"] == "metal"


def test_feedback_export_staging_isolation(tmp_path: Path):
    """Verify export_feedback_dataset stages corrected predictions into feedback_staging without modifying train.csv."""
    import subprocess
    import sys

    train_csv = BASE_DIR / "ml" / "data" / "splits" / "train.csv"
    before_sha = hashlib.sha256(train_csv.read_bytes()).hexdigest()

    staging_out = tmp_path / "feedback_staging"
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "ml" / "scripts" / "export_feedback_dataset.py"), "--staging-dir", str(staging_out)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "auto_merged_into_training=False" in proc.stdout
    assert (staging_out / "feedback_manifest.json").exists()
    after_sha = hashlib.sha256(train_csv.read_bytes()).hexdigest()
    assert before_sha == after_sha


def test_classify_text_endpoint_all_11_test_items(client: TestClient):
    """Verify all 11 user-specified items classify accurately with category, bin, explanation, and prep/disposal tips."""
    test_cases = [
        ("plastic bottles", "plastic", "Recyclable", "Blue Recycling Bin"),
        ("newspaper", "paper", "Recyclable", "Blue Recycling Bin"),
        ("cardboard", "paper", "Recyclable", "Blue Recycling Bin"),
        ("metal cans", "metal", "Recyclable", "Blue Recycling Bin"),
        ("glass jars", "glass", "Recyclable", "Blue Recycling Bin"),
        ("banana peels", "organic", "Compostable", "Green Organics"),
        ("batteries", "special_waste", "Special handling", "Battery Recycling Drop-Off"),
        ("old mobile phones", "special_waste", "Special handling", "E-Waste Recycling Depot"),
        ("expired medicines", "special_waste", "Special handling", "Pharmacy Take-Back Kiosk"),
        ("sanitary pads", "special_waste", "Non-recyclable", "General Waste Bin"),
        ("broken glass", "special_waste", "Special handling", "General Waste Bin"),
    ]

    for query, expected_cat, expected_rec, expected_bin_substr in test_cases:
        res = client.post("/api/classify/text", json={"query": query})
        assert res.status_code == 200, f"Failed on {query}: {res.text}"
        data = res.json()
        assert data["status"] == "ok"
        assert data["category"] == expected_cat
        assert data["recyclability_class"] == expected_rec
        assert expected_bin_substr in data["bin"]
        assert data["detected_item"] is not None
        assert len(data["prep_steps"]) > 0
        assert len(data["disposal_tips"]) > 0
        assert data["explanation"] is not None
        assert data["classification_method"] == "text_search"

    # Test unknown item safety fallback
    unk_res = client.post("/api/classify/text", json={"query": "unrecognized space debris unknown item 42"})
    assert unk_res.status_code == 200
    unk_data = unk_res.json()
    assert unk_data["status"] == "uncertain"
    assert unk_data["category"] is None
    assert len(unk_data["safe_handling_advice"]) > 0


def test_taxonomy_endpoint_and_material_directory(client: TestClient):
    """Verify /api/taxonomy provides 5 primary streams, special waste streams, and directory items."""
    res = client.get("/api/taxonomy")
    assert res.status_code == 200
    data = res.json()
    assert "version" in data
    assert "disclaimer" in data
    assert sorted(data["categories"]) == ["glass", "metal", "organic", "paper", "plastic"]
    assert set(data["primary_streams"].keys()) == {"plastic", "paper", "metal", "glass", "organic"}
    assert "batteries" in data["special_waste_streams"]
    assert "electronic_waste" in data["special_waste_streams"]
    assert len(data["material_directory"]) >= 11
    assert len(data["common_mistakes"]) >= 5


def test_analytics_summary_with_special_waste_and_frequent_items(client: TestClient):
    """Verify /api/analytics/summary returns 5-stream breakdown, special waste count, and frequent items."""
    res = client.get("/api/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "special_waste_count" in data
    assert "primary_stream_counts" in data
    assert set(data["primary_stream_counts"].keys()) == {"plastic", "paper", "metal", "glass", "organic"}
    assert "frequently_classified_items" in data
    assert isinstance(data["frequently_classified_items"], list)




