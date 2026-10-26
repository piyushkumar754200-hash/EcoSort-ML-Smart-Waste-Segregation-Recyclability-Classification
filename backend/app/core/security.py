"""Image validation, magic-byte inspection, EXIF stripping, quality diagnostics, and CSV sanitization."""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
from typing import List, Optional
import uuid

import numpy as np
from PIL import Image, ImageFilter

from app.config import settings

MAX_DIMENSION = 8192
MIN_RECOMMENDED_SHORTER_SIDE = 224


class ImageValidationError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class ValidatedImage:
    clean_rgb_image: Image.Image
    sanitized_filename: str
    file_sha256: str
    width: int
    height: int
    format_name: str
    quality_warnings: List[str]


def sanitize_filename(filename: Optional[str]) -> str:
    if not filename:
        return "upload.jpg"
    base = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    cleaned = cleaned.lstrip(".")
    if not cleaned:
        return "upload.jpg"
    return cleaned[:180]


def sanitize_csv_cell(value: object) -> str:
    s = "" if value is None else str(value)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


def _verify_magic_bytes(raw_bytes: bytes) -> str:
    if len(raw_bytes) < 12:
        raise ImageValidationError(
            status_code=422,
            code="CORRUPT_IMAGE",
            message="Uploaded file is too short to be a valid image.",
        )
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "WEBP"
    raise ImageValidationError(
        status_code=415,
        code="UNSUPPORTED_MEDIA_TYPE",
        message="Unsupported image format. Only JPEG, PNG, and WEBP files are accepted.",
    )


def detect_pile_or_multi_item_scene(rgb_image: Image.Image) -> tuple[bool, dict]:
    """
    Measured spatial edge & quadrant-activity detector for multi-item / pile scenes.
    Returns (is_pile, metrics_dict).
    """
    small = rgb_image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_map = 0.5 * (gx[:-1, :] + gy[:, :-1])
    edge_density = float(np.mean(edge_map > 18.0))

    h_mid, w_mid = edge_map.shape[0] // 2, edge_map.shape[1] // 2
    q_densities = [
        float(np.mean(edge_map[:h_mid, :w_mid] > 18.0)),
        float(np.mean(edge_map[:h_mid, w_mid:] > 18.0)),
        float(np.mean(edge_map[h_mid:, :w_mid] > 18.0)),
        float(np.mean(edge_map[h_mid:, w_mid:] > 18.0)),
    ]
    min_quadrant_edge = float(min(q_densities))

    bw = 12
    border = np.concatenate(
        [
            arr[:bw, :, :].reshape(-1, 3),
            arr[-bw:, :, :].reshape(-1, 3),
            arr[bw:-bw, :bw, :].reshape(-1, 3),
            arr[bw:-bw, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(border, axis=0)
    dist = np.linalg.norm(border - med, axis=1)
    border_uniformity = float(np.mean(dist < 28.0))

    b_edges = np.concatenate(
        [
            edge_map[:bw, :].reshape(-1),
            edge_map[-bw:, :].reshape(-1),
            edge_map[:, :bw].reshape(-1),
            edge_map[:, -bw:].reshape(-1),
        ]
    )
    border_edge_density = float(np.mean(b_edges > 18.0))

    is_pile = (
        edge_density >= 0.28
        and min_quadrant_edge >= 0.22
        and border_edge_density >= 0.22
        and border_uniformity < 0.32
    )
    return bool(is_pile), {
        "edge_density": round(edge_density, 4),
        "min_quadrant_edge": round(min_quadrant_edge, 4),
        "border_uniformity": round(border_uniformity, 4),
        "border_edge_density": round(border_edge_density, 4),
    }


def compute_quality_warnings(rgb_image: Image.Image) -> List[str]:
    warnings: List[str] = []
    w, h = rgb_image.size
    shorter = min(w, h)
    if shorter < MIN_RECOMMENDED_SHORTER_SIDE:
        warnings.append(
            f"Image is small ({w}x{h}). Larger photos (>= 224 px on the shorter side) usually give better results."
        )

    gray = np.asarray(rgb_image.convert("L"), dtype=np.float32)
    mean_lum = float(np.mean(gray))
    if mean_lum < 35.0:
        warnings.append(
            f"Image appears underexposed (brightness {mean_lum:.1f}/255). Brighter lighting improves material classification."
        )
    elif mean_lum > 235.0:
        warnings.append(
            f"Image appears overexposed (brightness {mean_lum:.1f}/255). Reduce glare or direct flash."
        )

    edges = rgb_image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_var = float(np.var(np.asarray(edges, dtype=np.float32)))
    if edge_var < 18.0:
        warnings.append(
            f"Image appears blurry or low-contrast (sharpness score {edge_var:.1f}). Hold the camera steady and focus on the item."
        )

    is_pile, pile_metrics = detect_pile_or_multi_item_scene(rgb_image)
    if is_pile:
        warnings.append(
            "This looks like multiple items or a pile. Photograph one item at a time for the most reliable result."
        )

    # Check if peripheral border has a dominant saturated background around a centered item (not a full-frame pile)
    arr = np.asarray(rgb_image, dtype=np.float32)
    if not is_pile and h >= 32 and w >= 32:
        bw = max(4, min(h, w) // 10)
        border = np.concatenate(
            [
                arr[:bw, :, :].reshape(-1, 3),
                arr[-bw:, :, :].reshape(-1, 3),
                arr[:, :bw, :].reshape(-1, 3),
                arr[:, -bw:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        max_c = np.max(border, axis=1)
        min_c = np.min(border, axis=1)
        sat = np.where(max_c > 1e-3, (max_c - min_c) / np.maximum(max_c, 1.0), 0.0)
        mean_border_sat = float(np.mean(sat))
        if mean_border_sat > 0.36 and pile_metrics["border_uniformity"] >= 0.32:
            warnings.append(
                "Dominant background color/texture detected around the edges. Placing the item on a plain, neutral surface improves accuracy on transparent items."
            )

    return warnings


def validate_and_sanitize_image(raw_bytes: bytes, original_filename: Optional[str] = None) -> ValidatedImage:
    if not raw_bytes or len(raw_bytes) == 0:
        raise ImageValidationError(
            status_code=400,
            code="EMPTY_FILE",
            message="Uploaded file is empty (0 bytes).",
        )

    max_bytes = int(settings.MAX_UPLOAD_MB * 1024 * 1024)
    if len(raw_bytes) > max_bytes:
        raise ImageValidationError(
            status_code=413,
            code="PAYLOAD_TOO_LARGE",
            message=f"File size ({len(raw_bytes) / (1024 * 1024):.2f} MB) exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
            details={"max_upload_mb": settings.MAX_UPLOAD_MB},
        )

    fmt = _verify_magic_bytes(raw_bytes)
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    safe_name = sanitize_filename(original_filename)

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img_probe:
            img_probe.verify()

        with Image.open(io.BytesIO(raw_bytes)) as img:
            w, h = img.size
            if w > MAX_DIMENSION or h > MAX_DIMENSION:
                raise ImageValidationError(
                    status_code=422,
                    code="DIMENSIONS_TOO_LARGE",
                    message=f"Image dimensions ({w}x{h}) exceed maximum allowed ({MAX_DIMENSION}x{MAX_DIMENSION}).",
                )
            rgb_data = img.convert("RGB")
            clean_rgb = Image.new("RGB", rgb_data.size)
            clean_rgb.putdata(list(rgb_data.getdata()))
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(
            status_code=422,
            code="CORRUPT_IMAGE",
            message=f"Image file could not be decoded: {exc}",
        ) from exc

    warnings = compute_quality_warnings(clean_rgb)
    return ValidatedImage(
        clean_rgb_image=clean_rgb,
        sanitized_filename=safe_name,
        file_sha256=file_sha256,
        width=clean_rgb.width,
        height=clean_rgb.height,
        format_name=fmt,
        quality_warnings=warnings,
    )


def save_sanitized_thumbnail(clean_rgb: Image.Image) -> Optional[str]:
    if not settings.STORE_THUMBNAILS:
        return None
    thumb_dir = Path(settings.THUMBNAIL_DIR)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb = clean_rgb.copy()
    thumb.thumbnail((settings.THUMBNAIL_MAX_DIM, settings.THUMBNAIL_MAX_DIM), Image.Resampling.BILINEAR)
    name = f"{uuid.uuid4().hex}.jpg"
    out_path = thumb_dir / name
    thumb.save(out_path, format="JPEG", quality=82)
    return name
