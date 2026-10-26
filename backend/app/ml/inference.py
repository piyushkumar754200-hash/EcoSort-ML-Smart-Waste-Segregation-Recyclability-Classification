"""ONNX Runtime inference service with shared letterbox preprocessing, adaptive 2x2 tile + de-colourised multi-view inference, temperature calibration, and 3-gate (confidence, margin, entropy) abstention."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from app.config import settings
from app.core.security import detect_pile_or_multi_item_scene

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
LETTERBOX_FILL_RGB = (124, 116, 104)
DEFAULT_CLASSES = ["glass", "metal", "organic", "paper", "plastic"]


def letterbox_pil_image(
    pil_image: Image.Image,
    target_size: int = 224,
    fill_color: Tuple[int, int, int] = LETTERBOX_FILL_RGB,
) -> Image.Image:
    """Aspect-ratio-preserving resize + symmetric padding to (target_size, target_size)."""
    rgb = pil_image.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_size, target_size), fill_color)
    scale = float(target_size) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas


def preprocess_pil_image(pil_image: Image.Image, target_size: int = 224) -> np.ndarray:
    """Preprocess PIL image using aspect-ratio-preserving letterbox + ImageNet normalization."""
    boxed = letterbox_pil_image(pil_image, target_size=target_size)
    arr = np.asarray(boxed, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    norm = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(norm, axis=0).astype(np.float32)


def compute_normalized_entropy(probs: np.ndarray | List[float]) -> float:
    """Compute normalized Shannon entropy H(p) / ln(K) in [0.0, 1.0]."""
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr / np.sum(arr)
    k = max(len(arr), 2)
    h = float(-np.sum(arr * np.log(arr)))
    return round(float(h / math.log(k)), 4)


normalized_shannon_entropy = compute_normalized_entropy


def evaluate_threshold_and_margin(
    top1_prob: float,
    top2_prob: float,
    min_thresh: float = 0.50,
    high_thresh: float = 0.80,
    min_margin: float = 0.12,
    normalized_entropy: Optional[float] = None,
    max_entropy: float = 0.82,
    top3_prob: Optional[float] = None,
    norm_entropy: Optional[float] = None,
) -> Tuple[Literal["ok", "uncertain"], Literal["HIGH", "MEDIUM", "LOW"], Optional[str]]:
    """
    Evaluate minimum confidence threshold, top1-top2 margin rule, and normalized entropy rule.
    Returns (status, confidence_level, uncertainty_reason).
    """
    if normalized_entropy is None and norm_entropy is not None:
        normalized_entropy = norm_entropy
    margin = float(top1_prob - top2_prob)
    near_equal_top3 = (top3_prob is not None) and ((top1_prob - float(top3_prob)) < 0.10)

    if near_equal_top3 or (normalized_entropy is not None and normalized_entropy > max_entropy):
        ent_str = f" (normalized entropy {normalized_entropy:.2f} > {max_entropy:.2f})" if normalized_entropy is not None else ""
        return (
            "uncertain",
            "LOW",
            f"The model could not distinguish between these materials{ent_str}. Top candidates have nearly equal probabilities.",
        )

    if top1_prob < min_thresh:
        return (
            "uncertain",
            "LOW",
            f"Top-1 confidence ({top1_prob * 100:.1f}%) is below the minimum certainty threshold ({min_thresh * 100:.0f}%).",
        )
    if margin < min_margin:
        return (
            "uncertain",
            "LOW",
            f"Top-2 candidate margin ({margin * 100:.1f}%) is below the minimum separation margin ({min_margin * 100:.0f}%).",
        )
    if top1_prob >= high_thresh:
        return ("ok", "HIGH", None)
    return ("ok", "MEDIUM", None)


@dataclass
class InferenceOutput:
    category: str
    confidence: float
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"]
    status: Literal["ok", "uncertain"]
    top_k: List[Dict[str, Any]]
    probabilities: Dict[str, float]
    inference_ms: float
    model_version: str
    is_smoke_test: bool
    uncertainty_reason: Optional[str] = None
    entropy: float = 0.0
    is_near_uniform: bool = False
    is_pile_detected: bool = False

    @property
    def top_label(self) -> str:
        return self.category


class WasteClassifierService:
    def __init__(self) -> None:
        self._session: Optional[ort.InferenceSession] = None
        self._model_card: Dict[str, Any] = {}
        self._loaded_mtime: float = 0.0
        self.class_names: List[str] = list(DEFAULT_CLASSES)
        self.temperature: float = 1.0
        self.image_size: int = 224
        self.model_version: str = "model not loaded"
        self.is_smoke_test: bool = False

    @property
    def classes(self) -> List[str]:
        return self.class_names

    @property
    def model_card(self) -> Dict[str, Any]:
        return self._model_card

    def load(self) -> bool:
        return self.ensure_loaded()

    def ensure_loaded(self) -> bool:
        onnx_path = Path(settings.MODEL_ONNX_PATH)
        card_path = Path(settings.MODEL_CARD_PATH)
        if not onnx_path.exists():
            self._session = None
            self.model_version = "model not loaded"
            return False

        mtime = onnx_path.stat().st_mtime
        if self._session is not None and mtime == self._loaded_mtime:
            return True

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._loaded_mtime = mtime

            if card_path.exists():
                with card_path.open("r", encoding="utf-8") as f:
                    self._model_card = json.load(f)
            else:
                self._model_card = {}

            self.class_names = self._model_card.get("class_names", list(DEFAULT_CLASSES))
            self.temperature = float(self._model_card.get("temperature", 1.0))
            self.image_size = int(self._model_card.get("input_size", 224))
            self.model_version = str(self._model_card.get("model_version", "3.0.0"))
            self.is_smoke_test = bool(self._model_card.get("is_smoke_test", False))
            return True
        except Exception:
            self._session = None
            self.model_version = "model not loaded"
            return False

    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(self.temperature, 1e-3)
        shifted = scaled - np.max(scaled)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict(self, pil_image: Image.Image) -> InferenceOutput:
        if not self.ensure_loaded() or self._session is None:
            raise RuntimeError("ONNX model is not loaded.")

        t0 = time.perf_counter()
        rgb = pil_image.convert("RGB")
        x_orig = preprocess_pil_image(rgb, target_size=self.image_size)
        x_flip = preprocess_pil_image(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT), target_size=self.image_size)

        input_name = self._session.get_inputs()[0].name
        logits_orig = self._session.run(None, {input_name: x_orig})[0][0]
        logits_flip = self._session.run(None, {input_name: x_flip})[0][0]
        logits_full = 0.65 * logits_orig + 0.35 * logits_flip

        probs_full = self._logits_to_probs(logits_full)
        initial_entropy = compute_normalized_entropy(probs_full)
        is_pile, _ = detect_pile_or_multi_item_scene(rgb)

        # Adaptive Multi-View (5 overlapping crops + grayscale de-colourised view) when scene is a pile or high-entropy
        w, h = rgb.size
        if (is_pile or initial_entropy > 0.72) and w >= 48 and h >= 48:
            scale = 0.55 if is_pile else 0.58
            cw = max(24, int(round(w * scale)))
            ch = max(24, int(round(h * scale)))
            cx0 = max(0, (w - cw) // 2)
            cy0 = max(0, (h - ch) // 2)
            crops = [
                rgb.crop((0, 0, cw, ch)),
                rgb.crop((w - cw, 0, w, ch)),
                rgb.crop((0, h - ch, cw, h)),
                rgb.crop((w - cw, h - ch, w, h)),
                rgb.crop((cx0, cy0, cx0 + cw, cy0 + ch)),
            ]
            gray_full = ImageOps.grayscale(rgb).convert("RGB")
            batch_views = np.concatenate(
                [preprocess_pil_image(c, target_size=self.image_size) for c in crops]
                + [preprocess_pil_image(gray_full, target_size=self.image_size)],
                axis=0,
            )
            multi_logits = self._session.run(None, {input_name: batch_views})[0]
            tiles_logits = np.mean(multi_logits[:5], axis=0)
            gray_logits = multi_logits[5]
            if is_pile:
                logits = 0.40 * logits_full + 0.30 * tiles_logits + 0.30 * gray_logits
            else:
                logits = 0.50 * logits_full + 0.30 * tiles_logits + 0.20 * gray_logits
            probs = self._logits_to_probs(logits)
        else:
            probs = probs_full

        inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        ranked_indices = np.argsort(probs)[::-1]
        top1_idx = int(ranked_indices[0])
        top2_idx = int(ranked_indices[1]) if len(ranked_indices) > 1 else top1_idx
        top3_idx = int(ranked_indices[2]) if len(ranked_indices) > 2 else top2_idx

        top1_cat = self.class_names[top1_idx]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        top3_prob = float(probs[top3_idx])
        norm_entropy = compute_normalized_entropy(probs)
        is_near_uniform = bool((top1_prob - top3_prob) < 0.10 or norm_entropy > float(settings.ENTROPY_MAX))

        status, conf_level, uncertainty_reason = evaluate_threshold_and_margin(
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            min_thresh=float(settings.CONFIDENCE_MIN),
            high_thresh=float(settings.CONFIDENCE_HIGH),
            min_margin=float(settings.MARGIN_MIN),
            normalized_entropy=norm_entropy,
            max_entropy=float(settings.ENTROPY_MAX),
            top3_prob=top3_prob,
        )

        top_k = [
            {
                "label": self.class_names[int(idx)],
                "category": self.class_names[int(idx)],
                "probability": round(float(probs[int(idx)]), 4),
            }
            for idx in ranked_indices[:3]
        ]
        prob_map = {
            self.class_names[i]: round(float(probs[i]), 4)
            for i in range(len(self.class_names))
        }

        return InferenceOutput(
            category=top1_cat,
            confidence=round(top1_prob, 4),
            confidence_level=conf_level,
            status=status,
            top_k=top_k,
            probabilities=prob_map,
            inference_ms=inference_ms,
            model_version=self.model_version,
            is_smoke_test=self.is_smoke_test,
            uncertainty_reason=uncertainty_reason,
            entropy=norm_entropy,
            is_near_uniform=is_near_uniform,
            is_pile_detected=is_pile,
        )

    def predict_sync(self, pil_image: Image.Image) -> InferenceOutput:
        return self.predict(pil_image)


classifier_service = WasteClassifierService()
