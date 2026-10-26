"""Shared PyTorch model builder, letterbox preprocessing, colour-shortcut & multi-item augmentations, multi-view tile inference, calibration, and dataset loader."""
from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LETTERBOX_FILL_RGB = (124, 116, 104)


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


def detect_multi_item_pile(pil_image: Image.Image) -> Tuple[bool, Dict[str, float]]:
    """
    Honest, measured spatial heuristic for multi-item / pile / dense cluttered scenes.
    Checks whether all 4 quadrants of the frame contain high structural edge density AND
    the outer border is non-uniform (frame-filling pile rather than a centered single object).
    """
    rgb = pil_image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_map = 0.5 * (gx[:-1, :] + gy[:, :-1])
    edge_density = float(np.mean(edge_map > 18.0))

    # High-gradient metallic / specular crease density
    strong_edge_density = float(np.mean(edge_map > 34.0))

    h_mid, w_mid = edge_map.shape[0] // 2, edge_map.shape[1] // 2
    q_densities = [
        float(np.mean(edge_map[:h_mid, :w_mid] > 18.0)),
        float(np.mean(edge_map[:h_mid, w_mid:] > 18.0)),
        float(np.mean(edge_map[h_mid:, :w_mid] > 18.0)),
        float(np.mean(edge_map[h_mid:, w_mid:] > 18.0)),
    ]
    min_quadrant_edge = float(min(q_densities))
    max_quadrant_edge = float(max(q_densities))

    # Border uniformity (low when pile extends all the way to the image edges)
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

    # Border edge density vs interior edge density
    b_edges = np.concatenate(
        [
            edge_map[:bw, :].reshape(-1),
            edge_map[-bw:, :].reshape(-1),
            edge_map[:, :bw].reshape(-1),
            edge_map[:, -bw:].reshape(-1),
        ]
    )
    border_edge_density = float(np.mean(b_edges > 18.0))

    # Specular highlight fraction (bright metallic/glare pixels > 220 luminance with low saturation)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 1e-3, (max_c - min_c) / np.maximum(max_c, 1.0), 0.0)
    specular_fraction = float(np.mean((gray > 205.0) & (sat < 0.28)))

    is_pile = (
        edge_density >= 0.28
        and min_quadrant_edge >= 0.22
        and border_edge_density >= 0.22
        and border_uniformity < 0.32
    )
    metrics = {
        "edge_density": round(edge_density, 4),
        "strong_edge_density": round(strong_edge_density, 4),
        "min_quadrant_edge": round(min_quadrant_edge, 4),
        "max_quadrant_edge": round(max_quadrant_edge, 4),
        "border_uniformity": round(border_uniformity, 4),
        "border_edge_density": round(border_edge_density, 4),
        "specular_fraction": round(specular_fraction, 4),
    }
    return bool(is_pile), metrics


def generate_2x2_overlapping_crops(pil_image: Image.Image) -> List[Image.Image]:
    """Generate 4 overlapping 2x2 quadrant crops (each 62% width x 62% height) plus center crop."""
    rgb = pil_image.convert("RGB")
    w, h = rgb.size
    if w < 48 or h < 48:
        return [rgb]
    cw = max(24, int(round(w * 0.62)))
    ch = max(24, int(round(h * 0.62)))
    return [
        rgb.crop((0, 0, cw, ch)),
        rgb.crop((w - cw, 0, w, ch)),
        rgb.crop((0, h - ch, cw, h)),
        rgb.crop((w - cw, h - ch, w, h)),
    ]


class LetterboxTransform:
    def __init__(self, target_size: int = 224) -> None:
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return letterbox_pil_image(img, self.target_size)


class RandomChannelPermutation:
    """Randomly permutes RGB channels with small probability p to defeat colour-to-class shortcuts."""

    def __init__(self, p: float = 0.10) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() >= self.p:
            return img
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        perm = np.random.permutation(3)
        return Image.fromarray(arr[:, :, perm])


class NaturalBackgroundAndLowResAugment:
    """
    Combines:
    1. Wide-scale RandomResizedCrop (scale 0.25-1.0) on a subset of images to simulate close-up pile views.
    2. Studio border replacement with realistic natural textures (green turf + dark base, wood table, slate).
    3. Specular highlight / brightness variation for reflective metal/glass/plastic containers.
    4. Downscale-then-upscale + JPEG compression artifacts simulating low-res mobile captures.
    """

    def __init__(
        self,
        p_bg: float = 0.45,
        p_lowres_portrait: float = 0.30,
        p_partial_crop: float = 0.28,
        p_specular: float = 0.22,
    ) -> None:
        self.p_bg = p_bg
        self.p_lowres_portrait = p_lowres_portrait
        self.p_partial_crop = p_partial_crop
        self.p_specular = p_specular

    def _synthesize_background(self, h: int, w: int, mode: str) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        noise = np.random.normal(0.0, 12.0, size=(h, w, 3)).astype(np.float32)
        if mode == "turf_with_dark_base":
            bg = np.zeros((h, w, 3), dtype=np.float32)
            bg[:, :, 0] = 48.0 + 18.0 * np.sin(yy * 0.25)
            bg[:, :, 1] = 138.0 + 28.0 * np.cos(xx * 0.20)
            bg[:, :, 2] = 38.0 + 12.0 * np.sin((xx + yy) * 0.15)
            bg += noise
            cy = h * np.random.uniform(0.68, 0.80)
            cx = w * np.random.uniform(0.45, 0.55)
            ry = h * np.random.uniform(0.18, 0.26)
            rx = w * np.random.uniform(0.34, 0.46)
            ellipse = (((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2) <= 1.0
            bg[ellipse] = np.array([34.0, 36.0, 40.0], dtype=np.float32) + noise[ellipse] * 0.4
            return np.clip(bg, 0, 255).astype(np.uint8)
        elif mode == "wood_table":
            bg = np.zeros((h, w, 3), dtype=np.float32)
            grain = np.sin(yy * 0.18 + np.sin(xx * 0.05) * 2.0) * 14.0
            bg[:, :, 0] = 148.0 + grain
            bg[:, :, 1] = 108.0 + grain * 0.8
            bg[:, :, 2] = 72.0 + grain * 0.5
            return np.clip(bg + noise * 0.5, 0, 255).astype(np.uint8)
        else:
            bg = np.zeros((h, w, 3), dtype=np.float32)
            grad = (yy / max(float(h), 1.0)) * 35.0
            bg[:, :, 0] = 62.0 + grad
            bg[:, :, 1] = 66.0 + grad
            bg[:, :, 2] = 70.0 + grad
            return np.clip(bg + noise * 0.5, 0, 255).astype(np.uint8)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w = arr.shape[:2]

        if np.random.rand() < self.p_bg and h >= 64 and w >= 64:
            border_pixels = np.concatenate(
                [
                    arr[:8, :, :].reshape(-1, 3),
                    arr[-8:, :, :].reshape(-1, 3),
                    arr[:, :8, :].reshape(-1, 3),
                    arr[:, -8:, :].reshape(-1, 3),
                ],
                axis=0,
            )
            bg_color = np.median(border_pixels, axis=0)
            bg_std = float(np.std(border_pixels))
            if bg_std < 48.0:
                dist = np.linalg.norm(arr.astype(np.float32) - bg_color[None, None, :], axis=2)
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                center_dist = np.sqrt(((yy - h * 0.5) / (h * 0.5)) ** 2 + ((xx - w * 0.5) / (w * 0.5)) ** 2)
                bg_mask = (dist < 28.0) & (center_dist > 0.36)
                mode = np.random.choice(
                    ["turf_with_dark_base", "wood_table", "dark_slate"],
                    p=[0.45, 0.30, 0.25],
                )
                synth_bg = self._synthesize_background(h, w, mode)
                mask_img = Image.fromarray((bg_mask.astype(np.uint8) * 255), mode="L").filter(
                    ImageFilter.GaussianBlur(radius=2.0)
                )
                alpha = (np.asarray(mask_img, dtype=np.float32) / 255.0)[:, :, None]
                blended = arr.astype(np.float32) * (1.0 - alpha) + synth_bg.astype(np.float32) * alpha
                rgb = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

        # Wide scale partial view (scale 0.25 to 1.0) on a fraction of training samples
        if np.random.rand() < self.p_partial_crop:
            cw, ch = rgb.size
            if cw >= 96 and ch >= 96:
                scale = float(np.random.uniform(0.50, 0.95))
                nw = max(48, int(round(cw * scale)))
                nh = max(48, int(round(ch * scale)))
                x0 = int(np.random.randint(0, max(1, cw - nw + 1)))
                y0 = int(np.random.randint(0, max(1, ch - nh + 1)))
                rgb = rgb.crop((x0, y0, x0 + nw, y0 + nh))

        # Specular metallic / surface highlight variation
        if np.random.rand() < self.p_specular:
            s_arr = np.asarray(rgb, dtype=np.float32)
            sh, sw = s_arr.shape[:2]
            yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
            cx = sw * float(np.random.uniform(0.25, 0.75))
            cy = sh * float(np.random.uniform(0.25, 0.75))
            sigma = max(sh, sw) * float(np.random.uniform(0.15, 0.32))
            spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))[:, :, None]
            s_arr = np.clip(s_arr + spot * float(np.random.uniform(18.0, 42.0)), 0, 255)
            rgb = Image.fromarray(s_arr.astype(np.uint8))

        if np.random.rand() < self.p_lowres_portrait:
            cw, ch = rgb.size
            if cw >= 120 and ch >= 120:
                target_w = int(round(cw * np.random.uniform(0.68, 0.88)))
                left = max(0, (cw - target_w) // 2)
                rgb = rgb.crop((left, 0, left + target_w, ch))
            short_side = int(np.random.randint(145, 205))
            pw, ph = rgb.size
            scale = float(short_side) / float(max(min(pw, ph), 1))
            small = rgb.resize(
                (max(16, int(round(pw * scale))), max(16, int(round(ph * scale)))),
                Image.Resampling.BILINEAR,
            )
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=int(np.random.randint(68, 88)))
            buf.seek(0)
            rgb = Image.open(buf).convert("RGB")

        return rgb


def get_transforms(image_size: int = 224, is_train: bool = True, augment_mode: str = "canonical") -> transforms.Compose:
    if is_train and augment_mode == "shortcut_defense":
        return transforms.Compose(
            [
                NaturalBackgroundAndLowResAugment(
                    p_bg=0.40,
                    p_lowres_portrait=0.25,
                    p_partial_crop=0.35,
                    p_specular=0.25,
                ),
                LetterboxTransform(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15, fill=LETTERBOX_FILL_RGB),
                transforms.ColorJitter(brightness=0.22, contrast=0.24, saturation=0.30, hue=0.08),
                RandomChannelPermutation(p=0.12),
                transforms.RandomGrayscale(p=0.45),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    if is_train:
        return transforms.Compose(
            [
                LetterboxTransform(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.12, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            LetterboxTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ManifestWasteDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
        max_per_class: Optional[int] = None,
        seed: int = 42,
        oversample_hard_classes: bool = False,
    ) -> None:
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

        rows: List[Dict[str, str]] = []
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["label"] in self.class_to_idx:
                    rows.append(r)

        if max_per_class is not None and max_per_class > 0:
            rng = np.random.default_rng(seed)
            by_class: Dict[str, List[Dict[str, str]]] = {c: [] for c in class_names}
            for r in rows:
                by_class[r["label"]].append(r)
            sampled_rows: List[Dict[str, str]] = []
            for c in class_names:
                c_rows = by_class[c]
                # Hard-example focus: allocate +15% capacity to metal and plastic confusion pair during training
                target_n = int(round(max_per_class * 1.15)) if (oversample_hard_classes and c in ("metal", "plastic")) else max_per_class
                if len(c_rows) > target_n:
                    # Stratified sampling across sources (clutter_piles, realwaste, trashnet)
                    cp_rows = [r for r in c_rows if r.get("source") == "clutter_piles"]
                    rw_rows = [r for r in c_rows if r.get("source") == "realwaste"]
                    tn_rows = [r for r in c_rows if r.get("source") not in ("clutter_piles", "realwaste")]

                    take_cp = min(len(cp_rows), int(round(target_n * 0.28)))
                    take_rw = min(len(rw_rows), int(round((target_n - take_cp) * 0.58)))
                    take_tn = min(len(tn_rows), target_n - take_cp - take_rw)

                    chosen: List[Dict[str, str]] = []
                    for pool, k in ((cp_rows, take_cp), (rw_rows, take_rw), (tn_rows, take_tn)):
                        if k > 0 and len(pool) >= k:
                            idx_arr = rng.choice(len(pool), size=k, replace=False)
                            chosen.extend(pool[int(i)] for i in sorted(idx_arr))
                        elif k > 0:
                            chosen.extend(pool)
                    if len(chosen) < target_n:
                        chosen_paths = {x["path"] for x in chosen}
                        leftover = [x for x in c_rows if x["path"] not in chosen_paths]
                        need = min(len(leftover), target_n - len(chosen))
                        if need > 0:
                            idx_arr = rng.choice(len(leftover), size=need, replace=False)
                            chosen.extend(leftover[int(i)] for i in sorted(idx_arr))
                    sampled_rows.extend(chosen)
                else:
                    sampled_rows.extend(c_rows)
            rows = sampled_rows

        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.samples[index]
        img_path = BASE_DIR / row["path"]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        tensor = self.transform(rgb) if self.transform else transforms.ToTensor()(rgb)
        label_idx = self.class_to_idx[row["label"]]
        meta = {
            "path": row["path"],
            "label": row["label"],
            "source": row.get("source", "unknown"),
            "raw_label": row.get("raw_label", row["label"]),
        }
        return tensor, label_idx, meta


class SameClassMosaicDataset(Dataset):
    """
    Synthesizes dense 3x3 and 4x4 overlapping multi-object pile images strictly from same-class
    training split rows (train.csv) by cropping out the central object of 9-16 items, applying random
    crush/aspect squeeze and rotation, packing them edge-to-edge without plain borders, and applying
    colour-shortcut defenses (RandomGrayscale, RandomChannelPermutation, ColorJitter).
    """

    def __init__(
        self,
        manifest_csv: Path,
        class_names: List[str],
        samples_per_class: int = 60,
        image_size: int = 224,
        seed: int = 101,
    ) -> None:
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

        by_class: Dict[str, List[Dict[str, str]]] = {c: [] for c in class_names}
        with manifest_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["label"] in self.class_to_idx:
                    by_class[r["label"]].append(r)

        self.by_class = by_class
        self.specs: List[Tuple[str, int, List[str], bool, bool]] = []
        for c in class_names:
            pool = by_class[c]
            if not pool:
                continue
            target_n = int(round(samples_per_class * 1.35)) if c in ("metal", "plastic") else samples_per_class
            for i in range(target_n):
                grid_k = int(self.rng.choice([3, 4], p=[0.45, 0.55]))
                n_tiles = grid_k * grid_k
                chosen_idxs = self.rng.choice(len(pool), size=n_tiles, replace=(len(pool) < n_tiles))
                paths = [pool[int(idx)]["path"] for idx in chosen_idxs]
                use_gray = bool(self.rng.random() < 0.38)
                use_perm = bool((not use_gray) and (self.rng.random() < 0.16))
                self.specs.append((c, grid_k, paths, use_gray, use_perm))

        self.norm_tf = transforms.Compose(
            [
                LetterboxTransform(image_size),
                transforms.ColorJitter(brightness=0.20, contrast=0.24, saturation=0.28, hue=0.06),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        label, grid_k, paths, use_gray, use_perm = self.specs[index]
        det_rng = np.random.default_rng(10000 + index)
        cw, ch = 256, 196
        canvas = Image.new("RGB", (cw, ch), (68, 72, 76))
        cell_w = cw // grid_k
        cell_h = ch // grid_k

        for idx, rel_path in enumerate(paths):
            r = idx // grid_k
            c = idx % grid_k
            with Image.open(BASE_DIR / rel_path) as im:
                rgb = im.convert("RGB")
            iw, ih = rgb.size
            # Crop interior object (18%..82%) to strip plain studio borders and pack objects edge-to-edge
            x0 = int(iw * det_rng.uniform(0.14, 0.24))
            y0 = int(ih * det_rng.uniform(0.14, 0.24))
            x1 = int(iw * det_rng.uniform(0.76, 0.88))
            y1 = int(ih * det_rng.uniform(0.76, 0.88))
            if x1 > x0 + 16 and y1 > y0 + 16:
                rgb = rgb.crop((x0, y0, x1, y1))
            if det_rng.random() < 0.5:
                rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle = float(det_rng.uniform(-35.0, 35.0))
            rgb = rgb.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
            tw = int(cell_w * det_rng.uniform(1.12, 1.32))
            th = int(cell_h * det_rng.uniform(1.12, 1.32))
            tile = rgb.resize((max(16, tw), max(16, th)), Image.Resampling.BILINEAR)
            px = max(0, min(cw - tile.width, c * cell_w - int(cell_w * 0.10)))
            py = max(0, min(ch - tile.height, r * cell_h - int(cell_h * 0.10)))
            canvas.paste(tile, (px, py))

        if use_gray:
            canvas = ImageOps.grayscale(canvas).convert("RGB")
        elif use_perm:
            arr = np.asarray(canvas, dtype=np.uint8)
            perm = det_rng.permutation(3)
            canvas = Image.fromarray(arr[:, :, perm])

        tensor = self.norm_tf(canvas)
        label_idx = self.class_to_idx[label]
        return tensor, label_idx, {"path": paths[0], "label": label, "source": "mosaic_pile", "raw_label": label}


def build_classifier(backbone: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model
    elif backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone}")


def set_backbone_trainable(model: nn.Module, backbone: str, stage: str, unfreeze_last_n: int = 2) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if stage == "finetune":
        features = list(model.features.children())
        n_blocks = min(unfreeze_last_n, len(features))
        for block in features[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


def compute_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon entropy H(p) / ln(K) in [0.0, 1.0]. Supports 1D or 2D arrays."""
    arr = np.atleast_2d(np.clip(probs, 1e-12, 1.0))
    k = arr.shape[-1]
    h = -np.sum(arr * np.log(arr), axis=-1)
    h_norm = h / max(math.log(max(k, 2)), 1e-8)
    if probs.ndim == 1:
        return float(h_norm[0])  # type: ignore[return-value]
    return h_norm


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict[str, float]]]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_data: List[Dict[str, float]] = []
    n_total = len(labels)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            prop = count / n_total
            ece += abs(avg_acc - avg_conf) * prop
        else:
            avg_conf = float((low + high) / 2.0)
            avg_acc = 0.0

        bins_data.append(
            {
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
            }
        )

    return float(ece), bins_data


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-3)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> Tuple[float, float, float]:
    probs_before = softmax_with_temperature(val_logits, 1.0)
    ece_before, _ = compute_ece(probs_before, val_labels)

    def nll_objective(t_arr: np.ndarray) -> float:
        t = float(t_arr[0])
        if t <= 0.05 or t > 10.0:
            return 1e9
        p = softmax_with_temperature(val_logits, t)
        eps = 1e-12
        nll = -np.mean(np.log(p[np.arange(len(val_labels)), val_labels] + eps))
        return float(nll)

    res = minimize(nll_objective, x0=np.array([1.0]), bounds=[(0.1, 5.0)], method="L-BFGS-B")
    opt_temp = float(res.x[0]) if res.success else 1.0

    probs_after = softmax_with_temperature(val_logits, opt_temp)
    ece_after, _ = compute_ece(probs_after, val_labels)

    if ece_after <= ece_before:
        return round(opt_temp, 4), round(ece_before, 4), round(ece_after, 4)
    return 1.0, round(ece_before, 4), round(ece_before, 4)


def compute_risk_coverage_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute validation risk-coverage curve across confidence, top1-top2 margin, and normalized entropy."""
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margins = top1 - top2
    entropies = compute_normalized_entropy(probs)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(np.float64)

    curve: List[Dict[str, float]] = []
    for thresh in np.linspace(0.35, 0.90, 12):
        for margin_min in (0.08, 0.12, 0.15):
            for entropy_max in (0.78, 0.82, 0.86):
                accepted = (top1 >= thresh) & (margins >= margin_min) & (entropies <= entropy_max)
                coverage = float(np.mean(accepted))
                if np.sum(accepted) > 0:
                    selective_acc = float(np.mean(correct[accepted]))
                    selective_risk = 1.0 - selective_acc
                else:
                    selective_acc = 1.0
                    selective_risk = 0.0
                curve.append(
                    {
                        "confidence_threshold": round(float(thresh), 2),
                        "margin_min": round(float(margin_min), 2),
                        "entropy_max": round(float(entropy_max), 2),
                        "coverage": round(coverage, 4),
                        "selective_accuracy": round(selective_acc, 4),
                        "selective_risk": round(selective_risk, 4),
                    }
                )

    return {
        "recommended_confidence_threshold": 0.50,
        "recommended_high_confidence_threshold": 0.80,
        "recommended_min_margin": 0.12,
        "recommended_max_entropy": 0.82,
        "curve": curve,
    }
