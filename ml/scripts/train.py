"""Config-driven, reproducible two-stage transfer learning trainer for EcoSort ML."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
import torchvision  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(4)
torch.backends.mkldnn.enabled = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_factory import (  # noqa: E402
    BASE_DIR,
    ManifestWasteDataset,
    build_classifier,
    fit_temperature_scaling,
    get_transforms,
)


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_class_weights(dataset: ManifestWasteDataset, num_classes: int) -> torch.Tensor:
    counts = Counter(dataset.class_to_idx[r["label"]] for r in dataset.samples)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        c = max(counts.get(i, 1), 1)
        weights.append(total / (num_classes * c))
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    return w_tensor / w_tensor.mean()


def extract_stem_features(
    stem: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    stem.eval()
    feats_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            f = stem(images)
            feats_list.append(f.cpu())
            labels_list.append(labels.cpu())
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


def evaluate_head(
    top_blocks: nn.Module,
    avgpool: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    top_blocks.eval()
    classifier.eval()
    running_loss = 0.0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            x = top_blocks(feats)
            x = avgpool(x)
            x = torch.flatten(x, 1)
            logits = classifier(x)
            loss = criterion(logits, labels)
            running_loss += float(loss.item()) * feats.size(0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    preds_np = np.argmax(logits_np, axis=1)

    avg_loss = running_loss / max(len(labels_np), 1)
    acc = float(accuracy_score(labels_np, preds_np))
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1, logits_np, labels_np


def plot_training_curves(history: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_macro_f1"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
    ax1.plot(epochs, train_loss, marker="o", label="Train Loss", color="#059669", linewidth=2)
    ax1.plot(epochs, val_loss, marker="s", label="Val Loss", color="#2563EB", linewidth=2)
    ax1.set_title(f"{title} — Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(epochs, train_f1, marker="o", label="Train Macro-F1", color="#059669", linewidth=2)
    ax2.plot(epochs, val_f1, marker="s", label="Val Macro-F1", color="#D97706", linewidth=2)
    ax2.set_title(f"{title} — Macro-F1 Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train waste classification model.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_path = BASE_DIR / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment_name"]
    backbone = cfg["backbone"]
    class_names: List[str] = cfg["class_names"]
    num_classes = len(class_names)
    seed = int(cfg.get("seed", 42))
    set_reproducible_seed(seed, bool(cfg.get("deterministic", True)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_dir = BASE_DIR / cfg["data"]["manifest_dir"]
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"

    manifest_sha256 = hashlib.sha256(train_csv.read_bytes() + val_csv.read_bytes()).hexdigest()

    image_size = int(cfg["data"].get("image_size", 224))
    batch_size = int(cfg["data"].get("batch_size", 32))
    max_per_class_train = cfg["data"].get("max_samples_per_class_train")
    max_per_class_val = cfg["data"].get("max_samples_per_class_val")

    train_ds = ManifestWasteDataset(
        train_csv,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=True, augment_mode="canonical"),
        max_per_class=max_per_class_train,
        seed=seed,
        oversample_hard_classes=True,
    )
    train_ds_defense = ManifestWasteDataset(
        train_csv,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=True, augment_mode="shortcut_defense"),
        max_per_class=max(60, int((max_per_class_train or 200) * 0.45)),
        seed=seed + 17,
        oversample_hard_classes=True,
    )
    from model_factory import SameClassMosaicDataset
    train_ds_mosaic = SameClassMosaicDataset(
        train_csv,
        class_names=class_names,
        samples_per_class=55,
        image_size=image_size,
        seed=seed + 59,
    )
    val_ds = ManifestWasteDataset(
        val_csv,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=False),
        max_per_class=max_per_class_val,
        seed=seed,
    )

    raw_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_defense_loader = DataLoader(train_ds_defense, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_mosaic_loader = DataLoader(train_ds_mosaic, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(
        f"[train] Experiment={exp_name} | Backbone={backbone} | Device={device} | "
        f"Train={len(train_ds)}+{len(train_ds_defense)}+{len(train_ds_mosaic)} | Val={len(val_ds)}",
        flush=True,
    )

    model = build_classifier(backbone, num_classes=num_classes, pretrained=bool(cfg.get("pretrained", True)))
    hub_cache = Path.home() / ".cache" / "torch"
    if hub_cache.exists():
        shutil.rmtree(hub_cache, ignore_errors=True)
    model.to(device)

    unfreeze_n = int(cfg["training"].get("unfreeze_last_n_blocks", 2))
    all_blocks = list(model.features.children())
    split_idx = max(1, len(all_blocks) - unfreeze_n)
    frozen_stem = nn.Sequential(*all_blocks[:split_idx]).to(device)
    top_blocks = nn.Sequential(*all_blocks[split_idx:]).to(device)

    for p in frozen_stem.parameters():
        p.requires_grad = False
    frozen_stem.eval()

    t_start = time.perf_counter()
    print(f"[train] Extracting frozen stem features (blocks 0..{split_idx - 1})...", flush=True)
    t_f1, t_l1 = extract_stem_features(frozen_stem, raw_train_loader, device)
    t_f2, t_l2 = extract_stem_features(frozen_stem, raw_defense_loader, device)
    t_f3, t_l3 = extract_stem_features(frozen_stem, raw_mosaic_loader, device)
    train_feats = torch.cat([t_f1, t_f2, t_f3], dim=0)
    train_labels = torch.cat([t_l1, t_l2, t_l3], dim=0)
    del t_f1, t_f2, t_f3, t_l1, t_l2, t_l3
    val_feats, val_labels_tensor = extract_stem_features(frozen_stem, raw_val_loader, device)
    del frozen_stem
    gc.collect()

    feat_train_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    feat_val_loader = DataLoader(
        TensorDataset(val_feats, val_labels_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    class_weights = compute_class_weights(train_ds, num_classes).to(device)
    # Hard-pair emphasis on metal (idx 1) and plastic (idx 4)
    for hard_cls in ("metal", "plastic"):
        if hard_cls in class_names:
            class_weights[class_names.index(hard_cls)] *= 1.12
    class_weights = class_weights / class_weights.mean()
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.015))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    warmup_epochs = int(cfg["training"].get("warmup_epochs", 3))
    finetune_epochs = int(cfg["training"].get("finetune_epochs", 6))
    total_epochs = warmup_epochs + finetune_epochs
    patience = int(cfg["training"].get("early_stopping_patience", 4))

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state_dict = None
    best_val_logits = None
    best_val_labels = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, total_epochs + 1):
        ep_t0 = time.perf_counter()
        if epoch == 1:
            print(f"[train] Stage 1: Frozen backbone head warm-up ({warmup_epochs} epochs)", flush=True)
            for p in top_blocks.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=float(cfg["training"]["learning_rate_head"]),
                weight_decay=float(cfg["training"]["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(warmup_epochs, 1))
        elif epoch == warmup_epochs + 1:
            print(f"[train] Stage 2: Unfreezing top-{unfreeze_n} feature blocks ({finetune_epochs} epochs)", flush=True)
            for p in top_blocks.parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True
            trainable_params = list(top_blocks.parameters()) + list(model.classifier.parameters())
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=float(cfg["training"]["learning_rate_finetune"]),
                weight_decay=float(cfg["training"]["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(finetune_epochs, 1))

        if epoch <= warmup_epochs:
            top_blocks.eval()
        else:
            top_blocks.train()
            # Keep pretrained BatchNorm2d running statistics frozen during small-batch fine-tuning
            for m in top_blocks.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        model.classifier.train()

        running_loss = 0.0
        train_preds: List[int] = []
        train_targets: List[int] = []

        for feats, labels in feat_train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            if epoch > warmup_epochs:
                noise = torch.randn_like(feats) * 0.015
                feats = feats + noise

            optimizer.zero_grad()
            x = top_blocks(feats)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            logits = model.classifier(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * feats.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_targets.extend(labels.cpu().tolist())

        scheduler.step()
        gc.collect()

        train_loss = running_loss / max(len(train_feats), 1)
        train_acc = float(accuracy_score(train_targets, train_preds))
        train_f1 = float(f1_score(train_targets, train_preds, average="macro", zero_division=0))

        val_loss, val_acc, val_f1, val_logits, val_labels = evaluate_head(
            top_blocks, model.avgpool, model.classifier, feat_val_loader, criterion, device
        )
        ep_sec = round(time.perf_counter() - ep_t0, 2)

        stage_name = "warmup" if epoch <= warmup_epochs else "finetune"
        rec = {
            "epoch": epoch,
            "stage": stage_name,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_macro_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "epoch_seconds": ep_sec,
        }
        history.append(rec)
        print(
            f"  Epoch {epoch}/{total_epochs} [{stage_name}] | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({ep_sec}s)",
            flush=True,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_logits = val_logits
            best_val_labels = val_labels
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epoch > warmup_epochs and epochs_no_improve >= patience:
                print(f"[train] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    wall_clock_sec = round(time.perf_counter() - t_start, 2)

    temperature = 1.0
    val_ece_before = 0.0
    val_ece_after = 0.0
    if best_val_logits is not None and best_val_labels is not None:
        temperature, val_ece_before, val_ece_after = fit_temperature_scaling(best_val_logits, best_val_labels)
        print(
            f"[calibration] Validation ECE before={val_ece_before:.4f} -> after={val_ece_after:.4f} "
            f"(optimal temperature T={temperature:.4f})",
            flush=True,
        )

    ckpt_dir = BASE_DIR / cfg["output"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["output"]["checkpoint_name"]

    logs_dir = BASE_DIR / cfg["output"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = BASE_DIR / cfg["output"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_name": exp_name,
        "backbone": backbone,
        "class_names": class_names,
        "image_size": image_size,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "temperature": temperature,
        "val_ece_before": val_ece_before,
        "val_ece_after": val_ece_after,
        "wall_clock_seconds": wall_clock_sec,
        "device": str(device),
        "seed": seed,
        "git_commit": get_git_commit(),
        "manifest_sha256": manifest_sha256,
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
        "is_smoke_test": False,
        "config_snapshot": cfg,
        "history": history,
    }

    torch.save(
        {
            "state_dict": best_state_dict,
            "metadata": metadata,
        },
        ckpt_path,
    )

    log_path = logs_dir / f"{exp_name}_training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    curves_path = reports_dir / f"{exp_name}_curves.png"
    plot_training_curves(history, curves_path, title=exp_name)

    print(
        f"[done] Saved best checkpoint to {ckpt_path} (best_val_macro_f1={best_val_f1:.4f}, "
        f"wall_clock={wall_clock_sec}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
