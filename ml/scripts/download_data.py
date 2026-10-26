"""Download RealWaste (UCI) and TrashNet (GitHub) waste datasets with memory-efficient streaming resize."""
import argparse
import io
from pathlib import Path
import time
import zipfile
from PIL import Image
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "ml" / "data" / "raw"

DATASETS = {
    "trashnet": {
        "url": "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
        "zip_name": "trashnet.zip",
        "extract_subpath": "trashnet",
        "max_per_raw_folder": 1000,
    },
    "realwaste": {
        "url": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
        "zip_name": "realwaste.zip",
        "extract_subpath": "realwaste",
        "max_per_raw_folder": 1000,
    },
}


def download_with_retries(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"[download] Fetching {url} (attempt {attempt}/{retries})...", flush=True)
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            print(f"[download] Saved {dest} ({dest.stat().st_size / (1024 * 1024):.2f} MB)", flush=True)
            return
        except Exception as exc:
            print(f"[download] Attempt {attempt} failed: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def extract_and_compact_archive(zip_path: Path, target_dir: Path, max_per_folder: int = 450) -> None:
    """Extract images resized to 256x256 JPEG (quality=85) to keep tmpfs RAM footprint < 70 MB."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] Extracting and compacting {zip_path.name} -> {target_dir}...", flush=True)
    folder_counts = {}
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in sorted(zf.namelist()):
            if member.endswith("/") or "__MACOSX" in member:
                continue
            p = Path(member)
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            raw_cls = p.parent.name
            if raw_cls in {"Miscellaneous Trash", "Textile Trash", "trash"}:
                continue
            cnt = folder_counts.get(raw_cls, 0)
            if cnt >= max_per_folder:
                continue

            out_file = target_dir / raw_cls / p.name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw_bytes = zf.read(member)
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    rgb = im.convert("RGB")
                    if max(rgb.size) > 256:
                        rgb.thumbnail((256, 256), Image.Resampling.BILINEAR)
                    rgb.save(out_file, format="JPEG", quality=85)
                folder_counts[raw_cls] = cnt + 1
                extracted += 1
            except Exception:
                continue

    print(f"[extract] Completed {zip_path.name}: {extracted} images across {folder_counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RealWaste and TrashNet datasets.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for key, meta in DATASETS.items():
        extract_dir = RAW_DIR / meta["extract_subpath"]
        zip_path = RAW_DIR / meta["zip_name"]

        existing_images = list(extract_dir.rglob("*.jpg")) + list(extract_dir.rglob("*.JPG")) + list(extract_dir.rglob("*.png"))
        if existing_images and not args.force:
            print(f"[skip] {key} already downloaded ({len(existing_images)} images found).", flush=True)
            continue

        if not zip_path.exists() or args.force:
            download_with_retries(meta["url"], zip_path)

        extract_and_compact_archive(zip_path, extract_dir, max_per_folder=meta["max_per_raw_folder"])
        if zip_path.exists():
            zip_path.unlink()

    print("[done] All dataset sources downloaded and compacted under ml/data/raw/", flush=True)


if __name__ == "__main__":
    main()
