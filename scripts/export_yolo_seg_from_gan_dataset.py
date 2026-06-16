"""Export a crack-only GAN B/mask dataset to YOLO segmentation format."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def mask_to_polygons(mask_path: Path, width: int, height: int, min_area: int, epsilon: float) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        coords: list[float] = []
        for x, y in approx:
            coords.extend([
                max(0.0, min(1.0, float(x) / width)),
                max(0.0, min(1.0, float(y) / height)),
            ])
        polygons.append(coords)
    return polygons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/crack_only_gray_A300_B600"))
    parser.add_argument("--out-root", type=Path, default=Path("data/yolo_crack_seg_from_gan_B600"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1.2)
    args = parser.parse_args()

    image_dir = args.dataset_root / "train/B"
    mask_dir = args.dataset_root / "train/mask"
    pairs = []
    for image_path in iter_images(image_dir):
        candidates = [
            mask_dir / f"{image_path.stem}.png",
            mask_dir / f"{image_path.stem}.jpg",
            mask_dir / image_path.name,
        ]
        mask_path = next((p for p in candidates if p.exists()), None)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = int(round(len(pairs) * args.val_ratio))
    val_indices = set(range(val_count))

    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    for split in ["train", "val"]:
        (args.out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for idx, (image_path, mask_path) in enumerate(pairs):
        with Image.open(image_path) as im:
            width, height = im.size
        polygons = mask_to_polygons(mask_path, width, height, args.min_area, args.epsilon)
        if not polygons:
            skipped += 1
            continue
        split = "val" if idx in val_indices else "train"
        out_img = args.out_root / "images" / split / image_path.name
        out_lbl = args.out_root / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_img)
        lines = ["0 " + " ".join(f"{v:.6f}" for v in polygon) for polygon in polygons]
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported += 1

    yaml_text = "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: crack\n"
    (args.out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"pairs={len(pairs)}")
    print(f"exported={exported}")
    print(f"skipped={skipped}")
    print(f"YOLO_ROOT={args.out_root}")
    print(f"DATA_YAML={args.out_root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
