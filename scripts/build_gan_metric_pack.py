"""Build a fixed review/metric pack for comparing GAN checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def copy_subset(paths: list[Path], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, src in enumerate(paths, 1):
        dst = out_dir / f"{prefix}_{idx:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        names.append(dst.name)
    return names


def contact_sheet(paths: list[Path], out: Path, cols: int = 8, thumb: int = 128) -> None:
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize((thumb, thumb), Image.Resampling.BICUBIC)
            imgs.append(im.copy())
    if not imgs:
        return
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/crack_only_gray_A300_B600"))
    parser.add_argument("--out-root", type=Path, default=Path("data/gan_metric_pack_crack_gray"))
    parser.add_argument("--a-count", type=int, default=30)
    parser.add_argument("--b-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    a_paths = iter_images(args.dataset_root / "train/A")
    b_paths = iter_images(args.dataset_root / "train/B")
    mask_dir = args.dataset_root / "train/mask"
    b_pairs = [(p, mask_dir / f"{p.stem}.png") for p in b_paths if (mask_dir / f"{p.stem}.png").exists()]

    rng.shuffle(a_paths)
    rng.shuffle(b_pairs)
    selected_a = sorted(a_paths[: args.a_count])
    selected_b = sorted(b_pairs[: args.b_count], key=lambda x: x[0].name)

    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    fixed_a_names = copy_subset(selected_a, args.out_root / "fixed_A", "A")
    real_b_names = []
    real_mask_names = []
    (args.out_root / "real_B").mkdir(parents=True, exist_ok=True)
    (args.out_root / "real_mask").mkdir(parents=True, exist_ok=True)
    for idx, (b_path, mask_path) in enumerate(selected_b, 1):
        b_dst = args.out_root / "real_B" / f"B_{idx:04d}{b_path.suffix.lower()}"
        m_dst = args.out_root / "real_mask" / f"B_{idx:04d}.png"
        shutil.copy2(b_path, b_dst)
        shutil.copy2(mask_path, m_dst)
        real_b_names.append(b_dst.name)
        real_mask_names.append(m_dst.name)

    contact_sheet(sorted((args.out_root / "fixed_A").iterdir()), args.out_root / "review" / "fixed_A_sheet.jpg")
    contact_sheet(sorted((args.out_root / "real_B").iterdir())[:80], args.out_root / "review" / "real_B_sheet.jpg")
    contact_sheet(sorted((args.out_root / "real_mask").iterdir())[:80], args.out_root / "review" / "real_mask_sheet.jpg")

    config = {
        "source_dataset": str(args.dataset_root),
        "fixed_A_count": len(fixed_a_names),
        "real_B_count": len(real_b_names),
        "real_mask_count": len(real_mask_names),
        "usage": "Generate checkpoints on fixed_A, then compare fake_B/fake_mask visually and by mask area/fragmentation against real_B/real_mask.",
    }
    (args.out_root / "metrics_pack.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OUT={args.out_root}")
    print(f"FIXED_A={len(fixed_a_names)}")
    print(f"REAL_B={len(real_b_names)}")
    print(f"REAL_MASK={len(real_mask_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
