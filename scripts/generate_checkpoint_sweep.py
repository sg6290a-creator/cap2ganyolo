"""Generate visual samples for every selected GAN checkpoint."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cycle_gan_mask import CycleGAN_Mask

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def epoch_from_path(path: Path) -> str:
    m = re.search(r"epoch_(\d+)", path.stem)
    return m.group(1) if m else path.stem


def load_image(path: Path, size: int) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im = im.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor, im


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = ((arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_to_image(mask: torch.Tensor) -> Image.Image:
    arr = mask.detach().cpu().numpy()
    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def diff_image(a: Image.Image, b: Image.Image) -> Image.Image:
    aa = np.asarray(a).astype(np.int16)
    bb = np.asarray(b).astype(np.int16)
    return Image.fromarray(np.abs(aa - bb).astype(np.uint8))


def preview(a: Image.Image, fake: Image.Image, mask: Image.Image, diff: Image.Image, thumb: int) -> Image.Image:
    items = [
        ("A", a.resize((thumb, thumb), Image.Resampling.BICUBIC)),
        ("G(A)", fake.resize((thumb, thumb), Image.Resampling.BICUBIC)),
        ("Mask", mask.convert("RGB").resize((thumb, thumb), Image.Resampling.NEAREST)),
        ("Diff", diff.resize((thumb, thumb), Image.Resampling.BICUBIC)),
    ]
    title_h = 22
    gap = 6
    canvas = Image.new("RGB", (thumb * 4 + gap * 3, thumb + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, img) in enumerate(items):
        x = i * (thumb + gap)
        draw.text((x + 2, 4), title, fill=(0, 0, 0))
        canvas.paste(img, (x, title_h))
    return canvas


def contact_sheet(images: list[Image.Image], cols: int, out: Path) -> None:
    if not images:
        return
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(images):
        sheet.paste(img, ((i % cols) * w, (i // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def make_model_args(device: str, image_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=device == "cuda",
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        netG="ResNet9",
        netD="basic",
        n_layers_D=3,
        init_type="normal",
        init_gain=0.02,
        image_size=image_size,
        lr=0.0002,
        beta_1=0.5,
        beta_2=0.999,
        epochs=1,
        decay_epochs=1,
        dataset="generate",
        weightsf=".",
        outf=".",
        thrs_mask=0.2,
        lambda_identity_A=5.0,
        lambda_identity_B=5.0,
        lambda_GAN_A2B=1.0,
        lambda_GAN_B2A=1.0,
        lambda_cycle_ABA=10.0,
        lambda_cycle_BAB=10.0,
        lambda_GAN_fit=150.0,
        lambda_background=0.1,
        netG_A2B="",
        netG_B2A="",
        netD_A="",
        netD_B="",
        netD_fit="",
        netD_mask="",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/crack_only_square/train/A"))
    parser.add_argument("--checkpoint-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("generated/crack_checkpoint_sweep_100"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--thumb-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    model = CycleGAN_Mask(make_model_args(device, args.image_size))
    model.create_network()
    model.netG_A2B.eval()

    input_paths = list_images(args.input_dir)[: args.count]
    loaded = [load_image(path, args.image_size) for path in input_paths]

    checkpoints: list[tuple[str, Path]] = []
    for ckpt_dir in args.checkpoint_dir:
        run_name = ckpt_dir.parent.parent.name
        for ckpt in sorted(ckpt_dir.glob("netG_A2B_epoch_*.pth"), key=lambda p: int(epoch_from_path(p))):
            checkpoints.append((run_name, ckpt))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "index.csv").open("w") as f:
        f.write("run,epoch,checkpoint,out_dir\n")

    with torch.no_grad():
        for run_name, ckpt in checkpoints:
            epoch = epoch_from_path(ckpt)
            out_root = args.out_dir / run_name / f"epoch_{epoch}"
            fake_dir = out_root / "fake_B"
            mask_dir = out_root / "fake_mask"
            diff_dir = out_root / "diff"
            prev_dir = out_root / "preview"
            for p in [fake_dir, mask_dir, diff_dir, prev_dir]:
                p.mkdir(parents=True, exist_ok=True)

            state = torch.load(ckpt, map_location=model.device)
            model.netG_A2B.load_state_dict(state)
            previews: list[Image.Image] = []

            for start in range(0, len(loaded), args.batch_size):
                batch_items = loaded[start : start + args.batch_size]
                tensors = torch.stack([item[0] for item in batch_items]).to(model.device)
                fake_batch, mask_batch, _ = model.netG_A2B(tensors)
                for j, (tensor_item, a_img) in enumerate(batch_items):
                    idx = start + j + 1
                    src_path = input_paths[start + j]
                    fake_img = tensor_to_image(fake_batch[j])
                    mask_img = mask_to_image(mask_batch[j])
                    diff_img = diff_image(a_img, fake_img)
                    stem = f"{idx:03d}_{src_path.stem}"
                    fake_img.save(fake_dir / f"{stem}.png")
                    mask_img.save(mask_dir / f"{stem}.png")
                    diff_img.save(diff_dir / f"{stem}.png")
                    prev = preview(a_img, fake_img, mask_img, diff_img, args.thumb_size)
                    prev.save(prev_dir / f"{stem}.png")
                    previews.append(prev)

            contact_sheet(previews, cols=5, out=out_root / f"contact_sheet_{run_name}_epoch_{epoch}.jpg")
            with (args.out_dir / "index.csv").open("a") as f:
                f.write(f"{run_name},{epoch},{ckpt},{out_root}\n")
            print(f"[done] {run_name} epoch {epoch}: {len(loaded)} samples -> {out_root}")

    print(f"OUT={args.out_dir}")
    print(f"INDEX={args.out_dir / 'index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
