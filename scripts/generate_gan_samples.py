"""Generate A->B defect samples from a trained CycleGAN mask generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
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


def load_image(path: Path, size: int) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im = im.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0), im


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = ((arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_to_image(mask: torch.Tensor) -> Image.Image:
    arr = mask.detach().cpu().squeeze().numpy()
    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def diff_image(a: Image.Image, b: Image.Image) -> Image.Image:
    aa = np.asarray(a).astype(np.int16)
    bb = np.asarray(b).astype(np.int16)
    diff = np.abs(aa - bb).astype(np.uint8)
    return Image.fromarray(diff)


def preview(a: Image.Image, fake: Image.Image, mask: Image.Image, diff: Image.Image) -> Image.Image:
    w, h = a.size
    title_h = 34
    gap = 12
    canvas = Image.new("RGB", (w * 4 + gap * 3, h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    items = [("A", a), ("G(A)", fake), ("Mask", mask.convert("RGB")), ("|A-G(A)|", diff)]
    for i, (title, img) in enumerate(items):
        x = i * (w + gap)
        draw.text((x + 4, 8), title, fill=(0, 0, 0))
        canvas.paste(img, (x, title_h))
    return canvas


def make_args(device: str, image_size: int, weight: Path) -> SimpleNamespace:
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
        netG_A2B=str(weight),
        netG_B2A="",
        netD_A="",
        netD_B="",
        netD_fit="",
        netD_mask="",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/combined_defects_square/train/A"))
    parser.add_argument("--weight", type=Path, default=Path("weights/combined_defects_epoch30/netG_A2B_epoch_30.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/gan_epoch30_384_30"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    model_args = make_args(device, args.image_size, args.weight)
    model = CycleGAN_Mask(model_args)
    model.create_network()
    state = torch.load(args.weight, map_location=model.device)
    model.netG_A2B.load_state_dict(state)
    model.netG_A2B.eval()

    fake_dir = args.out_dir / "fake_B"
    mask_dir = args.out_dir / "fake_mask"
    diff_dir = args.out_dir / "diff"
    preview_dir = args.out_dir / "preview"
    for path in [fake_dir, mask_dir, diff_dir, preview_dir]:
        path.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input_dir)[: args.count]
    with torch.no_grad():
        for idx, image_path in enumerate(images, 1):
            tensor, a_img = load_image(image_path, args.image_size)
            tensor = tensor.to(model.device)
            fake_tensor, mask_tensor, _ = model.netG_A2B(tensor)
            fake_img = tensor_to_image(fake_tensor)
            mask_img = mask_to_image(mask_tensor)
            diff_img = diff_image(a_img, fake_img)
            stem = f"{idx:03d}_{image_path.stem}"
            fake_img.save(fake_dir / f"{stem}.png")
            mask_img.save(mask_dir / f"{stem}.png")
            diff_img.save(diff_dir / f"{stem}.png")
            preview(a_img, fake_img, mask_img, diff_img).save(preview_dir / f"{stem}.png")
            print(f"[{idx}/{len(images)}] {image_path.name}")

    print(f"OUT={args.out_dir}")
    print(f"FAKE_DIR={fake_dir}")
    print(f"MASK_DIR={mask_dir}")
    print(f"PREVIEW_DIR={preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
