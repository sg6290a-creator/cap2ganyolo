# cap2ganyolo

Minimal package for crack-only grayscale CycleGAN-mask training and YOLO segmentation training.

This package intentionally excludes UI, SAM preprocessing, multi-class data builders, and old run artifacts.

## Included

- `data/crack_only_gray_A300_B600`
  - GAN training dataset.
  - Current checked data count: A 70, B 600, mask 600.
  - Images are grayscale RGB and square.
- `data/yolo_crack_seg_from_gan_B600`
  - YOLO segmentation dataset exported from the same real B/mask pairs used for GAN.
  - Train 510, val 90.
  - Class: `crack`.
- `data/gan_metric_pack_crack_gray`
  - Fixed A/B/mask review set for comparing GAN checkpoints.
- `yolo26n-seg.pt`
  - Segmentation base weight.

## Environment

Use the existing `yolo26` conda env if available.

```bash
cd cap2ganyolo

conda activate yolo26
pip install -r requirements.txt
```

If your shell cannot find `conda activate`, use:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
```

## 1. Fast Large-Image GAN Training

Recommended for a 90GB VRAM GPU:

```bash
cd cap2ganyolo

python -u train.py --cuda \
  --dataroot ./data \
  --dataset crack_only_gray_A300_B600 \
  --weightsf ./runs/gan/crack_gray_768_fast/weights \
  --outf ./runs/gan/crack_gray_768_fast/outputs \
  --epochs 40 \
  --batch_size 4 \
  --image_size 768 \
  --decay_epochs 12 \
  --save_freq 5 \
  --save_freq_images 100 \
  --lr 0.0002
```

Higher-detail option:

```bash
cd cap2ganyolo

python -u train.py --cuda \
  --dataroot ./data \
  --dataset crack_only_gray_A300_B600 \
  --weightsf ./runs/gan/crack_gray_1024/weights \
  --outf ./runs/gan/crack_gray_1024/outputs \
  --epochs 35 \
  --batch_size 2 \
  --image_size 1024 \
  --decay_epochs 10 \
  --save_freq 5 \
  --save_freq_images 100 \
  --lr 0.0002
```

Start with the 768 command. Compare epoch 5, 10, 15, 20 before running longer.

## 2. Generate GAN Review Samples

Use the fixed metric A set so every checkpoint is compared on the same input images.

Example for epoch 15 from the 768 run:

```bash
cd cap2ganyolo

python scripts/generate_gan_samples.py \
  --input-dir data/gan_metric_pack_crack_gray/fixed_A \
  --weight runs/gan/crack_gray_768_fast/weights/crack_only_gray_A300_B600/netG_A2B_epoch_15.pth \
  --out-dir generated/gan_review/crack_gray_768_epoch15 \
  --count 30 \
  --image-size 768 \
  --device cuda
```

Outputs:

- `fake_B`
- `fake_mask`
- `diff`
- `preview`

If `fake_mask` is almost all black or `diff` changes the full image background, stop that run and use an earlier checkpoint or reduce the B-domain further.

## 3. YOLO Segmentation Training

This trains YOLO segmentation from the real B/mask pairs used for GAN training, not from GAN-generated images.

Recommended:

```bash
cd cap2ganyolo

yolo segment train \
  data=$(pwd)/data/yolo_crack_seg_from_gan_B600/data.yaml \
  model=$(pwd)/yolo26n-seg.pt \
  epochs=120 \
  imgsz=770 \
  batch=16 \
  project=$(pwd)/runs/segment \
  name=crack_real_B600_yolo26seg_770
```

Higher resolution:

```bash
cd cap2ganyolo

yolo segment train \
  data=$(pwd)/data/yolo_crack_seg_from_gan_B600/data.yaml \
  model=$(pwd)/yolo26n-seg.pt \
  epochs=120 \
  imgsz=1036 \
  batch=8 \
  project=$(pwd)/runs/segment \
  name=crack_real_B600_yolo26seg_1036
```

YOLO26 may adjust image size to a multiple of stride 14. That is why the commands use `770` and `1036`.

## 4. Validate YOLO

```bash
cd cap2ganyolo

yolo segment val \
  data=$(pwd)/data/yolo_crack_seg_from_gan_B600/data.yaml \
  model=$(pwd)/runs/segment/crack_real_B600_yolo26seg_770/weights/best.pt \
  imgsz=770
```

## 5. Inference

```bash
cd cap2ganyolo

yolo segment predict \
  model=$(pwd)/runs/segment/crack_real_B600_yolo26seg_770/weights/best.pt \
  source=$(pwd)/data/yolo_crack_seg_from_gan_B600/images/val \
  imgsz=770 \
  save=True \
  save_txt=True \
  project=$(pwd)/runs/predict \
  name=crack_val_preview
```

## 6. Regenerate YOLO Dataset From GAN Dataset

Only needed if you edit `data/crack_only_gray_A300_B600/train/B` or `train/mask`.

```bash
cd cap2ganyolo

python scripts/export_yolo_seg_from_gan_dataset.py \
  --dataset-root data/crack_only_gray_A300_B600 \
  --out-root data/yolo_crack_seg_from_gan_B600 \
  --val-ratio 0.15 \
  --min-area 20 \
  --epsilon 1.2
```

## Practical Notes

- GAN and YOLO are separate here.
- GAN is for generating/reviewing synthetic crack appearance.
- YOLO training uses real B/mask labels from the GAN training dataset.
- Do not train YOLO on GAN outputs unless the GAN masks are visually verified first.
- For this dataset, very long GAN training can make background texture worse. Prefer checkpoint review every 5 epochs.
