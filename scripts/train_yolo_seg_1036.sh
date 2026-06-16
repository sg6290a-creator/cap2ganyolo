#!/usr/bin/env bash
set -euo pipefail

yolo segment train \
  data="$(pwd)/data/yolo_crack_seg_from_gan_B600/data.yaml" \
  model="$(pwd)/yolo26n-seg.pt" \
  epochs=120 \
  imgsz=1036 \
  batch=8 \
  project="$(pwd)/runs/segment" \
  name=crack_real_B600_yolo26seg_1036
