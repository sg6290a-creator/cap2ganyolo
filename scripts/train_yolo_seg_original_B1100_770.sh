#!/usr/bin/env bash
set -euo pipefail

yolo segment train \
  data="$(pwd)/data/yolo_crack_seg_from_original_B1100/data.yaml" \
  model="$(pwd)/yolo26n-seg.pt" \
  epochs=120 \
  imgsz=770 \
  batch=16 \
  project="$(pwd)/runs/segment" \
  name=crack_original_B1100_yolo26seg_770
