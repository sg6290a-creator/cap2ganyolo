#!/usr/bin/env bash
set -euo pipefail

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
