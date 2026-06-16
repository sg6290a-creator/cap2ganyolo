#!/usr/bin/env bash
set -euo pipefail

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
