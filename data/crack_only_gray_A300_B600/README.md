# Balanced Grayscale Crack GAN Dataset

Source B pool: `data/crack_only_gan_optimized_1100`

This dataset is intentionally smaller and grayscale. A images are selected from
the previous clean metal set plus additional normal images whose grayscale
contrast is closer to the crack B-domain. B/mask pairs are reduced to the best
600 masks by crack-size and fragmentation heuristics.

Counts:

```text
A: 300
B/mask: 600
image_size_saved: 512
```

Training command:

```bash
cd /home/lim-hyun-su/defect_synthesis/defect_synthesis_lite/defect_transfer_package

/home/lim-hyun-su/miniconda3/envs/yolo26/bin/python -u train.py --cuda \
  --dataroot ./data \
  --dataset crack_only_gray_A300_B600 \
  --weightsf ./data/crack_only_gray_A300_B600_train_512/weights \
  --outf ./data/crack_only_gray_A300_B600_train_512/outputs \
  --epochs 40 \
  --batch_size 1 \
  --image_size 512 \
  --decay_epochs 12 \
  --save_freq 5 \
  --save_freq_images 100 \
  --lr 0.0002
```
