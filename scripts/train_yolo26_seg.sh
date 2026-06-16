#!/usr/bin/env bash
set -euo pipefail

YOLO_BIN="${YOLO_BIN:-/home/lim-hyun-su/miniconda3/envs/yolo26/bin/yolo}"
DATA_YAML="${DATA_YAML:-data/yolo_defect_seg_square/data.yaml}"
MODEL="${MODEL:-yolo26n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-4}"
PROJECT="${PROJECT:-runs/segment}"
NAME="${NAME:-defect_seg_square_yolo26n}"

"${YOLO_BIN}" segment train   data="${DATA_YAML}"   model="${MODEL}"   epochs="${EPOCHS}"   imgsz="${IMGSZ}"   batch="${BATCH}"   project="${PROJECT}"   name="${NAME}"
