#!/usr/bin/env bash
# Download public T-REN checkpoints into logs/tren-ckpts/.

set -euo pipefail

DEST="${1:-logs/tren-ckpts}"
HF_BASE="https://huggingface.co/savyak2/T-REN/resolve/main"

mkdir -p "${DEST}"

echo "Downloading T-REN region encoder to ${DEST}/ ..."
wget -nv -L -O "${DEST}/tren_region_encoder.pth" \
  "${HF_BASE}/tren_region_encoder.pth"

echo "Done. Note: DINOv3 ViT-L backbone + dinotxt head weights are not in this HF repo;"
echo "      place dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth and"
echo "      dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth in ${DEST}/"
echo "      (see README) before training or running eval that loads FeatureExtractor."
