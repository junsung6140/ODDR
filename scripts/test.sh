#!/bin/bash
# Run ODDR on the four test sets of Table 1 and Table 2.
# Usage: bash scripts/test.sh [checkpoint] [output root]
CKPT=${1:-checkpoints/ODDR.pkl}
OUT=${2:-results}
SD=checkpoints/stable-diffusion-2-1-base

python inference.py -i data/AISTD/test/test_A -o "$OUT/AISTD" --ckpt_path "$CKPT" --pretrained_model_name_or_path "$SD"
python inference.py -i data/SRD/test/shadow   -o "$OUT/SRD"   --ckpt_path "$CKPT" --pretrained_model_name_or_path "$SD"
python inference.py -i data/LRSS/testA        -o "$OUT/LRSS"  --ckpt_path "$CKPT" --pretrained_model_name_or_path "$SD"
python inference.py -i data/UIUC/test_A       -o "$OUT/UIUC"  --ckpt_path "$CKPT" --pretrained_model_name_or_path "$SD"
