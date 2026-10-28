#!/bin/bash
# Python evaluation (256x256, PSNR / SSIM / RMSE in LAB). The paper numbers use evaluation/measure_shadow.m.
# Usage: bash scripts/eval.sh [result root]
OUT=${1:-results}

echo "== AISTD (Table 1)"
python evaluation/eval_shadow.py --result_dir "$OUT/AISTD" --gt_dir data/AISTD/test/test_C_fixed_official --mask_dir data/AISTD/test/test_B
echo "== SRD (Table 2)"
python evaluation/eval_shadow.py --result_dir "$OUT/SRD"  --gt_dir data/SRD/test/shadow_free
echo "== LRSS (Table 2)"
python evaluation/eval_shadow.py --result_dir "$OUT/LRSS" --gt_dir data/LRSS/testB
echo "== UIUC (Table 2)"
python evaluation/eval_shadow.py --result_dir "$OUT/UIUC" --gt_dir data/UIUC/test_C
