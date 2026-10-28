"""Python port of evaluation/measure_shadow.m (no MATLAB needed).

PSNR, SSIM and RMSE (mean absolute error in LAB, the convention of prior shadow removal work)
for the shadow region, the non-shadow region and the whole image, at 256x256.
"RMSE" follows the PP-Lab line of the MATLAB script: all = per-image mean,
shadow / non-shadow = per-pixel mean over the whole test set.

The numbers in the paper were computed with the MATLAB script. Resizing, LAB conversion and
SSIM are implemented slightly differently in Python, so expect small differences (about 0.05 dB).

Example:
    python evaluation/eval_shadow.py --result_dir results/AISTD \
        --gt_dir data/AISTD/test/test_C_fixed_official --mask_dir data/AISTD/test/test_B
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity
from skimage.transform import resize


def load(path, size=256, gray=False):
    img = Image.open(path).convert('L' if gray else 'RGB')
    img = np.asarray(img).astype(np.float64) / 255.0
    # MATLAB imresize: bicubic with antialiasing
    return resize(img, (size, size), order=3, mode='symmetric', anti_aliasing=True)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def ssim(a, b):
    # MATLAB ssim: gaussian window (sigma 1.5), applied on the full array
    return structural_similarity(a, b, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)


def list_png(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.png')))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--mask_dir', default=None,
                        help='shadow masks; without masks only the whole-image metrics are reported (Table 2)')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    res, gts = list_png(args.result_dir), list_png(args.gt_dir)
    masks = list_png(args.mask_dir) if args.mask_dir else [None] * len(res)
    assert len(res) == len(gts) == len(masks), f"{len(res)} results, {len(gts)} GT, {len(masks)} masks"

    p_all, p_s, p_n, s_all, s_s, s_n, mae_all = [], [], [], [], [], [], []
    tot_s, pix_s, tot_n, pix_n = 0.0, 0, 0.0, 0
    for r, g, m in zip(res, gts, masks):
        s = load(r, args.size)
        f = load(g, args.size)
        mask = load(m, args.size, gray=True) if m is not None else np.ones((args.size, args.size))
        smask = (mask > 0.5 / 255)[..., None].astype(np.float64)  # MATLAB: any non-zero uint8 value
        nmask = 1.0 - smask

        p_all.append(psnr(s, f))
        p_s.append(psnr(s * smask, f * smask))
        p_n.append(psnr(s * nmask, f * nmask))
        s_all.append(ssim(s, f))
        s_s.append(ssim(s * smask, f * smask))
        s_n.append(ssim(s * nmask, f * nmask))

        dist = np.abs(rgb2lab(np.clip(f, 0, 1)) - rgb2lab(np.clip(s, 0, 1)))
        mae_all.append(dist.sum() / (args.size * args.size))
        tot_s += (dist * smask).sum()
        pix_s += smask.sum()
        tot_n += (dist * nmask).sum()
        pix_n += nmask.sum()

    print(f"{len(res)} images")
    if args.mask_dir is None:
        print(f"All image: PSNR {np.mean(p_all):.2f}  SSIM {np.mean(s_all):.3f}  RMSE {np.mean(mae_all):.2f}")
        return
    print(f"           Shadow   Non-shadow   All")
    print(f"PSNR   {np.mean(p_s):10.2f} {np.mean(p_n):10.2f} {np.mean(p_all):10.2f}")
    print(f"SSIM   {np.mean(s_s):10.3f} {np.mean(s_n):10.3f} {np.mean(s_all):10.3f}")
    print(f"RMSE   {tot_s / pix_s:10.2f} {tot_n / pix_n:10.2f} {np.mean(mae_all):10.2f}")


if __name__ == "__main__":
    main()
