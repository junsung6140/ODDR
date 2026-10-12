<div align="center">

# ODDR: One-Step Deshadow Diffusion via Reward Guidance

[Junseong Shin](https://junsung6140.github.io/)<sup>1</sup> · Kijun Kim<sup>2</sup> · Minseong Kim<sup>2</sup> · Dongjin Kim<sup>3</sup> · Tae Hyun Kim<sup>1,3†</sup>

<sup>1</sup>Department of Artificial Intelligence, Hanyang University · <sup>2</sup>AX Future Technology Institute, KT · <sup>3</sup>Department of Computer Science, Hanyang University

**NeurIPS 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://junsung6140.github.io/oddr_project_page/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.01291-b31b1b)](https://arxiv.org/abs/2610.01291)

</div>

<p align="center"><img src="assets/teaser.jpg" width="70%" alt="Comparison"></p>

**Overall training pipeline.** (a) The baseline ODD is trained on synthetic shadow images ($x_{\text{syn}}$), optimizing the ODD-LoRA and ODD-Conv modules. (b) ODDR is then fine-tuned from ODD parameters using real-world shadow images ($x_{\text{real}}$) guided by ShadowReward ($r_\theta$) model, optimizing ODDR-LoRA module.

## Abstract

Recent advances in deep learning for shadow removal have significantly enhanced image quality and realism. However, most approaches rely on real-world paired datasets, which are costly to collect and often limited in scene diversity, leading to limited generalization. To address these limitations, we propose One-step Deshadow Diffusion via Reward guidance (ODDR), a new framework that achieves efficient and high-fidelity shadow removal without relying on real-world paired supervision. Our method begins with One-step Deshadow Diffusion (ODD), a baseline model trained on synthetic shadow data for efficient one-step shadow-free reconstruction. We further adapt ODD into ODDR using ShadowReward. In contrast to traditional, annotation-heavy approaches, ShadowReward is the first reward model for shadow removal trained entirely without human annotation. It learns to mimic human perceptual judgments by ranking synthetically generated images with controlled degradations, such as texture distortion and boundary artifacts. This reward-guided fine-tuning enables ODDR to narrow the synthetic-to-real domain gap. Extensive experiments show that ODD achieves strong performance without relying on real-world paired supervision, and ODDR further improves the results, narrowing the gap to fully supervised methods trained on real-world paired data while maintaining higher computational efficiency as a single-step model.

## Method

![Pipeline](assets/method.jpg)

**Overall training pipeline.** (a) The baseline ODD is trained on synthetic shadow images ($x_{\text{syn}}$), optimizing the ODD-LoRA and ODD-Conv modules. (b) ODDR is then fine-tuned from ODD parameters using real-world shadow images ($x_{\text{real}}$) guided by ShadowReward ($r_\theta$) model, optimizing ODDR-LoRA module.

## Results

Quantitative comparison on AISTD dataset. **Bold** and <u>underline</u> denote the best and second-best results, evaluated among the methods trained without real-world pairs. ✓ denotes mask-free methods and ✗ indicates methods that require shadow masks. D indicates whether real-world paired data is used for training.

| D | Methods | Mask-free | Shadow PSNR↑ | Shadow SSIM↑ | Shadow RMSE↓ | Non-Shadow PSNR↑ | Non-Shadow SSIM↑ | Non-Shadow RMSE↓ | All PSNR↑ | All SSIM↑ | All RMSE↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  | Real-world Shadow Image | – | 20.83 | 0.927 | 39.01 | 37.46 | 0.985 | 2.40 | 20.46 | 0.894 | 8.40 |
| w/ real-world pair | DHAN | ✗ | 32.92 | 0.988 | 9.6 | 27.15 | 0.971 | 7.4 | 25.66 | 0.956 | 7.8 |
| w/ real-world pair | SP+M-Net | ✗ | 37.60 | 0.990 | 6.3 | 36.02 | 0.976 | 3.0 | 32.94 | 0.962 | 3.5 |
| w/ real-world pair | Fu et al. | ✗ | 36.04 | 0.978 | 6.7 | 31.16 | 0.892 | 3.8 | 29.45 | 0.861 | 4.2 |
| w/ real-world pair | BMNet | ✗ | 37.87 | 0.991 | 5.62 | 37.51 | 0.985 | 2.45 | 33.98 | 0.972 | 2.97 |
| w/ real-world pair | SG-ShadowNet | ✗ | 36.80 | 0.990 | 5.93 | 35.57 | 0.978 | 2.92 | 32.46 | 0.962 | 3.41 |
| w/ real-world pair | DeS3 | ✓ | 36.49 | 0.989 | 6.59 | 34.72 | 0.972 | 3.32 | 31.39 | 0.957 | 3.86 |
| w/ real-world pair | OmniSR | ✓ | 37.01 | 0.990 | 3.08 | 37.82 | 0.984 | 2.40 | 33.35 | 0.970 | 6.57 |
| w/ real-world pair | SSR | ✓ | 39.94 | 0.993 | 2.89 | 38.04 | 0.983 | 2.63 | 35.19 | 0.974 | 4.23 |
| w/o real-world pair | MaskShadow-GAN | ✗ | 32.19 | 0.984 | 10.8 | 33.44 | 0.974 | 3.8 | 28.81 | 0.946 | 4.8 |
| w/o real-world pair | LG-ShadowNet | ✗ | 32.45 | 0.982 | 9.65 | 33.73 | 0.975 | 3.25 | 29.22 | 0.947 | 4.30 |
| w/o real-world pair | DCS | ✓ | 32.00 | 0.976 | 10.41 | 33.56 | 0.968 | 3.63 | 28.77 | 0.932 | 4.74 |
| w/o real-world pair | Le et al. | ✗ | 33.09 | 0.983 | 9.67 | 35.35 | <u>0.978</u> | 2.82 | 30.15 | 0.951 | 3.94 |
| w/o real-world pair | G2R | ✗ | 33.58 | 0.979 | 8.82 | 35.52 | 0.976 | 2.89 | 30.52 | 0.944 | 3.86 |
| w/o real-world pair | BCDiff | ✗ | 35.91 | 0.986 | 7.6 | 37.27 | **0.984** | 2.4 | 32.73 | 0.962 | 3.3 |
| w/o real-world pair | S3R-Net | ✓ | 32.08 | 0.983 | 9.94 | 33.61 | <u>0.978</u> | 3.55 | 29.05 | 0.952 | 12.03 |
| w/o real-world pair | ODD (ours) | ✓ | <u>36.67</u> | <u>0.990</u> | <u>6.27</u> | **37.45** | **0.984** | **2.49** | <u>33.32</u> | <u>0.971</u> | <u>3.11</u> |
| w/o real-world pair | ODDR (ours) | ✓ | **38.37** | **0.991** | **5.36** | <u>37.34</u> | **0.984** | <u>2.50</u> | **34.20** | **0.972** | **2.97** |

![AISTD comparison](assets/qual_aistd.jpg)

Visual comparison on AISTD. Top: non-shadow preservation (yellow arrows mark competitor failures). Bottom: shadow removal. Ours performs well in both.

Results on SRD, LRSS, and UIUC. The best results are highlighted in **bold**.

| Method | SRD PSNR↑ | SRD SSIM↑ | SRD RMSE↓ | LRSS PSNR↑ | LRSS SSIM↑ | LRSS RMSE↓ | UIUC PSNR↑ | UIUC SSIM↑ | UIUC RMSE↓ |
|---|---|---|---|---|---|---|---|---|---|
| S3R-Net | 22.51 | 0.878 | 10.60 | 22.54 | 0.786 | 10.45 | 23.92 | 0.857 | 8.53 |
| DCS | 21.99 | 0.868 | 11.47 | 22.10 | 0.774 | 11.63 | 23.94 | 0.852 | 9.69 |
| G2R | 24.23 | 0.898 | 8.40 | 22.79 | 0.778 | 9.64 | 27.03 | 0.867 | 6.23 |
| DeS3 | 22.44 | 0.900 | 10.29 | 21.10 | 0.767 | 12.75 | 19.02 | 0.822 | 11.23 |
| SSR | 25.86 | 0.922 | 8.25 | 24.47 | 0.790 | 9.89 | 27.79 | 0.873 | 7.61 |
| ODD | 27.38 | 0.941 | 6.46 | 25.12 | 0.797 | 7.84 | 29.25 | 0.886 | 5.10 |
| ODDR | **27.66** | **0.942** | **6.26** | **25.34** | **0.797** | **7.69** | **29.95** | **0.887** | **4.95** |

![Unseen datasets](assets/qual_unseen.jpg)

Visual comparisons on SRD, LRSS, and UIUC (top to bottom).

## Code

Coming soon.

## Citation

```bibtex
@inproceedings{shin2026oddr,
  title     = {ODDR: One-Step Deshadow Diffusion via Reward Guidance},
  author    = {Shin, Junseong and Kim, Kijun and Kim, Minseong and Kim, Dongjin and Kim, Tae Hyun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
