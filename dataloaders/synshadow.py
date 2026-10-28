"""Synthetic shadow generation with the affine illumination model of SynShadow.

Adapted from SynShadow (Inoue and Yamasaki, "Learning from Synthetic Shadows for Shadow
Detection and Removal", https://github.com/naoto0804/SynShadow, MIT License).

A shadow image is synthesized from a shadow-free image y and a shadow matte m as
    x = (1 - m) * y + m * (a * y + b),
where the slope a < 1 and the per-channel intercept b are sampled randomly.
"""
import random

import numpy as np
import torch


def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


def sample_darkening_params(opt):
    """Sample two points (x1, 0) and (1, y2) of the darkening line and return its slope."""
    assert 0.0 <= opt.xmin_at_y_0 <= 1.0
    assert 0.0 <= opt.xmax_at_y_0 <= 1.0
    assert 0.0 <= opt.ymin_at_x_255 <= 1.0
    assert 0.0 <= opt.ymax_at_x_255 <= 1.0
    assert 1.0 <= opt.slope_max

    while True:
        x1 = random.uniform(opt.xmin_at_y_0, opt.xmax_at_y_0)
        y1 = 0.0
        x2 = 1.0
        y2 = random.uniform(opt.ymin_at_x_255, opt.ymax_at_x_255)
        a = (y2 - y1) / (x2 - x1)
        if a < 1.0:
            break
    return a, x1, y1, x2, y2


def darken(x: torch.Tensor, opt):
    """Darken a [3, H, W] image in [0, 1] with a random affine model (shared slope, per-channel intercept)."""
    a, x1, y1, x2, y2 = sample_darkening_params(opt)
    if opt.x_turb_sigma == 0.0:
        b = y1 - a * x1
    else:
        # R and B intercepts are shifted from G (shadows are usually bluish)
        x1_G = x1
        mu, sigma = opt.x_turb_mu, opt.x_turb_sigma
        x1_R = x1_G + np.random.normal(mu, sigma)
        x1_B = x1_G - np.random.normal(mu, sigma)

        b = [y1 - a * x1_R, y1 - a * x1_G, y1 - a * x1_B]
        b = torch.Tensor(b).view(3, 1, 1)
        b = b.repeat(1, x.size(1), x.size(2))

    y = a * x + b
    y = torch.clamp(y, min=0.0, max=1.0)
    return y


def add_synshadow_args(parser):
    """Arguments of the affine darkening model (defaults of SynShadow)."""
    parser.add_argument('--xmin_at_y_0', type=float, default=0.0)
    parser.add_argument('--xmax_at_y_0', type=float, default=0.25)
    parser.add_argument('--ymax_at_x_255', type=float, default=0.9)
    parser.add_argument('--ymin_at_x_255', type=float, default=0.1)
    parser.add_argument('--x_turb_mu', type=float, default=0.05)
    parser.add_argument('--x_turb_sigma', type=float, default=0.025)
    parser.add_argument('--slope_max', type=float, default=1.0)
    return parser
