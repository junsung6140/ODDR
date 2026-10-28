"""Training datasets of ODDR.

SynShadowDataset     : ODD (stage 1). Shadow-free AISTD images + SynShadow mattes -> synthetic pairs.
RealShadowDataset    : ODDR (stage 2). Real AISTD shadow images only (no ground truth is used).
ShadowRankingDataset : ShadowReward. Human-annotation-free ranked lists built from shadow-free
                       images and SynShadow mattes (GT > easy > medium > hard degradations).
RewardEvalDataset    : optional ShadowReward validation (AISTD test GT vs. a baseline output).
"""
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils import data as data
from torchvision import transforms

from .synshadow import InfiniteSampler, darken

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def list_images(folder):
    return [os.path.join(folder, x) for x in sorted(os.listdir(folder)) if x.lower().endswith(IMG_EXTS)]


def reflect_pad_to(img, min_size):
    """Reflect-pad a PIL image so that both sides are at least `min_size`."""
    h, w = img.height, img.width
    pad_top = max((min_size - h) // 2, 0)
    pad_bottom = max(min_size - h - pad_top, 0)
    pad_left = max((min_size - w) // 2, 0)
    pad_right = max(min_size - w - pad_left, 0)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img = TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], padding_mode='reflect')
    return img


def color_jitter(images, p=0.3):
    """Apply the same random saturation and hue shift (random order) to a list of [3, H, W] tensors."""
    hue_interval = [-p, p]
    saturation_interval = [1 - p, 1 + p]
    fn_ids = torch.randperm(2)
    h_value = torch.empty(1).uniform_(hue_interval[0], hue_interval[1])
    s_value = torch.empty(1).uniform_(saturation_interval[0], saturation_interval[1])
    out = []
    for img in images:
        for fn_id in fn_ids:
            if fn_id == 0:
                img = TF.adjust_saturation(img, s_value)
            if fn_id == 1:
                img = TF.adjust_hue(img, h_value)
        out.append(img)
    return out


class SynShadowDataset(data.Dataset):
    """ODD training pairs: (synthetic shadow image, shadow-free image)."""

    def __init__(self, dataset_path, matte_dir, gt_dir='train_C_fixed_ours', crop_size=512,
                 random_flip=True, rotation_degree=(90, 180, 270), args=None):
        super().__init__()
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.rotation_degree = list(rotation_degree)
        self.clean_filenames = list_images(os.path.join(dataset_path, gt_dir))
        self.matte_filenames = list_images(matte_dir)
        self.args = args
        self.matte_iter = InfiniteSampler(len(self.matte_filenames))

    def random_transform(self, img):
        img = reflect_pad_to(img, self.crop_size)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        if random.random() > 0.5 and self.random_flip:
            img = TF.hflip(img)
        angle = random.choice(self.rotation_degree)
        img = TF.rotate(img, angle)
        return img

    def __getitem__(self, index):
        example = dict()
        gt = Image.open(self.clean_filenames[index]).convert('RGB')

        matte_index = next(self.matte_iter)
        matte = Image.open(self.matte_filenames[matte_index]).convert("L")
        matte = matte.resize((self.crop_size, self.crop_size))

        gt = TF.to_tensor(self.random_transform(gt))
        mask = TF.to_tensor(matte)
        gt, = color_jitter([gt])

        # x_syn = (1 - m) * y + m * (a * y + b)
        target_dark = darken(gt, self.args)
        shadow = target_dark * mask + (1 - mask) * gt

        example["output_pixel_values"] = gt * 2.0 - 1.0
        example["conditioning_pixel_values"] = shadow * 2.0 - 1.0
        return example

    def __len__(self):
        return len(self.clean_filenames)


class RealShadowDataset(data.Dataset):
    """ODDR fine-tuning inputs: real shadow images only."""

    def __init__(self, dataset_path, input_dir='train_A', crop_size=512, random_flip=True,
                 rotation_degree=(90, 180, 270)):
        super().__init__()
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.rotation_degree = list(rotation_degree)
        self.shadow_filenames = list_images(os.path.join(dataset_path, input_dir))

    def random_transform(self, img):
        img = reflect_pad_to(img, 512)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        if random.random() > 0.5 and self.random_flip:
            img = TF.hflip(img)
        angle = random.choice(self.rotation_degree)
        img = TF.rotate(img, angle)
        return img

    def __getitem__(self, index):
        example = dict()
        shadow = Image.open(self.shadow_filenames[index]).convert('RGB')
        shadow = TF.to_tensor(self.random_transform(shadow))
        shadow, = color_jitter([shadow])
        example["conditioning_pixel_values"] = shadow * 2.0 - 1.0
        return example

    def __len__(self):
        return len(self.shadow_filenames)


class ShadowRankingDataset(data.Dataset):
    """Human-annotation-free ranked lists for ShadowReward.

    Each sample contains a synthetic shadow input and a list of candidates
    [GT, easy, medium, hard], where each loser is the GT degraded inside the shadow region
    (and its boundary band) with a random mix of failure modes: boundary error, color shift,
    luminance (gamma) change, texture loss / ringing, artifacts (noise + resampling) and
    local warping. Harder levels use stronger degradations and mix more failure modes.
    """

    def __init__(self, dataset_path, matte_dir, gt_dir='train_C_fixed_ours', crop_size=512, patch_size=448,
                 random_flip=True, rotation_degree=(90, 180, 270), args=None,
                 num_losers=3,                      # list length J (easy / medium / hard)
                 difficulty_schedule=(0, 1, 2, 3),  # difficulty level of the i-th loser
                 failure_mix_per_loser=(2, 3, 4, 4),  # number of mixed failure modes of the i-th loser
                 boundary_band_px=(3, 7, 11, 15),   # boundary band radius per difficulty level
                 ):
        super().__init__()
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.random_flip = random_flip
        self.rotation_degree = list(rotation_degree)
        self.num_losers = num_losers
        self.difficulty_schedule = difficulty_schedule
        self.failure_mix_per_loser = failure_mix_per_loser
        self.boundary_band_px = boundary_band_px

        self.clean_filenames = list_images(os.path.join(dataset_path, gt_dir))
        self.matte_filenames = list_images(matte_dir)
        self.args = args
        self.matte_iter = InfiniteSampler(len(self.matte_filenames))

    # ---------------- utils ----------------
    def _get_masks(self, mask_tensor, r):
        """Returns shadow (S), non-shadow (N) and boundary band (B) masks, each [1, H, W]."""
        shadow = (mask_tensor > 0.5).float().unsqueeze(0)  # [1, 1, H, W]
        kernel = torch.ones((1, 1, 2 * r + 1, 2 * r + 1), device=shadow.device)
        dil = torch.nn.functional.conv2d(shadow, kernel, padding=r) > 0
        ero = torch.nn.functional.conv2d(shadow, kernel, padding=r) == ((2 * r + 1) * (2 * r + 1))
        B = (dil ^ ero).float().squeeze(0)
        S = shadow.squeeze(0)
        N = 1.0 - S
        return S, N, B

    def random_transform(self, img1, mask=None):
        # Note: the matte is resized to patch_size before this call and cropped with the same
        # window as the image (as in the original training code).
        min_size = 512
        h1, w1 = img1.height, img1.width
        pad_top1 = max((min_size - h1) // 2, 0)
        pad_bottom1 = max(min_size - h1 - pad_top1, 0)
        pad_left1 = max((min_size - w1) // 2, 0)
        pad_right1 = max(min_size - w1 - pad_left1, 0)
        if pad_top1 > 0 or pad_bottom1 > 0 or pad_left1 > 0 or pad_right1 > 0:
            img1 = TF.pad(img1, padding=[pad_left1, pad_top1, pad_right1, pad_bottom1], padding_mode='reflect')
            if mask is not None:
                mask = TF.pad(mask, padding=[pad_left1, pad_top1, pad_right1, pad_bottom1], padding_mode='reflect')

        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(self.crop_size, self.crop_size))
        img1 = TF.crop(img1, i, j, h, w)
        if mask is not None:
            mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5 and self.random_flip:
            img1 = TF.hflip(img1)
            if mask is not None:
                mask = TF.hflip(mask)

        angle = random.choice(self.rotation_degree)
        img1 = TF.rotate(img1, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)

        img1 = TF.resize(img1, size=(self.patch_size, self.patch_size))
        if mask is not None:
            mask = TF.resize(mask, size=(self.patch_size, self.patch_size), interpolation=TF.InterpolationMode.NEAREST)
        return img1, mask

    def apply_augmentation(self, img1, img2, img3_list):
        """Shared light color jitter for all images of one list."""
        p = 0.3
        hue_interval = [-p, p]
        saturation_interval = [1 - p, 1 + p]
        fn_ids = torch.randperm(2)
        h_value = torch.empty(1).uniform_(*hue_interval)
        s_value = torch.empty(1).uniform_(*saturation_interval)

        def _apply(img):
            out = img
            for fn_id in fn_ids:
                if fn_id == 0:
                    out = TF.adjust_saturation(out, s_value.item())
                if fn_id == 1:
                    out = TF.adjust_hue(out, h_value.item())
            return out

        img1 = _apply(img1)
        img2 = _apply(img2)
        img3_list = [_apply(im) for im in img3_list]
        return img1, img2, img3_list

    # ---------------- failure mode parameters ----------------
    def _param_table(self, level: int):
        """level: 0 (easy), 1 (medium), 2 (hard), 3 (harder). Differences are kept subtle."""
        if level not in (0, 1, 2, 3):
            raise ValueError(f"level must be 0..3, got {level}")

        tbl = {
            # boundary error (halo / double edge): weak blur + small blend in the boundary band
            "boundary": {
                "blur_k": {0: 1, 1: 3, 2: 3, 3: 5}[level],
                "blend_alpha": {0: (0.01, 0.03), 1: (0.02, 0.06), 2: (0.04, 0.09), 3: (0.06, 0.12)}[level],
            },
            # color mismatch: small hue / saturation / brightness change
            "color": {
                "hue": {0: (-0.008, 0.008), 1: (-0.015, 0.015), 2: (-0.025, 0.025), 3: (-0.035, 0.035)}[level],
                "sat": {0: (0.97, 1.03), 1: (0.94, 1.06), 2: (0.90, 1.08), 3: (0.88, 1.12)}[level],
                "bright": {0: (0.98, 1.02), 1: (0.96, 1.04), 2: (0.94, 1.06), 3: (0.92, 1.08)}[level],
            },
            # luminance mismatch: small gamma change
            "luma": {
                "gamma": {0: (0.98, 1.02), 1: (0.96, 1.04), 2: (0.94, 1.06), 3: (0.92, 1.08)}[level],
            },
            # texture: weak blur or unsharp masking
            "texture": {
                "blur_k": {0: 3, 1: 3, 2: 5, 3: 5}[level],
                "unsharp_amt": {0: 0.20, 1: 0.35, 2: 0.50, 3: 0.65}[level],
            },
            # artifacts: noise std and down-scale factor
            "artifact": {
                "noise_std": {0: 0.004, 1: 0.007, 2: 0.012, 3: 0.018}[level],
                "ds_factor": {0: 0.92, 1: 0.88, 2: 0.82, 3: 0.78}[level],
            },
            # structural distortion: displacement and warp grid spacing
            "warp": {
                "disp": {0: 0.20, 1: 0.35, 2: 0.60, 3: 0.85}[level],
                "grid": {0: 80, 1: 64, 2: 48, 3: 40}[level],
            },
        }
        return tbl

    # ---------------- failure modes ----------------
    def _apply_boundary_error(self, img, B, blur_k, blend_alpha):
        blurred = TF.gaussian_blur(img, blur_k)
        alpha = torch.empty_like(B).uniform_(blend_alpha[0], blend_alpha[1])
        alpha = alpha.unsqueeze(0)
        out = img * (1 - B) + (alpha * blurred + (1 - alpha) * img) * B
        return out.clamp(0, 1)

    def _apply_color_shift(self, img, S_or_B_mask, hue_rng, sat_rng, bright_rng):
        hue = float(torch.empty(1).uniform_(*hue_rng))
        sat = float(torch.empty(1).uniform_(*sat_rng))
        bri = float(torch.empty(1).uniform_(*bright_rng))
        shifted = TF.adjust_hue(img, hue)
        shifted = TF.adjust_saturation(shifted, sat)
        shifted = TF.adjust_brightness(shifted, bri)
        out = img * (1 - S_or_B_mask) + shifted * S_or_B_mask
        return out.clamp(0, 1)

    def _apply_gamma(self, img, S_mask, gamma_rng):
        g = float(torch.empty(1).uniform_(*gamma_rng))
        gamma_img = img.clamp(0, 1) ** g
        out = gamma_img * S_mask + img * (1 - S_mask)
        return out.clamp(0, 1)

    def _apply_texture(self, img, S_or_B_mask, blur_k, unsharp_amt):
        prob = random.random()
        if prob < 0.5:
            # smoothing (texture loss)
            blurred = TF.gaussian_blur(img, blur_k)
            out = blurred * S_or_B_mask + img * (1 - S_or_B_mask)
        else:
            # unsharp masking (ringing)
            blurred = TF.gaussian_blur(img, max(3, blur_k))
            sharpened = torch.clamp(img + unsharp_amt * (img - blurred), 0, 1)
            out = sharpened * S_or_B_mask + img * (1 - S_or_B_mask)
        return out.clamp(0, 1)

    def _apply_artifact(self, img, whole_mask, noise_std, ds_factor):
        """Gaussian noise + bilinear down / nearest up sampling (blocking-like artifacts)."""
        img = self._ensure_chw(img)
        whole_mask = self._ensure_1hw(whole_mask)
        device = img.device
        dtype = img.dtype

        noisy = torch.clamp(img + torch.randn_like(img) * float(noise_std), 0, 1)

        C, H, W = noisy.shape
        h2 = max(8, int(round(H * float(ds_factor))))
        w2 = max(8, int(round(W * float(ds_factor))))
        low = torch.nn.functional.interpolate(noisy.unsqueeze(0), size=(h2, w2), mode='bilinear', align_corners=False)
        up = torch.nn.functional.interpolate(low, size=(H, W), mode='nearest').squeeze(0)

        mask3 = whole_mask.expand(C, H, W).to(device=device, dtype=dtype)
        out = up * mask3 + img * (1 - mask3)
        return out.clamp(0, 1)

    def _apply_warp(self, img, B_mask, disp, grid):
        """Smooth random warp, strongest around the shadow boundary band."""
        img = self._ensure_chw(img)
        if isinstance(B_mask, Image.Image):
            B_mask = TF.to_tensor(B_mask.convert("L"))
        if B_mask.dim() == 2:
            B_mask = B_mask.unsqueeze(0)
        elif B_mask.dim() == 3 and B_mask.size(0) != 1:
            B_mask = B_mask.mean(dim=0, keepdim=True)
        img = img.contiguous()
        B_mask = B_mask.contiguous()

        C, H, W = img.shape
        device = img.device
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        yy = yy.float()
        xx = xx.float()

        gy = max(2, H // grid)
        gx = max(2, W // grid)
        dy = (torch.randn(1, 1, gy, gx, device=device) * float(disp))
        dx = (torch.randn(1, 1, gy, gx, device=device) * float(disp))
        dy = torch.nn.functional.interpolate(dy, size=(H, W), mode='bilinear', align_corners=False).squeeze()
        dx = torch.nn.functional.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False).squeeze()

        weight = TF.gaussian_blur(B_mask, kernel_size=9).squeeze(0)  # [H, W]

        new_y = (yy + dy * weight).clamp(0, H - 1)
        new_x = (xx + dx * weight).clamp(0, W - 1)
        grid_y = (new_y / (H - 1)) * 2 - 1
        grid_x = (new_x / (W - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(img.dtype)  # [1, H, W, 2]

        out = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, mode='bilinear', align_corners=False)
        return out.squeeze(0).clamp(0, 1)

    def apply_failure_mix(self, img, S, N, B, tbl, modes, mix_k):
        chosen = random.sample(modes, k=min(mix_k, len(modes)))
        for m in chosen:
            if m == "boundary":
                img = self._apply_boundary_error(img, B, tbl["boundary"]["blur_k"], tbl["boundary"]["blend_alpha"])
            elif m == "color":
                target = S if random.random() < 0.6 else B
                img = self._apply_color_shift(img, target, tbl["color"]["hue"], tbl["color"]["sat"], tbl["color"]["bright"])
            elif m == "luma":
                img = self._apply_gamma(img, S, tbl["luma"]["gamma"])
            elif m == "texture":
                target = S if random.random() < 0.5 else B
                img = self._apply_texture(img, target, tbl["texture"]["blur_k"], tbl["texture"]["unsharp_amt"])
            elif m == "artifact":
                whole = torch.ones_like(S)
                img = self._apply_artifact(img, whole, tbl["artifact"]["noise_std"], tbl["artifact"]["ds_factor"])
            elif m == "warp":
                img = self._apply_warp(img, B, tbl["warp"]["disp"], tbl["warp"]["grid"])
        return img.clamp(0, 1)

    def __getitem__(self, index):
        example = dict()

        gt = Image.open(self.clean_filenames[index]).convert('RGB')
        matte_index = next(self.matte_iter)
        matte = Image.open(self.matte_filenames[matte_index]).convert("L")
        mask = matte.resize((self.patch_size, self.patch_size), Image.NEAREST)

        gt, mask = self.random_transform(gt, mask)
        gt_tensor = TF.to_tensor(gt)
        mask_tensor = TF.to_tensor(mask)

        # synthetic shadow input
        target_dark = darken(gt_tensor, self.args)
        shadow_tensor = target_dark * mask_tensor + (1 - mask_tensor) * gt_tensor

        # losers with increasing difficulty
        modes = ["boundary", "color", "luma", "texture", "artifact", "warp"]
        losers = []
        for i in range(self.num_losers):
            level = self.difficulty_schedule[i % len(self.difficulty_schedule)]
            tbl = self._param_table(level)
            S, N, B = self._get_masks(mask_tensor, self.boundary_band_px[level])
            mix_k = self.failure_mix_per_loser[i % len(self.failure_mix_per_loser)]
            losers.append(self.apply_failure_mix(gt_tensor.clone(), S, N, B, tbl, modes, mix_k))

        # some failure modes (e.g. boundary) return [1, 3, H, W]
        losers = [l.squeeze(0) if (l.dim() == 4 and l.size(0) == 1) else l for l in losers]
        gt_tensor, shadow_tensor, losers = self.apply_augmentation(gt_tensor, shadow_tensor, losers)
        losers = [l.clamp(0, 1) for l in losers]

        example["input"] = shadow_tensor                                  # [3, H, W]
        example["candidates"] = torch.stack([gt_tensor] + losers, dim=0)  # [J + 1, 3, H, W], GT first
        example["rank"] = torch.arange(self.num_losers + 1, dtype=torch.long)  # [0, 1, 2, 3], 0 is the best
        return example

    def _ensure_chw(self, img):
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        if img.dim() == 4:
            if img.size(0) == 1:
                img = img.squeeze(0)
            else:
                raise ValueError(f"Expected CHW or [1,C,H,W], got {tuple(img.shape)}")
        elif img.dim() == 3:
            if img.shape[0] not in (1, 3) and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Expected CHW, got dim={img.dim()}")
        return img.contiguous()

    def _ensure_1hw(self, mask):
        if isinstance(mask, Image.Image):
            mask = TF.to_tensor(mask.convert("L"))
        if mask.dim() == 3 and mask.size(0) != 1 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1).contiguous()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3 or mask.size(0) != 1:
            mask = mask.mean(dim=0, keepdim=True)
        return mask.clamp(0, 1).contiguous()

    def __len__(self):
        return len(self.clean_filenames)


class RewardEvalDataset(data.Dataset):
    """Pairwise check used to monitor ShadowReward: is score(input, GT) > score(input, baseline output)?"""

    def __init__(self, test_path, baseline_dir, gt_dir='test_C_fixed_official', input_dir='test_A', patch_size=448):
        super().__init__()
        self.clean_filenames = [x for x in list_images(os.path.join(test_path, gt_dir)) if x.endswith('.png')]
        self.noisy_filenames = [x for x in list_images(os.path.join(test_path, input_dir)) if x.endswith('.png')]
        self.deshadow_filenames = [x for x in list_images(baseline_dir) if x.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        return {
            "gt": self.transform(Image.open(self.clean_filenames[index]).convert('RGB')),
            "shadow": self.transform(Image.open(self.noisy_filenames[index]).convert('RGB')),
            "deshadow": self.transform(Image.open(self.deshadow_filenames[index]).convert('RGB')),
        }

    def __len__(self):
        return len(self.clean_filenames)
