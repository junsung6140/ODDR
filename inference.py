"""Mask-free one-step shadow removal with ODDR (or ODD).

Example:
    python inference.py -i data/AISTD/test/test_A -o results/AISTD --ckpt_path checkpoints/ODDR.pkl
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from oddr import ODDInference

IMG_EXTS = ('png', 'jpg', 'jpeg', 'bmp', 'PNG', 'JPG', 'JPEG')


def pad_to_multiple_of_64(image):
    """Reflect-pad a PIL image so that both sides are multiples of 64."""
    ori_width, ori_height = image.size
    pad_h = (64 - ori_height % 64) % 64
    pad_w = (64 - ori_width % 64) % 64
    if pad_h > 0 or pad_w > 0:
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img_np = np.array(image)
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
        img_np = np.pad(img_np, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        image = Image.fromarray(img_np)
    return image


def unpad_to_original_size(padded_image, original_size):
    """Center-crop the padded image back to its original size."""
    ori_width, ori_height = original_size
    padded_width, padded_height = padded_image.size
    left = (padded_width - ori_width) // 2
    top = (padded_height - ori_height) // 2
    return padded_image.crop((left, top, left + ori_width, top + ori_height))


def get_image_list(path):
    if os.path.isdir(path):
        names = []
        for ext in IMG_EXTS:
            names += glob.glob(os.path.join(path, f'*.{ext}'))
        return sorted(set(names))
    return [path]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True, help='an input image or a folder of images')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='folder to save the results (png)')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/ODDR.pkl', help='ODDR.pkl or ODD.pkl')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='checkpoints/stable-diffusion-2-1-base',
                        help='Local diffusers folder of Stable Diffusion 2.1-base')
    parser.add_argument('--seed', type=int, default=42, help='seed of the VAE posterior sampling')
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # latent tiling for large images (in latent pixels, 1 latent pixel = 8 image pixels)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model = ODDInference(args)

    image_names = get_image_list(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        original_size = input_image.size
        lq = transforms.ToTensor()(pad_to_multiple_of_64(input_image))

        with torch.no_grad():
            lq = lq * 2 - 1
            output_image = model(lq.unsqueeze(0).to('cuda'))[0]
        output_pil = transforms.ToPILImage()(output_image.cpu() * 0.5 + 0.5)
        output_pil = unpad_to_original_size(output_pil, original_size)

        bname = os.path.splitext(os.path.basename(image_name))[0] + '.png'
        output_pil.save(os.path.join(args.output_dir, bname))


if __name__ == "__main__":
    main()
