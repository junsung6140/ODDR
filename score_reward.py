"""Score deshadowed images with ShadowReward.

Example:
    python score_reward.py --input_dir data/AISTD/test/test_A --result_dir results/AISTD
"""
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from shadow_reward import ShadowReward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='shadow input images')
    parser.add_argument('--result_dir', required=True, help='deshadowed images (same file names, png)')
    parser.add_argument('--reward_ckpt', default='checkpoints/ShadowReward.pth')
    parser.add_argument('--size', type=int, default=448)
    parser.add_argument('--topk', type=float, default=0.1)
    args = parser.parse_args()

    model = ShadowReward(feature_mode='concat', middle_dim=1024)
    model.load(args.reward_ckpt)
    model.eval()

    to_tensor = transforms.ToTensor()
    scores = []
    for inp_path in sorted(glob.glob(os.path.join(args.input_dir, '*'))):
        name = os.path.splitext(os.path.basename(inp_path))[0] + '.png'
        res_path = os.path.join(args.result_dir, name)
        if not os.path.isfile(res_path):
            continue
        x = to_tensor(Image.open(inp_path).convert('RGB')).unsqueeze(0).cuda()
        y = to_tensor(Image.open(res_path).convert('RGB')).unsqueeze(0).cuda()
        x = F.interpolate(x, size=args.size, mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=args.size, mode='bilinear', align_corners=False)
        with torch.no_grad():
            s = model(x, y, reduce='topk', topk=args.topk).item()
        scores.append(s)
        print(f'{name}\t{s:.6f}')
    print(f'Mean ShadowReward over {len(scores)} images: {sum(scores) / max(1, len(scores)):.6f}')


if __name__ == "__main__":
    main()
