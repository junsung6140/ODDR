"""Train ShadowReward with the ListMLE ranking loss (Eq. 7 of the paper) on synthetic ranked lists.

Only the convolutional head is trained; the DINOv2 encoder is frozen.
"""
import argparse
import os

import torch
import torch.utils.data
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloaders.shadow_dataset import RewardEvalDataset, ShadowRankingDataset
from dataloaders.synshadow import add_synshadow_args
from shadow_reward import ShadowReward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/AISTD/train", help="AISTD training folder")
    parser.add_argument("--gt_dir", type=str, default="train_C_fixed_ours", help="Shadow-free image sub-folder")
    parser.add_argument("--matte_dir", type=str, default="data/SynShadow/matte", help="SynShadow shadow mattes")
    parser.add_argument("--output_dir", default="experiments/ShadowReward")
    parser.add_argument("--tracker_project_name", type=str, default="train_reward")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=512, help="random crop size before resizing")
    parser.add_argument("--patch_size", type=int, default=448, help="ShadowReward input size (multiple of 14)")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--topk", type=float, default=0.1, help="Top-K ratio of the score pooling")
    parser.add_argument("--temperature", type=float, default=1.0, help="ListMLE temperature tau")

    # optional validation: pairwise accuracy of score(input, GT) > score(input, baseline output) on AISTD test
    parser.add_argument("--eval_test_path", type=str, default=None, help="AISTD test folder (optional)")
    parser.add_argument("--eval_baseline_dir", type=str, default=None, help="Outputs of a baseline on AISTD test")
    parser.add_argument("--eval_steps", type=int, default=100)

    add_synshadow_args(parser)
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShadowReward(feature_mode="concat", middle_dim=1024).to(device)
    model.set_train()
    layers_to_opt = list(model.token_head.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps, num_cycles=args.lr_num_cycles,
                                 power=args.lr_power)

    dataset_train = ShadowRankingDataset(dataset_path=args.dataset_path, matte_dir=args.matte_dir, gt_dir=args.gt_dir,
                                         crop_size=args.crop_size, patch_size=args.patch_size, args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)

    dl_test = None
    if args.eval_test_path is not None and args.eval_baseline_dir is not None:
        dataset_test = RewardEvalDataset(args.eval_test_path, args.eval_baseline_dir, patch_size=args.patch_size)
        dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "logs", args.tracker_project_name))

    model.train()
    global_step = 0
    best_accuracy = 0.0
    progress_bar = tqdm(range(args.max_train_steps), desc="Steps")

    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            inp = batch["input"].to(device)          # [B, 3, H, W] synthetic shadow image
            cands = batch["candidates"].to(device)   # [B, K, 3, H, W] (GT, easy, medium, hard)
            ranks = batch["rank"].to(device)         # [B, K] (0 is the best)

            optimizer.zero_grad()
            loss = model.get_listwise_loss(shadow=inp, candidates=cands, ranks=ranks, reduce="topk",
                                           topk=args.topk, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            progress_bar.update(1)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

            if dl_test is not None and global_step % args.eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    correct = []
                    for test_batch in dl_test:
                        x_inp = test_batch["shadow"].to(device)
                        gt_score = model(x_inp, test_batch["gt"].to(device), reduce="topk", topk=args.topk)
                        ds_score = model(x_inp, test_batch["deshadow"].to(device), reduce="topk", topk=args.topk)
                        correct.append((gt_score > ds_score).cpu())
                    accuracy = torch.cat(correct).float().mean().item()
                writer.add_scalar("eval/accuracy", accuracy, global_step)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    model.save(os.path.join(args.output_dir, "checkpoints", "best_model.pth"))
                model.token_head.train()

            if global_step % args.checkpointing_steps == 0:
                model.save(os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pth"))

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    writer.close()


if __name__ == "__main__":
    main(parse_args())
