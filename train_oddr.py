"""Stage 2: fine-tune ODDR-LoRA on real shadow images with ShadowReward.

L_ODDR = ||y_ODDR - y_ODD||_2^2 + lambda_align * LPIPS(y_ODDR, y_ODD) - lambda_r * r(x_real, y_ODDR)
(Eq. 4 of the paper). y_ODD is the output of the frozen ODD model; no ground truth is used.
"""
import argparse
import os
from pathlib import Path

import diffusers
import lpips
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm

from dataloaders.shadow_dataset import RealShadowDataset
from oddr import ODD_ADAPTERS, ODDR_ADAPTER, ODDTrainModel
from shadow_reward import ShadowReward
from train_odd import build_lr_scheduler


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--pretrained_model_name_or_path", default="checkpoints/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--base_ckpt", default="checkpoints/ODD.pkl", type=str, help="ODD checkpoint (stage 1)")
    parser.add_argument("--reward_ckpt", default="checkpoints/ShadowReward.pth", type=str)
    parser.add_argument("--dataset_path", default="data/AISTD/train", type=str, help="AISTD training folder")
    parser.add_argument("--input_dir", default="train_A", type=str, help="Real shadow image sub-folder")
    parser.add_argument("--output_dir", default="experiments/ODDR")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="train_oddr")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # training
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # loss
    parser.add_argument("--lambda_l2", default=1.0, type=float, help="weight of the MSE alignment to ODD")
    parser.add_argument("--lambda_lpips", default=2.0, type=float, help="lambda_align")
    parser.add_argument("--lambda_reward", default=0.1, type=float, help="lambda_r")
    parser.add_argument("--reward_topk", default=0.1, type=float, help="Top-K ratio of ShadowReward pooling")
    parser.add_argument("--reward_size", default=448, type=int, help="ShadowReward input resolution")

    # LoRA
    parser.add_argument("--lora_rank", default=16, type=int)

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def to_reward_input(x, size=448):
    """[-1, 1] -> [0, 1], resized to size x size (ShadowReward normalizes internally)."""
    x = (x + 1) / 2
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # frozen reward model
    reward_model = ShadowReward(feature_mode='concat', middle_dim=1024)
    reward_model.load(args.reward_ckpt)
    reward_model.eval()
    reward_model.requires_grad_(False)
    reward_model = accelerator.prepare(reward_model)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # frozen ODD (anchor) and trainable ODDR (ODD + ODDR-LoRA)
    base_ckpt = torch.load(args.base_ckpt, map_location="cpu")
    model_base = ODDTrainModel(args)
    model_base.load_ckpt(base_ckpt)

    model_gen = ODDTrainModel(args)
    model_gen.load_ckpt(base_ckpt)
    model_gen.set_train_oddr()
    model_gen.unet.set_adapter(ODD_ADAPTERS + [ODDR_ADAPTER])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # trainable parameters: ODDR-LoRA only
    layers_to_opt = [p for n, p in model_gen.unet.named_parameters() if ODDR_ADAPTER in n]

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)
    # stage 2 does not scale the warmup by the number of processes
    lr_scheduler = build_lr_scheduler(args, optimizer, num_processes=1)

    dataset_train = RealShadowDataset(dataset_path=args.dataset_path, input_dir=args.input_dir,
                                      crop_size=args.patch_size)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)

    model_gen, model_base, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        model_gen, model_base, optimizer, dl_train, lr_scheduler)
    net_lpips = accelerator.prepare(net_lpips)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(model_gen):
                x_src = batch["conditioning_pixel_values"]

                x_tgt_pred = model_gen(x_src, batch=batch, args=args)
                with torch.no_grad():
                    x_base = model_base(x_src)

                loss_reward = - reward_model(to_reward_input(x_src.float(), args.reward_size),
                                             to_reward_input(x_tgt_pred.float(), args.reward_size),
                                             reduce="topk", topk=args.reward_topk).view(-1).mean() * args.lambda_reward
                loss_l2 = F.mse_loss(x_tgt_pred, x_base, reduction='mean') * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_base.float()).mean() * args.lambda_lpips
                loss = loss_reward + loss_l2 + loss_lpips

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_reward": loss_reward.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "lr": optimizer.param_groups[0]['lr'],
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

                    accelerator.log(logs, step=global_step)
            if global_step > args.max_train_steps:
                break
        if global_step > args.max_train_steps:
            break


if __name__ == "__main__":
    main(parse_args())
