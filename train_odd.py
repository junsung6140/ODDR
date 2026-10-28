"""Stage 1: train ODD (ODD-LoRA + ODD-Conv) on synthetic shadow pairs.

L_ODD = ||y_hat - y||_1 + lambda_per * LPIPS(y_hat, y)   (Eq. 3 of the paper)
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
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm

from dataloaders.shadow_dataset import SynShadowDataset
from dataloaders.synshadow import add_synshadow_args
from oddr import ODD_ADAPTERS, ODDTrainModel


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--pretrained_model_name_or_path", default="checkpoints/stable-diffusion-2-1-base", type=str,
                        help="Local diffusers folder of Stable Diffusion 2.1-base")
    parser.add_argument("--dataset_path", default="data/AISTD/train", type=str,
                        help="AISTD training folder (contains train_C_fixed_ours)")
    parser.add_argument("--gt_dir", default="train_C_fixed_ours", type=str, help="Shadow-free image sub-folder")
    parser.add_argument("--matte_dir", default="data/SynShadow/matte", type=str, help="SynShadow shadow mattes")
    parser.add_argument("--output_dir", default="experiments/ODD")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="train_odd")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # training
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=40000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        help='"cosine" (linear warmup + cosine annealing to 1e-6) or any diffusers scheduler')
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
    parser.add_argument("--lambda_l1", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float, help="lambda_per")

    # LoRA
    parser.add_argument("--lora_rank", default=16, type=int)

    # synthetic shadow (SynShadow affine model)
    add_synshadow_args(parser)

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def build_lr_scheduler(args, optimizer, num_processes):
    if args.lr_scheduler == 'cosine':
        # linear warmup from 1e-6, then cosine annealing to 1e-6
        warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps * num_processes
        base_lr = optimizer.param_groups[0]["lr"]
        min_warmup_lr = 1e-6

        def warmup_lambda(current_step, warmup_steps):
            warmup_factor = float(current_step) / float(max(1, warmup_steps))
            return warmup_factor * (1.0 - min_warmup_lr / base_lr) + (min_warmup_lr / base_lr)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: warmup_lambda(step, warmup_steps))
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=args.max_train_steps * args.gradient_accumulation_steps * num_processes,
            eta_min=1e-6,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


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

    model_gen = ODDTrainModel(args)
    model_gen.set_train_odd()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    model_gen.unet.set_adapter(ODD_ADAPTERS)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # trainable parameters: ODD-LoRA, UNet conv_in, ODD-Conv
    layers_to_opt = [p for n, p in model_gen.unet.named_parameters() if "lora" in n]
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    layers_to_opt += list(model_gen.vae.decoder.skip_conv_1.parameters())
    layers_to_opt += list(model_gen.vae.decoder.skip_conv_2.parameters())
    layers_to_opt += list(model_gen.vae.decoder.skip_conv_3.parameters())
    layers_to_opt += list(model_gen.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)
    lr_scheduler = build_lr_scheduler(args, optimizer, accelerator.num_processes)

    dataset_train = SynShadowDataset(dataset_path=args.dataset_path, matte_dir=args.matte_dir, gt_dir=args.gt_dir,
                                     crop_size=args.patch_size, args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)

    model_gen, optimizer, dl_train, lr_scheduler = accelerator.prepare(model_gen, optimizer, dl_train, lr_scheduler)
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
                x_tgt = batch["output_pixel_values"]

                x_tgt_pred = model_gen(x_src, batch=batch, args=args)

                loss_l1 = F.l1_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l1
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l1 + loss_lpips

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
                        "loss_l1": loss_l1.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
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
