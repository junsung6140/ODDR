#!/bin/bash
# Stage 1: ODD on synthetic shadow pairs (AISTD shadow-free images + SynShadow mattes).
# Settings of the released ODD.pkl: 1 GPU, batch size 4, 40K steps; ODD.pkl is the 30K-step checkpoint.
accelerate launch train_odd.py \
    --pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1-base" \
    --dataset_path="data/AISTD/train" \
    --gt_dir="train_C_fixed_ours" \
    --matte_dir="data/SynShadow/matte" \
    --output_dir="experiments/ODD" \
    --learning_rate=2e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --max_train_steps=40000 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --patch_size=512 \
    --lora_rank=16 \
    --lambda_l1=1 \
    --lambda_lpips=2 \
    --dataloader_num_workers=4 \
    --checkpointing_steps=1000 \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --report_to="tensorboard" \
    --seed=123
