#!/bin/bash
# Stage 2: ODDR-LoRA fine-tuning with ShadowReward on real AISTD shadow images (no GT).
# Settings of the released ODDR.pkl: 1 GPU, batch size 4, 20K steps; ODDR.pkl is the 16.5K-step checkpoint.
accelerate launch train_oddr.py \
    --pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1-base" \
    --base_ckpt="checkpoints/ODD.pkl" \
    --reward_ckpt="checkpoints/ShadowReward.pth" \
    --dataset_path="data/AISTD/train" \
    --input_dir="train_A" \
    --output_dir="experiments/ODDR" \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --max_train_steps=20000 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --patch_size=512 \
    --lora_rank=16 \
    --lambda_l2=1 \
    --lambda_lpips=2 \
    --lambda_reward=0.1 \
    --reward_topk=0.1 \
    --dataloader_num_workers=4 \
    --checkpointing_steps=500 \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --report_to="tensorboard" \
    --seed=123
