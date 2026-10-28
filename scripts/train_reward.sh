#!/bin/bash
# ShadowReward on human-annotation-free ranked lists (AISTD shadow-free images + SynShadow mattes).
# Settings of the released ShadowReward.pth (the final 20K-step checkpoint), 1 GPU.
python train_reward.py \
    --dataset_path="data/AISTD/train" \
    --gt_dir="train_C_fixed_ours" \
    --matte_dir="data/SynShadow/matte" \
    --output_dir="experiments/ShadowReward" \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=1000 \
    --max_train_steps=20000 \
    --train_batch_size=16 \
    --crop_size=512 \
    --patch_size=448 \
    --topk=0.1 \
    --temperature=1.0 \
    --checkpointing_steps=1000 \
    --dataloader_num_workers=8 \
    --seed=123
