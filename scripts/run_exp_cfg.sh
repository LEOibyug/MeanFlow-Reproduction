#!/bin/bash
# GPU 0: Run MeanFlow with CFG (Exp C)
# 目标：验证 1-NFE 下的 CFG 能力。
# 预测结果：生成的图像主体（如鱼、狗）特征会比普通 MeanFlow 更明显，背景更纯净。

mkdir -p checkpoints

# 使用 GPU 0 运行
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name exp_cfg \
    --data_dir data/latents \
    --batch_size 1024 \
    --epochs 200 \
    --use_cfg \
    --gpu_id 0
