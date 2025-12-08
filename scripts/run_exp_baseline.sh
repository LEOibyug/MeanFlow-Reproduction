#!/bin/bash
# GPU 0: Run Baseline (Exp A)
# 预测结果:1-step 生成全是噪声

mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name exp_baseline \
    --data_dir data/latents \
    --batch_size 1024 \
    --epochs 200 \
    --baseline \
    --gpu_id 0
