#!/bin/bash
# GPU 1: Run MeanFlow (Exp B)
# 策略：代码会自动使用 Batch=128, Steps=100000, No-Compile

mkdir -p checkpoints

# 请确保在运行前已在终端执行过 conda activate MF
conda activate MF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 python train.py \
    --exp_name exp_meanflow \
    --data_dir data/latents \
    --gpu_id 0
