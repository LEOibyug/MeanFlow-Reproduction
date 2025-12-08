#!/bin/bash
# GPU 0: Run Baseline (Exp A)
# 策略：代码会自动使用 Batch=1024, Steps=20000, Compile

mkdir -p checkpoints

# 请确保在运行前已在终端执行过 conda activate MF
conda activate MF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name exp_baseline \
    --data_dir data/latents \
    --baseline \
    --gpu_id 0
