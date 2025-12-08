#!/bin/bash
# GPU 1: Run MeanFlow (Exp B)
# 预测结果：能生成清晰图像

mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=1 python train.py \
    --exp_name exp_meanflow \
    --data_dir data/latents \
    --batch_size 1024 \
    --epochs 200 \
    --gpu_id 0  # 这里指容器内相对ID，如果用CUDA_VISIBLE_DEVICES控制，代码丈即可
