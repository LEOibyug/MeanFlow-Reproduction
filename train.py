import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.func import jvp
from tqdm import tqdm
from models.dit import DiT_MeanFlow

# 开启 A100 黑科技
torch.set_float32_matmul_precision('high')

def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据加载 (VRAM Resident) ---
    print(f"Loading dataset to {device}...")
    try:
        all_latents = torch.load(f"{args.data_dir}/latents.pt", map_location=device)
        all_labels = torch.load(f"{args.data_dir}/labels.pt", map_location=device)
    except FileNotFoundError:
        print(f"Error: latents.pt not found in {args.data_dir}")
        return

    num_samples = all_latents.shape[0]
    print(f"Dataset loaded: {num_samples} samples. VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # --- 2. 动态策略配置 (关键修改!) ---
    if args.baseline:
        # GPU 0: Baseline 模式
        # 任务简单 (无 JVP)，可以使用大 Batch 和编译加速
        BATCH_SIZE = 256       # 激进的大 Batch
        TOTAL_STEPS = 80000     # 2万步 * 1024 = 2000万样本量
        USE_COMPILE = True      # 开启编译
        print(">>> Mode: Baseline (High Throughput Config)")
    else:
        # GPU 1: MeanFlow 模式
        # 任务困难 (JVP 双倍显存)，必须保守
        BATCH_SIZE = 64        # 安全的小 Batch
        TOTAL_STEPS = 200000    # 10万步 * 128 = 1280万样本量 (足够收敛)
        USE_COMPILE = False     # 关闭编译 (避免 JVP 兼容性问题)
        print(">>> Mode: MeanFlow (JVP Safe Config)")

    # 模型初始化
    model = DiT_MeanFlow(num_classes=10).to(device)
    
    if USE_COMPILE:
        print("Compiling model (wait ~1 min)...")
        model = torch.compile(model)
    else:
        print("Skipping compilation for JVP safety.")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    model.train()
    
    # 日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.exp_name}_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss"])

    print(f"Start Training: {args.exp_name} | Steps: {TOTAL_STEPS} | BS: {BATCH_SIZE}")
    pbar = tqdm(range(TOTAL_STEPS))
    
    for step in pbar:
        optimizer.zero_grad()
        
        # 极速采样
        indices = torch.randint(0, num_samples, (BATCH_SIZE,), device=device)
        x = all_latents[indices]
        y = all_labels[indices]
        B = x.shape[0]
        
        # 构造时间 t, r
        t = torch.rand(B, device=device)
        if args.baseline:
            r = t # Baseline: target is instantaneous velocity
        else:
            r = torch.rand(B, device=device) # MeanFlow: target is average velocity
            
        # 构造加噪样本 z_t
        # Flow Matching: z_t = (1-t)x + t*eps
        # 注意: 这里的 t=0 是数据，t=1 是噪声 (符合论文)
        eps = torch.randn_like(x)
        z_t = (1 - t.view(B, 1, 1, 1)) * x + t.view(B, 1, 1, 1) * eps
        v_instant = eps - x # dx/dt = eps - x
        
        # CFG Drop Logic
        if args.use_cfg and torch.rand(1) < 0.1:
            y_cond = torch.ones_like(y) * 10 
        else:
            y_cond = y
            
        # --- 核心分支 ---
        if args.baseline or args.ablation_no_jvp:
            # 简单前向传播
            u_pred = model(z_t, t, r, y_cond)
            target = v_instant
        else:
            # JVP 前向传播
            # 定义闭包
            def model_fn_wrapper(z_in, t_in):
                return model(z_in, t_in, r, y_cond)

            # 计算数值 u 和导数 du/dt
            # Primals: (z, t)
            # Tangents: (dz/dt, dt/dt) = (v, 1)
            u_pred, du_dt = jvp(model_fn_wrapper, (z_t, t), (v_instant, torch.ones_like(t)))
            
            # MeanFlow Target: v - (t-r) * du/dt
            dt = (t - r).view(B, 1, 1, 1)
            target = v_instant - dt * du_dt
            target = target.detach() # Stop gradient!
        
        loss = F.mse_loss(u_pred, target)
        loss.backward()
        optimizer.step()
        
        # 监控与保存
        if step % 100 == 0:
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([step, loss.item()])
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
        # 每 10% 保存一次，或者最后一次
        save_interval = TOTAL_STEPS // 10
        if step > 0 and (step % save_interval == 0 or step == TOTAL_STEPS - 1):
            os.makedirs("checkpoints", exist_ok=True)
            # 处理 compile 后的 state_dict
            state_dict = model.state_dict()
            # 如果是 compiled model, key 会有 _orig_mod 前缀，这里最好处理一下
            # 但为了简单，我们在加载(sample.py)时处理
            torch.save(state_dict, f"checkpoints/{args.exp_name}_step{step}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/latents")
    # batch_size 和 epochs 参数已被废弃，由代码内部策略接管
    parser.add_argument("--gpu_id", type=int, default=0)
    
    parser.add_argument("--baseline", action="store_true", help="Run Standard FM (Exp A)")
    parser.add_argument("--use_cfg", action="store_true", help="Enable CFG logic (Exp C)")
    parser.add_argument("--ablation_no_jvp", action="store_true", help="Disable JVP (Exp D)")
    
    args = parser.parse_args()
    train(args)
