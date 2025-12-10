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

# 开启 A100/H100/3090+ 显卡的高精度计算优化
torch.set_float32_matmul_precision('high')

def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据加载 ---
    print(f"Loading dataset to {device}...")
    try:
        all_latents = torch.load(f"{args.data_dir}/latents.pt", map_location=device)
        all_labels = torch.load(f"{args.data_dir}/labels.pt", map_location=device)
    except FileNotFoundError:
        print(f"Error: latents.pt not found in {args.data_dir}")
        return

    num_samples = all_latents.shape[0]
    print(f"Dataset loaded: {num_samples} samples. VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # --- 2. 策略配置 ---
    if args.baseline:
        BATCH_SIZE = 256
        TOTAL_STEPS = 80000
        USE_COMPILE = True
        print(">>> Mode: Baseline (High Throughput Config)")
    else:
        BATCH_SIZE = 64
        TOTAL_STEPS = 200000
        USE_COMPILE = False
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
    
    # 日志初始化
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.exp_name}_log.csv")
    
    # 记录 Weighted Loss 和 Raw MSE
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["step", "weighted_loss", "mse_loss"])

    print(f"Start Training: {args.exp_name} | Steps: {TOTAL_STEPS} | BS: {BATCH_SIZE} | Adaptive P: {args.adaptive_loss_p}")
    pbar = tqdm(range(TOTAL_STEPS))
    
    for step in pbar:
        optimizer.zero_grad()
        
        # --- 数据采样 ---
        indices = torch.randint(0, num_samples, (BATCH_SIZE,), device=device)
        x = all_latents[indices]
        y = all_labels[indices]
        B = x.shape[0]
        
        # --- 时间与路径采样 (修复版) ---
        t_raw = torch.rand(B, device=device)
        r_raw = torch.rand(B, device=device)
        
        if args.baseline:
            t = t_raw
            r = t_raw
        else:
            # MeanFlow 混合策略: 
            # 25% 概率做 MeanFlow (t!=r), 75% 概率做标准 FM (t=r)
            # 这能显著稳定 Loss, 避免纯 MeanFlow 导致的训练困难
            use_meanflow_mask = torch.rand(B, device=device) < 0.25
            
            t = torch.max(t_raw, r_raw)
            r_temp = torch.min(t_raw, r_raw)
            
            # 如果是 MeanFlow 任务，使用 r_temp；否则使用 t (即 r=t)
            r = torch.where(use_meanflow_mask, r_temp, t)
            
            # 仅对 MeanFlow 任务防止 t=r
            r = torch.where(use_meanflow_mask, torch.clamp(r, max=t - 1e-3), r)

        # --- 构造加噪样本 ---
        # z_t = (1-t)x + t*eps
        eps = torch.randn_like(x)
        z_t = (1 - t.view(B, 1, 1, 1)) * x + t.view(B, 1, 1, 1) * eps
        v_instant = eps - x
        
        # CFG Drop
        if args.use_cfg and torch.rand(1) < 0.1:
            y_cond = torch.ones_like(y) * 10 
        else:
            y_cond = y
            
        # --- 前向传播 (含 JVP) ---
        if args.baseline or args.ablation_no_jvp:
            u_pred = model(z_t, t, r, y_cond)
            target = v_instant
        else:
            # 使用 torch.func.jvp 计算全导数
            def model_fn_wrapper(z_in, t_in):
                return model(z_in, t_in, r, y_cond)

            u_pred, du_dt = jvp(model_fn_wrapper, (z_t, t), (v_instant, torch.ones_like(t)))
            
            # MeanFlow Target: v - (t-r) * du/dt
            dt = (t - r).view(B, 1, 1, 1)
            target = v_instant - dt * du_dt
            target = target.detach() # Stop Gradient
        
        # --- Adaptive Loss Weighting ---
        if args.adaptive_loss_p > 0:
            # 1. 计算每个样本的 L2 误差平方
            diff = u_pred - target
            error_sq = diff.pow(2).sum(dim=[1, 2, 3])
            
            # 2. 计算权重 (Detach!)
            c = 1e-3
            weights = 1.0 / (error_sq + c).pow(args.adaptive_loss_p)
            weights = weights.detach()
            
            # 3. 计算用于 Backward 的加权 Loss
            loss = (weights * error_sq).mean()
            
            # 4. 计算 Raw MSE 用于监控 (不参与梯度)
            with torch.no_grad():
                raw_mse = F.mse_loss(u_pred, target)
        else:
            loss = F.mse_loss(u_pred, target)
            raw_mse = loss

        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # --- 监控与保存 ---
        if step % 100 == 0:
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([step, loss.item(), raw_mse.item()])
            
            pbar.set_description(f"MSE: {raw_mse.item():.4f}")
            
        save_interval = TOTAL_STEPS // 10
        if step > 0 and (step % save_interval == 0 or step == TOTAL_STEPS - 1):
            os.makedirs("checkpoints", exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, f"checkpoints/{args.exp_name}_step{step}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/latents")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    parser.add_argument("--baseline", action="store_true", help="Run Standard FM (Exp A)")
    parser.add_argument("--use_cfg", action="store_true", help="Enable CFG logic (Exp C)")
    parser.add_argument("--ablation_no_jvp", action="store_true", help="Disable JVP (Exp D)")
    
    parser.add_argument("--adaptive_loss_p", type=float, default=1.0, 
                        help="Power p for adaptive loss weighting. Set 0 to disable.")
    
    args = parser.parse_args()
    train(args)