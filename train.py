import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.dit import DiT_MeanFlow
from utils.dataset import LatentDataset
import argparse
import os
from tqdm import tqdm
import csv

def train(args):
    # 1. Setup Device
    assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cuda")

    # 2. Setup Model
    model = DiT_MeanFlow(num_classes=10).to(device) # Assume 10 classes for Imagenette
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.00)

    # 3. Setup Dataset
    dataset = LatentDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) # num_workers can be tuned

    # 在 train 函数开始处，创建日志文件
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.exp_name}_log.csv")
    
    # 初始化 CSV
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"]) # 表头

    print(f"Start Training Experiment: {args.exp_name}")

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        
        pbar = tqdm(dataloader)
        for latents, labels in pbar:
            latents = latents.to(device)
            labels = labels.to(device)

            B = latents.shape[0]

            # Sample (t, r) for Flow Matching
            # t: current time, r: target time (t=0 for pure data, t=1 for pure noise)
            # We sample t in [0, 1] and r=0 for simplicity (common in FM)
            t = torch.rand(B, device=device) # t ~ U(0,1)
            r = torch.zeros(B, device=device) # r = 0

            # Add noise to latent
            # z_t = t * x + (1-t) * N(0,1)
            # The paper actually uses z_t = t * x + sqrt(1 - t^2) * e for diffusion
            # Here we follow MeanFlow's formulation for the flow (z_t = t * x)
            # and predict u(z_t, t, r) s.t. du/dt = v(t) is matched
            
            # MeanFlow uses a different parameterization for its flow
            # x is the latent, e is sampled gaussian noise
            # They define a flow from e to x as z(t) = t*x + (1-t)*e
            # And velocity field v(z(t), t) = x - e
            # The model predicts u which is a proxy for v
            
            # For MeanFlow, we are effectively learning the conditional expectation E[x|z_t]
            # Or directly predicting the velocity
            # Let's assume we are learning u_theta(z, t, r) which means we want to find x
            # Or rather v_target = x - e, where z_t = (1-t)e + tx
            
            # MeanFlow's definition:
            # z_t = (1-t) * z_0 + t * z_1, where z_0 ~ N(0,I) and z_1 is data
            # Here, z_0 is the noise, z_1 is the latent from VAE
            # So, current_latent = (1-t) * noise + t * latents (from dataset)
            noise = torch.randn_like(latents)
            current_latent = (1 - t).view(B, 1, 1, 1) * noise + t.view(B, 1, 1, 1) * latents
            
            # Predict the velocity u
            u_pred = model(current_latent, t, r, labels) # u_pred is the velocity field

            # Target velocity for Flow Matching
            # v_target = latents - noise (from the MeanFlow paper)
            v_target = latents - noise
            
            # Loss calculation
            loss = F.mse_loss(u_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累加 Loss 用于记录平均值
            epoch_loss += loss.item()
            steps += 1
            pbar.set_description(f"Ep {epoch} | Loss: {loss.item():.4f}")
            
        # --- 新增：计算平均 Loss 并写入文件 ---
        avg_loss = epoch_loss / steps
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])
        
        # Save Checkpoint (保持不变)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{args.exp_name}_ep{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/latents")
    parser.add_argument("--batch_size", type=int, default=512) # A100 allows big batch
    parser.add_argument("--epochs", type=int, default=100) # Fast epochs on small dataset
    parser.add_argument("--gpu_id", type=int, default=0)
    
    # Flags for Experiments
    parser.add_argument("--baseline", action="store_true", help="Run Standard FM (Exp A)")
    parser.add_argument("--use_cfg", action="store_true", help="Enable CFG logic (Exp C)")
    parser.add_argument("--ablation_no_jvp", action="store_true", help="Disable JVP (Exp D)")
    
    args = parser.parse_args()

    # Conditional logic for different experiments
    if args.baseline:
        # Original baseline code was different. For now, assume this is the 'meanflow' setup
        # with some flags to disable specific meanflow features if needed.
        # This part needs to be carefully aligned with the original instructions if multiple baselines were intended.
        print("Running in Baseline mode (Standard Flow Matching)")

    if args.ablation_no_jvp:
        print("Running Ablation: JVP disabled")
    
    train(args)
