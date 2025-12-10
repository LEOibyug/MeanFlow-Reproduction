import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
from PIL import Image
import os
import argparse
import numpy as np
import time
from tqdm import tqdm
from models.dit import DiT_MeanFlow

def clean_state_dict(state_dict):
    """移除 torch.compile 产生的 _orig_mod 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def sample_compare(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    
    print(f"Loading MeanFlow model from {args.ckpt}...")
    model = DiT_MeanFlow(num_classes=10).to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(clean_state_dict(state_dict))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    model.eval()
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # --- 1. 固定初始条件 (Control Variates) ---
    target_class = 1 # English Springer (ID 1)
    y = torch.tensor([target_class], device=device)
    
    # 使用相同的初始噪声，确保两条轨迹起跑线一致
    z_init = torch.randn(1, 4, 32, 32, device=device)
    
    # 准备时间步
    steps = args.steps
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    print(f"\nComparing 2 Strategies ({steps} steps, CFG={args.cfg_scale})...")

    # ==========================================
    # Strategy A: Standard MeanFlow (Next-Step)
    # ==========================================
    print(">>> Running Strategy A: Standard (Target = Next Step)...")
    z = z_init.clone()
    trajectory_A = [z.clone()]
    
    start_time_A = time.time()
    with torch.no_grad():
        for i in tqdm(range(steps), desc="Strategy A"):
            t_curr = timesteps[i]
            t_next = timesteps[i+1] # Target is next step
            
            t_input = torch.ones(1, device=device) * t_curr
            r_input = torch.ones(1, device=device) * t_next # <--- A: r = t_next
            
            # CFG Inference
            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t_input, t_input], dim=0)
                r_in = torch.cat([r_input, r_input], dim=0)
                y_uncond = torch.tensor([10], device=device)
                y_in = torch.cat([y, y_uncond], dim=0)
                
                u_out = model(z_in, t_in, r_in, y_in)
                u_cond, u_uncond = u_out.chunk(2, dim=0)
                u = u_uncond + args.cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z, t_input, r_input, y)
            
            dt = t_curr - t_next
            z = z - u * dt
            trajectory_A.append(z.clone())
    time_A = time.time() - start_time_A

    # ==========================================
    # Strategy B: Recursive Loop (t=1, r=1)
    # ==========================================
    print(">>> Running Strategy B: Recursive Loop (t=1, r=1)...")
    z = z_init.clone() # Reset to same noise
    trajectory_B = [z.clone()]
    
    # 每一轮的步长 (假设总共走完 1.0 的距离)
    dt_fixed = 1.0 / steps
    
    start_time_B = time.time()
    with torch.no_grad():
        for i in tqdm(range(steps), desc="Strategy B"):
            # 逻辑: 始终认为当前输入是噪声 (t=1), 且目标参数也设为 r=1
            t_input = torch.ones(1, device=device) * 1.0 
            r_input = torch.ones(1, device=device) * 1.0 # <--- B: r = 1 (Always)
            
            # CFG Inference
            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t_input, t_input], dim=0)
                r_in = torch.cat([r_input, r_input], dim=0)
                y_uncond = torch.tensor([10], device=device)
                y_in = torch.cat([y, y_uncond], dim=0)
                
                u_out = model(z_in, t_in, r_in, y_in)
                u_cond, u_uncond = u_out.chunk(2, dim=0)
                u = u_uncond + args.cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z, t_input, r_input, y)
            
            # Update z based on output u
            # 注意: 如果模型严格遵循 MeanFlow 定义，t=1, r=1 时 u 可能接近 0
            z = z - u * dt_fixed
            trajectory_B.append(z.clone())
    time_B = time.time() - start_time_B

    # ==========================================
    # Visualization & Layout
    # ==========================================
    # 1. Select Indices (10 frames)
    total_states = len(trajectory_A)
    num_frames = 10
    if total_states <= num_frames:
        indices = list(range(total_states))
    else:
        indices = np.linspace(0, total_states - 1, num_frames, dtype=int)
    
    print(f"Visualizing frames: {indices}")
    
    # 2. Collect Latents
    # Row 1: Strategy A frames
    latents_A = torch.cat([trajectory_A[i] for i in indices], dim=0)
    # Row 2: Strategy B frames
    latents_B = torch.cat([trajectory_B[i] for i in indices], dim=0)
    
    # Combine all (Total 20 frames if steps >= 10)
    all_latents = torch.cat([latents_A, latents_B], dim=0)
    
    # 3. Batch Decode
    with torch.no_grad():
        all_latents = all_latents / 0.18215
        images = []
        # Decode one by one to save VRAM
        for j in range(all_latents.shape[0]):
            img = vae.decode(all_latents[j:j+1]).sample
            images.append(img)
        images = torch.cat(images, dim=0)

    # 4. Save Image
    images = (images / 2 + 0.5).clamp(0, 1)
    
    # nrow=10 意味着每行放 10 张图。
    # 自动排成 2 行 x 10 列
    grid = make_grid(images, nrow=len(indices), padding=2, pad_value=1.0)
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    save_path = f"results/{args.output_name}_compare_{steps}steps.png"
    im.save(save_path)
    
    print("\n" + "="*40)
    print(f"Comparison Complete!")
    print(f"Strategy A Time: {time_A:.4f}s")
    print(f"Strategy B Time: {time_B:.4f}s")
    print(f"Image Saved to : {save_path}")
    print("Layout: Top Row = Standard, Bottom Row = Recursive(t=1, r=1)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MeanFlow checkpoint")
    parser.add_argument("--output_name", type=str, default="compare", help="Output filename prefix")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG Scale")
    
    args = parser.parse_args()
    sample_compare(args)