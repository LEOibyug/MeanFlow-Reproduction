import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
from PIL import Image
import os
import argparse
import numpy as np
import time
from datetime import datetime
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

def sample_triangle_session(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # 0. 创建本次会话的专属文件夹
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("results", f"triangle_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    print(f"Session directory created: {session_dir}")

    # ==========================================
    # 1. 模型加载 (只执行一次)
    # ==========================================
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
    
    print("\n" + "="*50)
    print("Model loaded! Entering interactive generation loop.")
    print(f"All images will be saved to: {session_dir}")
    print("="*50)

    # ==========================================
    # 2. 交互式生成循环
    # ==========================================
    counter = 0
    
    while True:
        counter += 1
        print(f"\n>>> Generation Run #{counter}")
        
        # --- 2.1 准备初始条件 (每次重新生成噪声) ---
        target_class = 4 # English Springer -------------------------------
        y = torch.tensor([target_class], device=device)
        
        # 新的随机噪声
        z_init = torch.randn(1, 4, 32, 32, device=device)
        
        all_rows = []
        # 修改点：最大帧数改为 5 (因为去掉了噪声，5步推理正好产生 5 张非噪声图)
        max_frames = 5 
        step_counts = [5, 4, 3, 2, 1]
        
        start_time = time.time()
        
        # --- 2.2 倒三角推理循环 ---
        for steps in step_counts:
            # 重置状态
            z = z_init.clone()
            trajectory = [z.clone()]
            
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
            
            with torch.no_grad():
                for i in tqdm(range(steps), desc=f"Inferencing ({steps} steps)", leave=False):
                    t_curr = timesteps[i]
                    t_next = timesteps[i+1]
                    
                    t_input = torch.ones(1, device=device) * t_curr
                    r_input = torch.ones(1, device=device) * t_next
                    
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
                    trajectory.append(z.clone())
            
            # --- 修改点：剔除噪声帧 (trajectory[0]) ---
            # trajectory 包含 [噪声, 第一步结果, ..., 最终结果]
            # 我们取 trajectory[1:]
            vis_trajectory = trajectory[1:]
            
            # 解码当前行
            row_latents = torch.cat(vis_trajectory, dim=0)
            
            with torch.no_grad():
                row_latents = row_latents / 0.18215
                row_images = []
                for j in range(row_latents.shape[0]):
                    img = vae.decode(row_latents[j:j+1]).sample
                    row_images.append(img)
                row_images = torch.cat(row_images, dim=0)
                
            row_images = (row_images / 2 + 0.5).clamp(0, 1)
            
            # 填充空白 (白色) 以对齐网格
            pad_count = max_frames - row_images.shape[0]
            if pad_count > 0:
                white_pad = torch.ones(pad_count, 3, 256, 256, device=device)
                row_images = torch.cat([row_images, white_pad], dim=0)
                
            all_rows.append(row_images)

        # --- 2.3 保存结果 ---
        final_tensor = torch.cat(all_rows, dim=0)
        
        # padding=4 增加每张图之间的间隔，看起来更像画廊
        grid = make_grid(final_tensor, nrow=max_frames, padding=4, pad_value=1.0)
        
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        
        # 修改点：保存到 session 文件夹，命名为 runX.png
        filename = f"run{counter}.png"
        save_path = os.path.join(session_dir, filename)
        im.save(save_path)
        
        elapsed = time.time() - start_time
        print(f"Saved: {save_path} (Time: {elapsed:.2f}s)")
        
        # ==========================================
        # 3. 询问用户
        # ==========================================
        while True:
            user_input = input("\nGenerate another one? (y/n): ").strip().lower()
            if user_input == 'y':
                break
            elif user_input == 'n':
                print("Exiting...")
                return
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MeanFlow checkpoint")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG Scale")
    
    args = parser.parse_args()
    sample_triangle_session(args)