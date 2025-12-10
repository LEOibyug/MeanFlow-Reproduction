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

def sample_meanflow_multistep(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    
    print(f"Loading MeanFlow model from {args.ckpt}...")
    model = DiT_MeanFlow(num_classes=10).to(device)
    
    # 加载权重
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

    # --- 准备采样条件 ---
    # 固定生成 "English Springer" (狗, ID=1)
    target_class = 8 
    class_name = "English Springer"
    
    y = torch.tensor([target_class], device=device)
    z = torch.randn(1, 4, 32, 32, device=device) # 初始噪声 (t=1.0)
    
    # 记录轨迹：trajectory[0] 是初始噪声
    trajectory = [z.clone()]
    
    # --- MeanFlow 多步推理设置 ---
    steps = args.steps
    # 生成时间点序列: [1.0, ..., 0.0]
    # 例如 2步: [1.0, 0.5, 0.0]
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    print(f"Running MeanFlow Multi-Step Sampling ({steps} steps)...")
    
    # --- 计时开始 ---
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(steps), desc="Sampling"):
            # MeanFlow 的核心差异：
            # 当前时刻 t
            t_curr = timesteps[i]
            # 目标时刻 r (即下一个时间步)
            t_next = timesteps[i+1]
            
            # 构造输入
            t_input = torch.ones(1, device=device) * t_curr
            r_input = torch.ones(1, device=device) * t_next # 告诉模型：我要去 t_next
            
            # --- CFG 预测平均速度 u ---
            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t_input, t_input], dim=0)
                r_in = torch.cat([r_input, r_input], dim=0)
                y_uncond = torch.tensor([10], device=device) # Null Token
                y_in = torch.cat([y, y_uncond], dim=0)
                
                u_out = model(z_in, t_in, r_in, y_in)
                u_cond, u_uncond = u_out.chunk(2, dim=0)
                u = u_uncond + args.cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z, t_input, r_input, y)
            
            # --- 步进更新 ---
            # MeanFlow 公式: z_r = z_t - (t - r) * u
            dt = t_curr - t_next
            z = z - u * dt
            
            trajectory.append(z.clone())

    # --- 计时结束 ---
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    inference_time = end_time - start_time

    # --- 智能抽帧逻辑 (10帧) ---
    total_states = len(trajectory) 
    num_frames = 10 
    
    if total_states <= num_frames:
        selected_indices = list(range(total_states))
    else:
        # 均匀选取 10 帧 (含首尾)
        selected_indices = np.linspace(0, total_states - 1, num_frames, dtype=int)
        
    print(f"Visualizing steps: {selected_indices}")
    
    # --- 批量解码 ---
    viz_latents = torch.cat([trajectory[i] for i in selected_indices], dim=0)
    
    with torch.no_grad():
        viz_latents = viz_latents / 0.18215
        images = []
        for j in range(viz_latents.shape[0]):
            img = vae.decode(viz_latents[j:j+1]).sample
            images.append(img)
        images = torch.cat(images, dim=0)

    # --- 后处理与保存 ---
    images = (images / 2 + 0.5).clamp(0, 1)
    
    # 强制单行显示: nrow = 图片数量
    grid = make_grid(images, nrow=len(images), padding=2, pad_value=1.0) 
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    save_path = f"results/{args.output_name}_meanflow_{steps}steps_process.png"
    im.save(save_path)
    
    print("\n" + "="*40)
    print(f"MeanFlow Inference Complete!")
    print(f"Steps          : {steps}")
    print(f"Total Time     : {inference_time:.4f} seconds")
    print(f"Avg Time/Step  : {(inference_time/steps)*1000:.2f} ms")
    print(f"Visualization  : {save_path}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MeanFlow checkpoint")
    parser.add_argument("--output_name", type=str, default="mf_viz", help="Output filename prefix")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps (e.g. 2, 5, 10)")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG Scale")
    
    args = parser.parse_args()
    sample_meanflow_multistep(args)