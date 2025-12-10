import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
from PIL import Image
import os
import argparse
from tqdm import tqdm
from models.dit import DiT_MeanFlow

# ImageNette 类别映射
ID2LABEL = {
    0: 'Tench (鱼)', 1: 'English Springer (狗)', 2: 'Cassette Player (录音机)', 
    3: 'Chain Saw (电锯)', 4: 'Church (教堂)', 5: 'French Horn (圆号)', 
    6: 'Garbage Truck (垃圾车)', 7: 'Gas Pump (加油泵)', 8: 'Golf Ball (高尔夫)', 9: 'Parachute (降落伞)'
}

def clean_state_dict(state_dict):
    """移除 torch.compile 产生的 _orig_mod 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def sample_multistep(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    
    print(f"Loading Baseline model from {args.ckpt}...")
    
    # 1. 初始化模型
    model = DiT_MeanFlow(num_classes=10).to(device)
    
    # 2. 加载权重
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(clean_state_dict(state_dict))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    
    # 3. Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # 4. 准备采样条件
    num_classes = 10
    # 生成 10 张图，对应 10 个类别
    y = torch.arange(num_classes, device=device)
    num_samples = len(y)
    
    # 初始噪声 z1 (t=1.0)
    z = torch.randn(num_samples, 4, 32, 32, device=device) 
    
    # 5. 多步 Euler 采样循环
    steps = args.steps
    # 生成时间序列: [1.0, ..., 0.0]
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    dt = 1.0 / steps
    
    print(f"Running Multi-Step Sampling (Steps={steps}, CFG={args.cfg_scale})...")
    
    with torch.no_grad():
        for i in tqdm(range(steps)):
            # 当前时间 t
            t_curr = timesteps[i]
            
            # 构造 Batch 时间输入
            t_input = torch.ones(num_samples, device=device) * t_curr
            
            # 【关键】对于 Baseline 模型，训练时 t=r，所以这里 r 也设为 t
            r_input = t_input 
            
            # --- CFG 预测速度场 v ---
            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t_input, t_input], dim=0)
                r_in = torch.cat([r_input, r_input], dim=0)
                
                y_uncond = torch.ones_like(y) * num_classes # Null Token
                y_in = torch.cat([y, y_uncond], dim=0)
                
                # 前向传播
                v_output = model(z_in, t_in, r_in, y_in)
                v_cond, v_uncond = v_output.chunk(2, dim=0)
                
                # CFG 组合
                v = v_uncond + args.cfg_scale * (v_cond - v_uncond)
            else:
                v = model(z, t_input, r_input, y)
            
            # --- Euler 更新步 ---
            # dZ_t = v_t * dt
            # 我们是从 t=1 走到 t=0，所以是减去 v * dt
            z = z - v * dt

        # 6. VAE 解码
        z = z / 0.18215 
        samples = vae.decode(z).sample

    # 7. 保存结果
    samples = (samples / 2 + 0.5).clamp(0, 1)
    grid = make_grid(samples, nrow=5, padding=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    save_path = f"results/{args.output_name}_baseline_{steps}steps.png"
    im.save(save_path)
    print(f"\nDone! Saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Baseline checkpoint")
    parser.add_argument("--output_name", type=str, default="sample", help="Output filename prefix")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps (e.g. 20, 50, 100)")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG Scale (Try 4.0 for better quality)")
    
    args = parser.parse_args()
    sample_multistep(args)