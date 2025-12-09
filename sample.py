import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision.utils import make_grid
from PIL import Image
import os
import argparse
from models.dit import DiT_MeanFlow

# ImageNette 的 10 个类别映射
ID2LABEL = {
    0: 'Tench (鱼)', 1: 'English Springer (狗)', 2: 'Cassette Player (录音机)', 
    3: 'Chain Saw (电锯)', 4: 'Church (教堂)', 5: 'French Horn (圆号)', 
    6: 'Garbage Truck (垃圾车)', 7: 'Gas Pump (加油泵)', 8: 'Golf Ball (高尔夫)', 9: 'Parachute (降落伞)'
}

def clean_state_dict(state_dict):
    """移除 torch.compile 产生的 _orig_mod 前缀，确保权重键名匹配"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    
    print(f"Loading model from {args.ckpt}...")
    
    # 1. 初始化模型结构 (必须与训练时一致)
    model = DiT_MeanFlow(num_classes=10).to(device)
    
    # 2. 加载权重
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
        # 兼容性处理：如果 checkpoint 保存的是整个 model 而不是 state_dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
            
        model.load_state_dict(clean_state_dict(state_dict))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    
    # 3. Load VAE (用于解码 Latent -> Pixel)
    # 第一次运行会自动下载，如果已缓存则直接加载
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # 4. 准备采样条件
    # 生成 10 张图，对应 ImageNette 的 10 个类别
    num_samples = 10
    
    # 初始噪声 (t=1)
    z1 = torch.randn(num_samples, 4, 32, 32, device=device) 
    
    # 类别条件 (0, 1, 2, ... 9)
    y = torch.arange(10, device=device)
    
    # 时间条件: MeanFlow 从 t=1 (噪声) 走到 r=0 (数据)
    t = torch.ones(num_samples, device=device)   # Start time = 1.0
    r = torch.zeros(num_samples, device=device)  # Target time = 0.0

    print("Sampling (1-NFE)...")
    with torch.no_grad():
        # --- 核心采样公式 ---
        # MeanFlow 预测的是平均速度 u
        # 轨迹方程: z_r = z_t - (t - r) * u
        # 这里: z_0 = z_1 - (1.0 - 0.0) * u = z_1 - u
        
        # 预测速度场
        u = model(z1, t, r, y)
        
        # 一步更新 (Euler step)
        z0 = z1 - u
        
        # VAE 解码 (Latent -> Pixel)
        # scaling factor 来自 SD 官方标准
        z0 = z0 / 0.18215 
        samples = vae.decode(z0).sample

    # 5. 后处理与保存
    # Denormalize: [-1, 1] -> [0, 1]
    samples = (samples / 2 + 0.5).clamp(0, 1)
    
    # 拼图: 5列 2行
    grid = make_grid(samples, nrow=5, padding=2)
    
    # 转为 PIL Image 并保存
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    save_path = f"results/{args.output_name}.png"
    im.save(save_path)
    print(f"Done! Result saved to: {save_path}")
    print("Classes in order: ", [ID2LABEL[i] for i in range(10)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--output_name", type=str, default="sample_result", help="Output filename (without extension)")
    args = parser.parse_args()
    sample(args)
