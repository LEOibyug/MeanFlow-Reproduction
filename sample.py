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
    """移除 torch.compile 产生的 _orig_mod 前缀"""
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
    
    # 1. 初始化模型 (ImageNette=10类)
    model = DiT_MeanFlow(num_classes=10).to(device)
    
    # 2. 加载权重
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
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
    
    # 3. Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # 4. 准备采样条件
    num_classes = 10
    # 每个类别生成一张图
    y = torch.arange(num_classes, device=device)
    num_samples = len(y)
    
    # 初始噪声 (t=1)
    z1 = torch.randn(num_samples, 4, 32, 32, device=device) 
    
    # 时间条件: MeanFlow 从 t=1 (噪声) 到 r=0 (数据)
    t = torch.ones(num_samples, device=device)
    r = torch.zeros(num_samples, device=device)

    print(f"Sampling (1-NFE) with CFG Scale = {args.cfg_scale}...")
    
    with torch.no_grad():
        if args.cfg_scale > 1.0:
            # --- CFG 采样逻辑 ---
            # 1. 构造双倍 Batch: [有条件, 无条件]
            z_in = torch.cat([z1, z1], dim=0)
            t_in = torch.cat([t, t], dim=0)
            r_in = torch.cat([r, r], dim=0)
            
            # y_uncond 设为 10 (Null Token)
            y_uncond = torch.ones_like(y) * num_classes 
            y_in = torch.cat([y, y_uncond], dim=0)
            
            # 2. 一次前向传播
            model_output = model(z_in, t_in, r_in, y_in)
            
            # 3. 拆分输出
            u_cond, u_uncond = model_output.chunk(2, dim=0)
            
            # 4. CFG 引导公式: uncond + scale * (cond - uncond)
            u = u_uncond + args.cfg_scale * (u_cond - u_uncond)
        else:
            # 标准采样 (无 CFG)
            u = model(z1, t, r, y)
        
        # MeanFlow 核心公式: z0 = z1 - u
        z0 = z1 - u
        
        # VAE 解码
        z0 = z0 / 0.18215 
        samples = vae.decode(z0).sample

    # 5. 后处理与保存
    samples = (samples / 2 + 0.5).clamp(0, 1)
    grid = make_grid(samples, nrow=5, padding=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    
    save_path = f"results/{args.output_name}_cfg{args.cfg_scale}.png"
    im.save(save_path)
    print(f"Done! Result saved to: {save_path}")
    print("Classes: ", [ID2LABEL[i] for i in range(10)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output_name", type=str, default="sample_result", help="Output filename prefix")
    
    # 新增 CFG 参数
    parser.add_argument("--cfg_scale", type=float, default=4.0, 
                        help="Classifier-Free Guidance scale. Default=4.0. Set 1.0 to disable.")
    
    args = parser.parse_args()
    sample(args)