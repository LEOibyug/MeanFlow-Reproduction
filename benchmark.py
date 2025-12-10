import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
import os
import argparse
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

def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on device: {device}")
    
    # --- 1. 初始化模型 ---
    print("Loading model...")
    model = DiT_MeanFlow(num_classes=10).to(device)
    
    # 加载权重 (为了模拟真实情况，虽然测速其实不需要权重)
    if os.path.exists(args.ckpt):
        try:
            ckpt = torch.load(args.ckpt, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            model.load_state_dict(clean_state_dict(state_dict))
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}). Using random weights for benchmarking.")
    else:
        print("Warning: Checkpoint not found. Using random weights for benchmarking.")

    model.eval()
    
    # --- 2. Load VAE ---
    # 如果只想测 DiT 速度，可以注释掉这部分，但为了模拟完整生成流程，建议保留
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # --- 3. 准备测试数据 ---
    batch_size = args.batch_size
    total_images = args.num_samples
    num_batches = (total_images + batch_size - 1) // batch_size
    
    # 固定输入以保证一致性
    z1 = torch.randn(batch_size, 4, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    t = torch.ones(batch_size, device=device)
    r = torch.zeros(batch_size, device=device)
    
    # CFG 输入准备
    z_in = torch.cat([z1, z1], dim=0) if args.cfg_scale > 1.0 else z1
    t_in = torch.cat([t, t], dim=0) if args.cfg_scale > 1.0 else t
    r_in = torch.cat([r, r], dim=0) if args.cfg_scale > 1.0 else r
    y_uncond = torch.ones_like(y) * 10
    y_in = torch.cat([y, y_uncond], dim=0) if args.cfg_scale > 1.0 else y

    # --- 4. Warmup (预热) ---
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3): # 跑 3 次预热
            _ = model(z_in, t_in, r_in, y_in)
            
    if device == "cuda":
        torch.cuda.synchronize()

    # --- 5. 正式测试 ---
    print(f"Start benchmarking: {total_images} images (Batch Size: {batch_size})...")
    
    start_time = time.time()
    
    # 显卡精确计时器
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Inferencing"):
            # A. 模型前向 (DiT)
            if args.cfg_scale > 1.0:
                out = model(z_in, t_in, r_in, y_in)
                u_cond, u_uncond = out.chunk(2, dim=0)
                u = u_uncond + args.cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z1, t, r, y)
            
            # B. 步进更新 (1-Step)
            z0 = z1 - u
            
            # C. VAE 解码 (Latent -> Pixel)
            if not args.skip_vae:
                z0 = z0 / 0.18215 
                _ = vae.decode(z0).sample

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0
    else:
        elapsed_sec = time.time() - start_time

    # --- 6. 结果计算 ---
    fps = total_images / elapsed_sec
    latency_per_img = (elapsed_sec * 1000) / total_images # ms

    print("\n" + "="*40)
    print(f"Benchmark Results (1-NFE {'+ CFG' if args.cfg_scale > 1 else ''})")
    print("="*40)
    print(f"Total Images   : {total_images}")
    print(f"Batch Size     : {batch_size}")
    print(f"CFG Scale      : {args.cfg_scale}")
    print(f"VAE Decode     : {'Enabled' if not args.skip_vae else 'Disabled'}")
    print("-" * 40)
    print(f"Total Time     : {elapsed_sec:.4f} sec")
    print(f"Throughput     : {fps:.2f} images/sec (FPS)")
    print(f"Latency        : {latency_per_img:.2f} ms/image")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/dummy.pth", help="Path to checkpoint (optional)")
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for inference")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale (set >1 to benchmark CFG cost)")
    parser.add_argument("--skip_vae", action="store_true", help="Skip VAE decoding to test DiT pure speed")
    
    args = parser.parse_args()
    benchmark(args)