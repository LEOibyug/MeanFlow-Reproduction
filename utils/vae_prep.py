import argparse
import os
import torch
from torchvision import datasets, transforms
from diffusers import AutoencoderKL
from tqdm import tqdm

def prepare_latents(data_dir, output_dir, batch_size=32, device="cuda"):
    """
    读取整个数据集，编码为 Latent，保存为单个 .pt 文件。
    这种格式能被 train.py 瞬间加载进显存。
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing VAE on {device}...")
    # 使用 Stability AI 官方微调过的 MSE VAE，解码质量更好
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    # 启用切片编码以节省显存 (防止大 Batch 爆显存)
    vae.enable_slicing()

    # 标准 ImageNet 预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Map to [-1, 1]
    ])
    
    print(f"Loading images from {data_dir}...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    all_latents = []
    all_labels = []
    
    print("Start Encoding...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Encoding"):
            imgs = imgs.to(device)
            
            # VAE Encode: Image -> Latent Distribution -> Sample
            posterior = vae.encode(imgs).latent_dist
            latents = posterior.sample()
            
            # 关键缩放因子! (Stable Diffusion 标准)
            latents = latents * 0.18215
            
            # 移回 CPU 以节省显存，防止列表撑爆 GPU
            all_latents.append(latents.cpu())
            all_labels.append(labels)
            
    print("Concatenating tensors...")
    # 合并为一个大 Tensor
    final_latents = torch.cat(all_latents, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    print(f"Latents Shape: {final_latents.shape}") # (N, 4, 32, 32)
    print(f"Labels Shape: {final_labels.shape}")   # (N,)
    
    # 保存
    latent_path = os.path.join(output_dir, "latents.pt")
    label_path = os.path.join(output_dir, "labels.pt")
    
    print(f"Saving to {latent_path}...")
    torch.save(final_latents, latent_path)
    torch.save(final_labels, label_path)
    
    print("Done! Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images (e.g. data/imagenette2/train)")
    parser.add_argument("--output_dir", type=str, default="data/latents", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    prepare_latents(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )
