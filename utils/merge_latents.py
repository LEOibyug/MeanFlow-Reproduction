import torch
import os
import glob
from tqdm import tqdm

def merge_data(data_dir):
    print(f"Scanning files in {data_dir}...")
    
    # 查找所有 latent_xxxxx.pt 文件
    latent_files = sorted(glob.glob(os.path.join(data_dir, "latent_*.pt")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "label_*.pt")))
    
    if not latent_files:
        print("Error: No latent_*.pt files found!")
        return

    print(f"Found {len(latent_files)} batches. Merging...")
    
    all_latents = []
    all_labels = []
    
    # 逐个读取并放入列表
    for lf, labf in tqdm(zip(latent_files, label_files), total=len(latent_files)):
        all_latents.append(torch.load(lf))
        all_labels.append(torch.load(labf))
    
    # 拼接成大 Tensor
    merged_latents = torch.cat(all_latents, dim=0)
    merged_labels = torch.cat(all_labels, dim=0)
    
    print(f"Merged shape: {merged_latents.shape}")
    
    # 保存为 train.py 期待的文件名
    torch.save(merged_latents, os.path.join(data_dir, "latents.pt"))
    torch.save(merged_labels, os.path.join(data_dir, "labels.pt"))
    
    print("Success! Saved latents.pt and labels.pt")

if __name__ == "__main__":
    merge_data("data/latents")
