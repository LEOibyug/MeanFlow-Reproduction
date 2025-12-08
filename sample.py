# sample.py
import torch
from models.dit import DiT_MeanFlow
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
import os

def sample(exp_name, epoch, device="cuda"):
    # Load Model
    model = DiT_MeanFlow(num_classes=10).to(device)
    ckpt_path = f"checkpoints/{exp_name}_ep{epoch}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load VAE for decoding
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    
    # Setup 1-step Generation
    # Start from Noise (t=1) -> Go to Data (t=0)
    B = 4
    z1 = torch.randn(B, 4, 32, 32).to(device) # Noise
    t = torch.ones(B).to(device)  # Start time = 1
    r = torch.zeros(B).to(device) # Target time = 0
    y = torch.tensor([0, 1, 2, 9]).to(device) # Sample labels (Fish, etc.)
    
    with torch.no_grad():
        # 1-NFE Prediction
        # u is average velocity from t=1 to r=0
        # Formula: z_r = z_t - (t-r) * u
        u = model(z1, t, r, y)
        z0 = z1 - (1.0 - 0.0) * u
        
        # Decode
        z0 = z0 / 0.18215
        imgs = vae.decode(z0).sample
        
    # Save Grid
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    grid = make_grid(imgs, nrow=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(f"results/{exp_name}_sample.png")
    print(f"Saved result to results/{exp_name}_sample.png")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # 示例调用
    # sample("exp_meanflow", 99)
