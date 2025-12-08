import argparse
import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from tqdm import tqdm

def main(args):
    # 1. Setup VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)

    # 2. Setup Dataset
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Process Images
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Processing images")):
        images = images.to(args.device)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample().mul_(0.18215)
        
        for j in range(latents.shape[0]):
            idx = i * args.batch_size + j
            filename = os.path.join(args.output_dir, f"latent_{idx:05d}.pt")
            torch.save({'latent': latents[j].cpu(), 'label': labels[j].cpu()}, filename)

    print(f"Finished processing {len(dataset)} images. Latents saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the original image dataset (e.g., imagenette2)")
    parser.add_argument("--output_dir", type=str, default="data/latents", help="Directory to save VAE latents")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
