import torch
from torch.utils.data import Dataset, DataLoader
import glob
class LatentDataset(Dataset):
    def __init__(self, data_dir):
        self.latent_paths = glob.glob(f"{data_dir}/*.pt")

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        latent = torch.load(self.latent_paths[idx])
        # Latent: [C, H, W], Label: [1]
        return latent['latent'], latent['label']

if __name__ == "__main":
    # Test
    dataset = LatentDataset("../data/latents")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for latent, label in dataloader:
        print(latent.shape, label.shape)
        break
