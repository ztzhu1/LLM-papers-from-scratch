import datasets
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CIFARDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["img"]
        coarse_label = torch.tensor(sample["coarse_label"], dtype=torch.long)
        fine_label = torch.tensor(sample["fine_label"], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)['pixel_values'][0]
        return {"images": image, "labels": fine_label}
