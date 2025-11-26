import datasets
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CIFARDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform_config=None):
        self.dataset = dataset
        if transform_config is None:
            self.transform = None
        else:
            interp = {
                "bilinear": transforms.InterpolationMode.BILINEAR,
                "bicubic": transforms.InterpolationMode.BICUBIC,
            }[transform_config["interpolation"]]
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        transform_config["input_size"][1:],
                        interpolation=interp,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=transform_config["mean"], std=transform_config["std"]
                    ),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["img"]
        coarse_label = torch.tensor(sample["coarse_label"], dtype=torch.long)
        fine_label = torch.tensor(sample["fine_label"], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, coarse_label, fine_label
