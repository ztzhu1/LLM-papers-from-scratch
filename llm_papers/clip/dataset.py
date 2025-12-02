from PIL import Image
import datasets
import torch
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform=None, tokenizer=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.label_names = dataset.features["fine_label"].names
        self.captions = [f"a photo of a {label}" for label in self.label_names]
        self.input_ids = None
        self.attention_mask = None
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                self.captions,
                return_tensors="pt",
                truncation=True,
                padding=True,
                pad_to_multiple_of=8,
            )
            self.input_ids = encoding["input_ids"]
            self.attention_mask = encoding["attention_mask"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["img"]
        fine_label = torch.tensor(sample["fine_label"], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)["pixel_values"][0]
        result = {"pixel_values": image, "labels": fine_label}
        if self.tokenizer is not None:
            caption = f"a photon of a {self.label_names[fine_label]}"
            encoding = self.tokenizer(caption, return_tensors="pt", truncation=True)
            result["input_ids"] = encoding["input_ids"][0]
            result["attention_mask"] = encoding["attention_mask"][0]
        return result


class COCODataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform=None, tokenizer=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        caption = sample["captions"][0]

        if self.transform is not None:
            image = self.transform(image)["pixel_values"][0]
        result = {"pixel_values": image}
        if self.tokenizer is not None:
            encoding = self.tokenizer(caption, return_tensors="pt", truncation=True)
            result["input_ids"] = encoding["input_ids"][0]
            result["attention_mask"] = encoding["attention_mask"][0]
        return result
