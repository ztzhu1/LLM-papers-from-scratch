from pathlib import Path

from PIL import Image
import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from llm_papers.utils import project_dir


def load_cifar100(transform, tokenizer: PreTrainedTokenizerBase):
    test_dataset = datasets.load_dataset("uoft-cs/cifar100", split="test")
    test_dataset = CIFARDataset(test_dataset, transform, tokenizer)
    return test_dataset


def load_coco2017(transform, tokenizer: PreTrainedTokenizerBase, image_path=None):
    train_dataset = datasets.load_dataset("phiyodr/coco2017", split="train")
    if image_path is None:
        image_path = project_dir / "clip" / "data" / "coco2017"
    image_path = Path(image_path)

    def create_full_path(example):
        example["image_path"] = (image_path / example["file_name"]).as_posix()
        return example

    train_dataset = train_dataset.map(create_full_path, num_proc=4)
    train_dataset = COCODataset(train_dataset, transform, tokenizer)
    return train_dataset


class CIFARDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform, tokenizer):
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
        pixel_values = sample["img"]
        label = torch.tensor(sample["fine_label"], dtype=torch.long)

        if self.transform is not None:
            pixel_values = self.transform(pixel_values)["pixel_values"][0]
        data = {"pixel_values": pixel_values, "labels": label}
        return data


class COCODataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, transform, tokenizer):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.indexes = []
        for img_index, captions in enumerate(dataset["captions"]):
            for caption_index, caption in enumerate(captions):
                self.indexes.append((img_index, caption_index))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_idx, caption_index = self.indexes[idx]
        sample = self.dataset[img_idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        caption = sample["captions"][caption_index]

        if self.transform is not None:
            image = self.transform(image)["pixel_values"][0]
        result = {"pixel_values": image}
        if self.tokenizer is not None:
            encoding = self.tokenizer(caption, return_tensors="pt", truncation=True)
            result["input_ids"] = encoding["input_ids"][0]
            result["attention_mask"] = encoding["attention_mask"][0]
        return result
