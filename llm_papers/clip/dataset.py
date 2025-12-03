import math
from pathlib import Path
from typing import Iterator

from PIL import Image
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import trange
from transformers import PreTrainedTokenizerBase

from llm_papers.utils import project_dir


def get_cifar100_path():
    return project_dir / "clip" / "data" / "cifar100_test.pt"


def load_cifar100(
    transform=None, tokenizer: PreTrainedTokenizerBase = None, load_if_possible=True
):
    path = get_cifar100_path()
    if path.exists() and load_if_possible:
        return CIFARDataset()
    test_dataset = datasets.load_dataset("uoft-cs/cifar100", split="test")
    test_dataset = CIFARDataset(
        test_dataset, transform, tokenizer, load_if_possible=load_if_possible
    )
    return test_dataset


def preprocess_cifar100(transform, tokenizer, save=False, overwrite=False):
    save_path = get_cifar100_path()
    if save_path.exists() and not overwrite:
        if save:
            raise FileExistsError
        return torch.load(save_path)
    dataset = datasets.load_dataset("uoft-cs/cifar100", split="test")
    label_names = dataset.features["fine_label"].names
    captions = [f"a photo of a {label}" for label in label_names]
    result = tokenizer(
        captions,
        return_tensors="pt",
        truncation=True,
        padding=True,
        pad_to_multiple_of=8,
    )
    result = dict(result)
    result["label_names"] = label_names
    result["labels"] = torch.tensor(dataset["fine_label"])

    pixel_values = []
    batch_size = 100
    for i in trange(math.ceil(len(dataset) / batch_size)):
        values = transform(dataset["img"][i * batch_size : (i + 1) * batch_size])
        pixel_values.append(values["pixel_values"].to(torch.float16))
    result["pixel_values"] = torch.cat(pixel_values, dim=0)
    if save:
        torch.save(result, save_path)
        del result
    else:
        return result


def get_coco2017_path():
    return project_dir / "clip" / "data" / "coco2017_train.pt"


def load_coco2017(
    transform=None, tokenizer: PreTrainedTokenizerBase = None, image_path=None
):
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
    def __init__(
        self,
        dataset: datasets.Dataset = None,
        transform=None,
        tokenizer=None,
        load_if_possible=True,
    ):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        load_preprocessed = False
        if load_if_possible:
            path = get_cifar100_path()
            load_preprocessed = path.exists()
        self.load_preprocessed = load_preprocessed
        if self.load_preprocessed:
            data = torch.load(path)
            self.label_names = data["label_names"]
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            self.pixel_values = data["pixel_values"]
            self.labels = data["labels"]
        else:
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
        if self.load_preprocessed:
            return len(self.labels)
        return len(self.dataset)

    def __getitem__(self, idx):
        if np.iterable(idx) and not isinstance(idx[0], int):
            assert len(idx) == 1
            idx = idx[0]
        if self.load_preprocessed:
            data = {
                "pixel_values": self.pixel_values[idx],
                "labels": self.labels[idx],
                "input_ids": torch.tensor([0, 2]),
                "attention_mask": torch.tensor([1, 1]),
            }
        else:
            sample = self.dataset[idx]
            pixel_values = sample["img"]
            label = torch.tensor(sample["fine_label"], dtype=torch.long)

            if self.transform is not None:
                pixel_values = self.transform(pixel_values)["pixel_values"]
            if isinstance(idx, int):
                pixel_values = pixel_values[0]
            data = {"pixel_values": pixel_values, "labels": label}
        # placeholder
        data["input_ids"] = torch.tensor([0, 2])
        data["attention_mask"] = torch.tensor([1, 1])
        return data


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
