"""Chinese-CLIP-256 Weibo and overlapping FineLIP-248 full English articles."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from clip_encoders import tokenize_article, language_for, MEAN, STD


def text_value(row) -> str:
    value = row.get("text", row.get("content", ""))
    return "" if pd.isna(value) else str(value)


class ViRTADataset(Dataset):
    def __init__(self, csv_path: str, dataset_name: str, split: str, train: bool = False):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        self.dataset_name, self.split = dataset_name, split
        self.language = language_for(dataset_name)
        self.max_length = 256 if self.language == "zh" else 248
        self.train = train
        required = {"label"}
        if not ({"text", "content"} & set(self.frame.columns)) or not ({"image", "images"} & set(self.frame.columns)):
            raise ValueError(f"{csv_path} needs text/content and image/images columns")
        if not required <= set(self.frame.columns):
            raise ValueError(f"{csv_path} needs a label column")
        if train:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224) if self.language == "zh" else 224,
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self): return len(self.frame)

    def _news_id(self, index: int) -> str:
        row = self.frame.iloc[index]
        value = row.get("news_id", row.get("id", row.get("Unnamed: 0", index)))
        return f"{self.dataset_name}_{self.split}_{value}"

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = Path(str(row.get("image", row.get("images", ""))).split("|")[0])
        if not image_path.is_absolute():
            if self.dataset_name == "gossipcop" and len(image_path.parts) == 1:
                image_path = self.csv_path.parent / "Images" / f"gossip_{self.split}" / image_path.name
            else:
                image_path = self.csv_path.parent / image_path
        try:
            with Image.open(image_path) as source:
                if source.mode == "P" and "transparency" in source.info:
                    source = source.convert("RGBA")
                image = source.convert("RGB")
        except (OSError, FileNotFoundError) as exc:
            raise OSError(f"Cannot load image for {self._news_id(index)}: {image_path}") from exc
        result = {"image": self.image_transform(image),
                  "label": torch.tensor(int(row["label"])), "news_id": self._news_id(index)}
        result.update(tokenize_article(text_value(row), self.language))
        return result


def collate_fn(rows):
    batch = {"image": torch.stack([row["image"] for row in rows]),
             "label": torch.stack([row["label"] for row in rows]),
             "news_id": [row["news_id"] for row in rows],
             "token_count": torch.stack([row["token_count"] for row in rows])}
    chunks = max(len(row['input_ids']) for row in rows)
    length = rows[0]['input_ids'].shape[-1]
    for key in ('input_ids','attention_mask','token_type_ids'):
        value = rows[0][key]
        batch[key] = value.new_zeros(len(rows),chunks,length)
        for i,row in enumerate(rows):
            batch[key][i,:len(row[key])] = row[key]
    batch['chunk_mask'] = batch['attention_mask'].any(-1)
    batch['chunk_spans'] = torch.full((len(rows),chunks,2),-1,dtype=torch.long)
    for i,row in enumerate(rows):
        batch['chunk_spans'][i,:len(row['chunk_spans'])] = row['chunk_spans']
    return batch


DATA_ROOT_KEYS = {"weibo": "WEIBO_DATA_ROOT", "gossipcop": "GOSSIP_DATA_ROOT", "twitter": "TWITTER_DATA_ROOT"}


def configure_data_root(dataset, path):
    if path is not None:
        os.environ[DATA_ROOT_KEYS[dataset]] = str(Path(path).expanduser().resolve())


def data_root(dataset):
    folder = "gossip" if dataset == "gossipcop" else dataset
    default = Path(__file__).resolve().parent / "data" / folder / "splits"
    return Path(os.environ.get(DATA_ROOT_KEYS[dataset], default)).expanduser()


def has_explicit_validation(dataset: str) -> bool:
    filename = "val.csv" if dataset == "weibo" else f"{'gossip' if dataset == 'gossipcop' else dataset}_val.csv"
    return (data_root(dataset) / filename).is_file()


def build_dataset(dataset: str, split: str, train: bool):
    if dataset not in DATA_ROOT_KEYS:
        raise ValueError("dataset must be weibo, gossipcop, or twitter")
    filename = f"{split}.csv" if dataset == "weibo" else f"{'gossip' if dataset == 'gossipcop' else dataset}_{split}.csv"
    return ViRTADataset(str(data_root(dataset) / filename), dataset, split, train=train)
