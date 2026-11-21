"""Read the fixed split and nested training subsets as 224x224 tensors."""

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ["Food Organics", "Glass", "Metal", "Paper", "Plastic"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(augment):
    steps = [transforms.Resize((224, 224))]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class WasteDataset(Dataset):
    def __init__(self, split, subset_seed=None, images_per_class=None, augment=False):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        if split != "train" and (subset_seed is not None or images_per_class is not None):
            raise ValueError("Subsets are drawn only from the training split.")
        if split == "train" and (subset_seed is None) != (images_per_class is None):
            raise ValueError("A training subset needs both subset_seed and images_per_class.")
        if images_per_class not in {None, 10, 50, 200}:
            raise ValueError("images_per_class must be 10, 50, or 200.")
        if augment and split != "train":
            raise ValueError("Augmentation is applied only to training images.")

        self.split = split
        self.augment = augment
        self.transform = build_transform(augment)
        self.samples = load_samples(split, subset_seed, images_per_class)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(PROJECT / path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def load_samples(split, subset_seed, images_per_class):
    if images_per_class is None:
        return load_split(split)
    return load_subset(subset_seed, images_per_class)


def load_split(split):
    samples = []
    with (PROJECT / "splits" / "split.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split:
                samples.append((row["path"], CLASS_TO_INDEX[row["label"]]))
    if not samples:
        raise ValueError(f"No images found for split {split}.")
    return samples


def load_subset(subset_seed, images_per_class):
    with (PROJECT / "splits" / "subsets.json").open(encoding="utf-8") as handle:
        subsets = json.load(handle)
    try:
        by_class = subsets["seeds"][str(subset_seed)][str(images_per_class)]
    except KeyError as exc:
        raise ValueError(
            f"No subset for seed {subset_seed} and {images_per_class} images per class."
        ) from exc

    samples = []
    for label in CLASSES:
        paths = by_class[label]
        if len(paths) != images_per_class:
            raise ValueError(f"{label} has {len(paths)} images, expected {images_per_class}.")
        samples.extend((path, CLASS_TO_INDEX[label]) for path in paths)
    return samples


if __name__ == "__main__":
    train_set = WasteDataset("train", subset_seed=0, images_per_class=50, augment=True)
    val_set = WasteDataset("val")
    test_set = WasteDataset("test")
    image, label = train_set[0]
    print(f"train subset {len(train_set)}")
    print(f"val {len(val_set)}")
    print(f"test {len(test_set)}")
    print(f"tensor {tuple(image.shape)} label {label} ({CLASSES[label]})")
