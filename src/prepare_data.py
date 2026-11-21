"""Fixed stratified split and nested training subsets for five RealWaste classes."""

import csv
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "splits"

CLASSES = ["Food Organics", "Glass", "Metal", "Paper", "Plastic"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_SEED = 42
SUBSET_SEEDS = (0, 1, 2)
SUBSET_SIZES = (10, 50, 200)


def list_images(label):
    folder = RAW / label
    if not folder.is_dir():
        raise SystemExit(f"Missing class folder: {folder}")
    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise SystemExit(f"No images in {folder}")
    return sorted(paths, key=lambda path: path.name)


def relative(path):
    return path.relative_to(PROJECT).as_posix()


def split_counts(n):
    n_test = int(round(n * 0.20))
    n_val = int(round(n * 0.10))
    n_train = n - n_test - n_val
    if n_train < SUBSET_SIZES[-1]:
        raise SystemExit(
            f"Training pool has {n_train} images, fewer than {SUBSET_SIZES[-1]}."
        )
    return n_train, n_val, n_test


def split_class(paths):
    n_train, n_val, n_test = split_counts(len(paths))
    order = paths.copy()
    np.random.default_rng(SPLIT_SEED).shuffle(order)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    if not (len(train) == n_train and len(val) == n_val and len(test) == n_test):
        raise SystemExit("Split sizes do not add up.")
    return train, val, test


def nested_subsets(train_by_label):
    subsets = {}
    for seed in SUBSET_SEEDS:
        rng = np.random.default_rng(seed)
        subsets[str(seed)] = {str(size): {} for size in SUBSET_SIZES}
        for label in CLASSES:
            paths = sorted(train_by_label[label], key=lambda path: path.name)
            ordered = [paths[i] for i in rng.permutation(len(paths))]
            for size in SUBSET_SIZES:
                subsets[str(seed)][str(size)][label] = [
                    relative(path) for path in ordered[:size]
                ]
    return subsets


def check(rows, subsets):
    by_path = {}
    counts = {label: {"train": 0, "val": 0, "test": 0} for label in CLASSES}
    for row in rows:
        if row["path"] in by_path:
            raise SystemExit(f"Duplicate path in split: {row['path']}")
        by_path[row["path"]] = row
        counts[row["label"]][row["split"]] += 1

    for seed, by_size in subsets.items():
        for label in CLASSES:
            small = set(by_size["10"][label])
            middle = set(by_size["50"][label])
            large = set(by_size["200"][label])
            if not (len(small) == 10 and len(middle) == 50 and len(large) == 200):
                raise SystemExit(f"Bad subset size for seed {seed}, {label}")
            if not small <= middle <= large:
                raise SystemExit(f"Subsets are not nested for seed {seed}, {label}")
            for path in large:
                row = by_path.get(path)
                if row is None or row["split"] != "train" or row["label"] != label:
                    raise SystemExit(f"Subset path is outside the training pool: {path}")
    return counts


def main():
    rows = []
    train_by_label = {}
    for label in CLASSES:
        train, val, test = split_class(list_images(label))
        train_by_label[label] = train
        for split_name, paths in ("train", train), ("val", val), ("test", test):
            rows.extend(
                {"path": relative(path), "label": label, "split": split_name}
                for path in paths
            )

    subsets = nested_subsets(train_by_label)
    counts = check(rows, subsets)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "split_seed": SPLIT_SEED,
        "subset_seeds": list(SUBSET_SEEDS),
        "classes": CLASSES,
        "sizes": list(SUBSET_SIZES),
        "numpy": np.__version__,
        "seeds": subsets,
    }
    (OUT / "subsets.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"{'class':<16} {'train':>6} {'val':>6} {'test':>6} {'total':>6}")
    for label in CLASSES:
        item = counts[label]
        total = item["train"] + item["val"] + item["test"]
        print(
            f"{label:<16} {item['train']:>6} {item['val']:>6} {item['test']:>6} {total:>6}"
        )
    print(f"rows {len(rows)}")
    print("nested subsets: seeds 0, 1, 2; per class 10 inside 50 inside 200; all paths are train")


if __name__ == "__main__":
    main()
