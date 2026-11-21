"""Report figures from the one-time test evaluation and the saved training curves."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CLASSES = ["Food Organics", "Glass", "Metal", "Paper", "Plastic"]
SHORT = ["Food", "Glass", "Metal", "Paper", "Plastic"]
SIZES = (10, 50, 200)
MODELS = ("cnn", "resnet")
MODEL_LABELS = {"cnn": "Small CNN", "resnet": "ResNet-18"}
MODEL_COLORS = {"cnn": "#0072B2", "resnet": "#D55E00"}
SEED_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}


def slug(label):
    return label.lower().replace(" ", "_")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"]) if "seed" in row else None
        row["images_per_class"] = int(row["images_per_class"])
        row["augment"] = row["augment"] == "True"
        for key, value in list(row.items()):
            if key.endswith("macro_f1") or key.startswith("f1_") or key.startswith("mean_") or key.startswith("std_"):
                row[key] = float(value)
    return rows


def style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def plot_macro_f1(summary):
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(SIZES))
    for model in MODELS:
        for augment, marker, linestyle in ((False, "o", "-"), (True, "s", "--")):
            means = []
            stds = []
            for size in SIZES:
                match = next(
                    row
                    for row in summary
                    if row["model"] == model
                    and row["images_per_class"] == size
                    and row["augment"] is augment
                )
                means.append(match["mean_test_macro_f1"])
                stds.append(match["std_test_macro_f1"])
            label = f"{MODEL_LABELS[model]}, {'augmentation' if augment else 'no augmentation'}"
            axis.errorbar(
                positions,
                means,
                yerr=stds,
                color=MODEL_COLORS[model],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                capsize=3,
                label=label,
            )
    axis.set_xticks(positions, [str(size) for size in SIZES])
    axis.set_xlabel("Training images per class")
    axis.set_ylabel("Test Macro-F1")
    axis.set_ylim(0.2, 1.0)
    axis.grid(axis="y", linestyle=":", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    figure.savefig(FIGURES / "test_macro_f1.png")
    plt.close(figure)


def plot_per_class(summary):
    chosen = [
        row
        for row in summary
        if row["images_per_class"] == 200 and row["augment"] is False
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    offsets = {"cnn": -0.18, "resnet": 0.18}
    x = np.arange(len(CLASSES))
    for row in chosen:
        means = [row[f"mean_f1_{slug(label)}"] for label in CLASSES]
        stds = [row[f"std_f1_{slug(label)}"] for label in CLASSES]
        axis.bar(
            x + offsets[row["model"]],
            means,
            width=0.36,
            yerr=stds,
            capsize=3,
            color=MODEL_COLORS[row["model"]],
            label=MODEL_LABELS[row["model"]],
        )
    axis.set_xticks(x, SHORT)
    axis.set_xlabel("Class")
    axis.set_ylabel("Test F1")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", linestyle=":", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "per_class_f1.png")
    plt.close(figure)


def average_confusion(records, model):
    chosen = [
        np.array(record["matrix"], dtype=float)
        for record in records
        if record["model"] == model
        and record["images_per_class"] == 200
        and record["augment"] is False
    ]
    if len(chosen) != 3:
        raise SystemExit(f"Expected 3 confusion matrices for {model}, found {len(chosen)}.")
    counts = np.sum(chosen, axis=0)
    return counts / counts.sum(axis=1, keepdims=True)


def plot_confusion(records):
    figure = plt.figure(figsize=(10.6, 4.6))
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.32)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    color_axis = figure.add_subplot(grid[0, 2])
    image = None
    for axis, model in zip(axes, MODELS):
        matrix = average_confusion(records, model)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(len(SHORT)), SHORT, rotation=30, ha="right")
        axis.set_yticks(range(len(SHORT)), SHORT)
        axis.set_xlabel("Predicted")
        axis.set_title(MODEL_LABELS[model])
        for row in range(len(CLASSES)):
            for column in range(len(CLASSES)):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=9,
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("True")
    figure.colorbar(image, cax=color_axis)
    figure.savefig(FIGURES / "confusion_matrices.png")
    plt.close(figure)


def plot_curves():
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for axis, size in zip(axes, (10, 200)):
        for seed in (0, 1, 2):
            path = RESULTS / f"cnn_seed{seed}_n{size}_noaug.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            epochs = [row["epoch"] for row in checkpoint["history"]]
            scores = [row["val_macro_f1"] for row in checkpoint["history"]]
            axis.plot(epochs, scores, color=SEED_COLORS[seed], linewidth=1.6, label=f"Seed {seed}")
        axis.set_xlim(1, 40)
        axis.set_xlabel("Epoch")
        axis.set_title(f"{size} images per class")
        axis.grid(axis="y", linestyle=":", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Validation Macro-F1")
    axes[0].set_ylim(0, 1.0)
    axes[1].legend(frameon=False, loc="lower right")
    figure.savefig(FIGURES / "cnn_val_curves.png")
    plt.close(figure)


def plot_split_counts():
    counts = {label: {"train": 0, "val": 0, "test": 0} for label in CLASSES}
    with (ROOT / "splits" / "split.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["label"]][row["split"]] += 1

    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(CLASSES))
    width = 0.24
    colors = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
    for offset, split in enumerate(("train", "val", "test")):
        values = [counts[label][split] for label in CLASSES]
        axis.bar(x + (offset - 1) * width, values, width=width, color=colors[split], label=split)
    axis.set_xticks(x, SHORT)
    axis.set_xlabel("Class")
    axis.set_ylabel("Images")
    axis.grid(axis="y", linestyle=":", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.savefig(FIGURES / "split_counts.png")
    plt.close(figure)


def plot_samples():
    by_class = {label: [] for label in CLASSES}
    with (ROOT / "splits" / "split.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                by_class[row["label"]].append(row["path"])

    figure, axes = plt.subplots(len(CLASSES), 4, figsize=(8.2, 10.2))
    for row_index, label in enumerate(CLASSES):
        paths = sorted(by_class[label])
        indexes = [round(i * (len(paths) - 1) / 3) for i in range(4)]
        for column, index in enumerate(indexes):
            axis = axes[row_index, column]
            with Image.open(ROOT / paths[index]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row_index, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=28)
    figure.subplots_adjust(wspace=0.08, hspace=0.08, left=0.22)
    figure.savefig(FIGURES / "sample_images.png")
    plt.close(figure)


def main():
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = read_csv(RESULTS / "test_summary.csv")
    records = json.loads((RESULTS / "test_confusion.json").read_text(encoding="utf-8"))
    plot_macro_f1(summary)
    plot_per_class(summary)
    plot_confusion(records)
    plot_curves()
    plot_split_counts()
    plot_samples()
    for path in sorted(FIGURES.glob("*.png")):
        print(path.name, flush=True)


if __name__ == "__main__":
    main()
