"""Score every saved checkpoint once on the fixed test split."""

import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES, WasteDataset
from src.models import NUM_CLASSES, SmallCNN, set_model_mode
from src.train import macro_f1

RESULTS = ROOT / "results"
SCORE_FIELDS = [
    "model",
    "seed",
    "images_per_class",
    "augment",
    "best_epoch",
    "val_macro_f1",
    "test_macro_f1",
    *[f"f1_{label.lower().replace(' ', '_')}" for label in CLASSES],
]


def per_class_f1(predictions, targets):
    scores = []
    for class_index in range(NUM_CLASSES):
        true_positive = ((predictions == class_index) & (targets == class_index)).sum().item()
        predicted = (predictions == class_index).sum().item()
        actual = (targets == class_index).sum().item()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return scores


def confusion_counts(predictions, targets):
    matrix = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target][prediction] += 1
    return matrix


def build_from_checkpoint(checkpoint):
    name = checkpoint["model"]
    if name == "cnn":
        model = SmallCNN()
    elif name == "resnet":
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model {name}.")
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict(model, name, loader, device):
    set_model_mode(model, name, False)
    predictions = []
    targets = []
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(predictions), torch.cat(targets)


def mean_std(values):
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, variance**0.5


def write_summary(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row["images_per_class"], row["augment"])
        groups.setdefault(key, []).append(row)

    summary = []
    f1_fields = [field for field in SCORE_FIELDS if field.startswith("f1_")]
    for key in sorted(groups, key=lambda item: (item[1], item[0], item[2])):
        group = groups[key]
        mean, std = mean_std([row["test_macro_f1"] for row in group])
        record = {
            "model": key[0],
            "images_per_class": key[1],
            "augment": key[2],
            "n_seeds": len(group),
            "mean_test_macro_f1": mean,
            "std_test_macro_f1": std,
        }
        for field in f1_fields:
            class_mean, class_std = mean_std([row[field] for row in group])
            record[f"mean_{field}"] = class_mean
            record[f"std_{field}"] = class_std
        summary.append(record)

    path = RESULTS / "test_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main():
    checkpoints = sorted(RESULTS.glob("*.pt"))
    if len(checkpoints) != 36:
        raise SystemExit(f"Expected 36 checkpoints, found {len(checkpoints)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = WasteDataset("test")
    loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device {device_name} test {len(test_set)}", flush=True)

    rows = []
    matrices = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["classes"] != CLASSES:
            raise SystemExit(f"{path.name} was trained on a different class list.")
        model = build_from_checkpoint(checkpoint).to(device)
        predictions, targets = predict(model, checkpoint["model"], loader, device)
        class_scores = per_class_f1(predictions, targets)
        score = macro_f1(predictions, targets)
        if abs(score - sum(class_scores) / NUM_CLASSES) > 1e-12:
            raise SystemExit(f"Macro-F1 mismatch for {path.name}.")
        row = {
            "model": checkpoint["model"],
            "seed": checkpoint["seed"],
            "images_per_class": checkpoint["images_per_class"],
            "augment": checkpoint["augment"],
            "best_epoch": checkpoint["best_epoch"],
            "val_macro_f1": checkpoint["best_val_macro_f1"],
            "test_macro_f1": score,
        }
        for label, class_score in zip(CLASSES, class_scores):
            row[f"f1_{label.lower().replace(' ', '_')}"] = class_score
        rows.append(row)
        matrices.append(
            {
                "file": path.name,
                "model": row["model"],
                "seed": row["seed"],
                "images_per_class": row["images_per_class"],
                "augment": row["augment"],
                "matrix": confusion_counts(predictions, targets),
            }
        )
        print(
            f"{path.name} test_macro_f1 {score:.4f} val_macro_f1 {row['val_macro_f1']:.4f}",
            flush=True,
        )
        del model

    rows.sort(key=lambda row: (row["images_per_class"], row["seed"], row["model"], row["augment"]))
    with (RESULTS / "test_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "test_confusion.json").write_text(json.dumps(matrices), encoding="utf-8")

    print("mean test macro-F1 over seeds 0, 1, 2", flush=True)
    for record in write_summary(rows):
        augment = "aug" if record["augment"] else "noaug"
        print(
            f"{record['model']} n{record['images_per_class']} {augment} "
            f"{record['mean_test_macro_f1']:.4f} +/- {record['std_test_macro_f1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
