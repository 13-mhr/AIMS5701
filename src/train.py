"""Train the small CNN or the frozen ResNet-18 head. The test set is not used."""

import argparse
import copy
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES, WasteDataset
from src.models import NUM_CLASSES, build_model, set_model_mode, trainable_parameters

LEARNING_RATES = {"cnn": 1e-3, "resnet": 1e-2}
WEIGHT_DECAY = 1e-4
RESULTS = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def macro_f1(predictions, targets):
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
    return sum(scores) / NUM_CLASSES


def run_epoch(model, name, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    set_model_mode(model, name, training)
    total_loss = 0.0
    seen = 0
    predictions = []
    targets = []
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        seen += batch_size
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    return total_loss / seen, macro_f1(torch.cat(predictions), torch.cat(targets))


def checkpoint_path(args):
    augment = "aug" if args.augment else "noaug"
    return RESULTS / f"{args.model}_seed{args.seed}_n{args.images_per_class}_{augment}.pt"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = WasteDataset(
        "train",
        subset_seed=args.seed,
        images_per_class=args.images_per_class,
        augment=args.augment,
    )
    val_set = WasteDataset("val")
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, pin_memory=use_cuda
    )

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, args.model),
        lr=LEARNING_RATES[args.model],
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    device_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    print(
        f"model {args.model} device {device_name} train {len(train_set)} val {len(val_set)} "
        f"augment {args.augment} classes {len(CLASSES)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(
            model, args.model, train_loader, device, criterion, optimizer
        )
        val_loss, val_f1 = run_epoch(model, args.model, val_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"epoch {epoch:02d} train_loss {train_loss:.4f} train_macro_f1 {train_f1:.4f} "
            f"val_loss {val_loss:.4f} val_macro_f1 {val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(args)
    torch.save(
        {
            "model": args.model,
            "seed": args.seed,
            "images_per_class": args.images_per_class,
            "augment": args.augment,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "history": history,
            "state_dict": best_state,
            "classes": CLASSES,
        },
        path,
    )
    print(f"best epoch {best_epoch} val_macro_f1 {best_f1:.4f}", flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
