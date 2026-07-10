import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


class EventDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.x = torch.from_numpy(data["x"]).float()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.y = torch.from_numpy(data["y"]).float()
        self.n_tracks = data["n_tracks"]
        self.event_id = data["event_id"]
        self.source_file = data["source_file"]

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:
        print(f"Skipping CUDA seed setup because CUDA is not fully usable yet: {exc}")


def make_loader(path, batch_size, num_workers, shuffle):
    dataset = EventDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def binary_metrics(y_true, logits, threshold=0.5):
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true.astype(np.int32),
        pred,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(y_true.astype(np.int32), pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(y_true, probs)),
        "threshold": float(threshold),
        "n_events": int(len(y_true)),
        "confusion_matrix": cm.astype(int).tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics, probs, pred


def dataset_summary(path):
    data = np.load(path, allow_pickle=True)
    y = data["y"].astype(np.int32)
    labels, counts = np.unique(y, return_counts=True)
    return {
        "events": int(y.shape[0]),
        "class_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "shape": list(data["x"].shape),
    }


def write_confusion_matrix_csv(path, matrix):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "pred_background_0", "pred_softbomb_1"])
        writer.writerow(["true_background_0", int(matrix[0][0]), int(matrix[0][1])])
        writer.writerow(["true_softbomb_1", int(matrix[1][0]), int(matrix[1][1])])


@torch.no_grad()
def evaluate_model(model, loader, device, loss_fn, threshold=0.5):
    model.eval()
    losses = []
    ys = []
    logits_all = []
    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        losses.append(float(loss.detach().cpu()))
        ys.append(y.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    logits = np.concatenate(logits_all)
    metrics, probs, pred = binary_metrics(y_true, logits, threshold)
    metrics["loss"] = float(np.mean(losses))
    return metrics, y_true, logits, probs, pred


def write_history(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
