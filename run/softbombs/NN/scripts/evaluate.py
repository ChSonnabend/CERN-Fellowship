#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import load_config
from softbombs.model import build_model_from_config
from softbombs.train_utils import evaluate_model, make_loader, write_confusion_matrix_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="holdout", choices=["train", "val", "test", "holdout"])
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    dataset_dir = Path(train_cfg["dataset_dir"])
    output_dir = Path(train_cfg["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device_name = train_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_model_from_config(config, checkpoint["input_dim"], checkpoint["max_tracks"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = make_loader(dataset_dir / f"{args.split}.npz", train_cfg["batch_size"], train_cfg["num_workers"], False)
    metrics, y_true, logits, probs, pred = evaluate_model(
        model,
        loader,
        device,
        nn.BCEWithLogitsLoss(),
        threshold=float(train_cfg.get("threshold", 0.5)),
    )
    write_json(output_dir / f"{args.split}_metrics.json", metrics)
    write_confusion_matrix_csv(output_dir / f"{args.split}_confusion_matrix.csv", metrics["confusion_matrix"])

    data = np.load(dataset_dir / f"{args.split}.npz", allow_pickle=True)
    with open(output_dir / f"{args.split}_predictions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "source_file", "label", "logit", "probability", "prediction", "n_tracks"])
        for row in zip(data["event_id"], data["source_file"], y_true, logits, probs, pred, data["n_tracks"]):
            writer.writerow(row)
    print(metrics)


if __name__ == "__main__":
    main()
