#!/usr/bin/env python3
import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import ensure_dir, load_config
from softbombs.model import build_model_from_config
from softbombs.onnx_export import export_checkpoint_to_onnx
from softbombs.train_utils import (
    dataset_summary,
    evaluate_model,
    make_loader,
    seed_everything,
    write_confusion_matrix_csv,
    write_history,
    write_json,
)


def make_grad_scaler(device, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def write_predictions_csv(path, split_path, y_true, logits, probs, pred):
    data = np.load(split_path, allow_pickle=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "source_file", "label", "logit", "probability", "prediction", "n_tracks"])
        for row in zip(data["event_id"], data["source_file"], y_true, logits, probs, pred, data["n_tracks"]):
            writer.writerow(row)


def validate_dataset_matches_config(config, dataset_dir, sample):
    dataset_cfg = config.get("dataset", {})
    metadata_path = dataset_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

    expected_max_tracks = int(dataset_cfg.get("max_tracks", sample["x"].shape[1]))
    actual_max_tracks = int(metadata.get("max_tracks", sample["x"].shape[1]))
    if actual_max_tracks != expected_max_tracks:
        raise RuntimeError(
            "Dataset/config mismatch: "
            f"config dataset.max_tracks={expected_max_tracks}, but dataset has max_tracks={actual_max_tracks} "
            f"(train.npz shape is {tuple(sample['x'].shape)}). "
            "Rebuild the dataset with scripts/build_dataset.py before training."
        )

    expected_include_mult = bool(dataset_cfg.get("include_event_multiplicity", True))
    feature_names = list(metadata.get("feature_names", []))
    actual_include_mult = bool(metadata.get("include_event_multiplicity", "event_multiplicity" in feature_names))
    if actual_include_mult != expected_include_mult:
        raise RuntimeError(
            "Dataset/config mismatch: "
            f"config dataset.include_event_multiplicity={expected_include_mult}, "
            f"but dataset include_event_multiplicity={actual_include_mult} with feature_names={feature_names}. "
            "Rebuild the dataset with scripts/build_dataset.py before training."
        )


def run_holdout_outputs(config, output_dir, dataset_dir, train_cfg, device, loss_fn):
    split = train_cfg.get("post_training_split", "holdout")
    split_path = dataset_dir / f"{split}.npz"
    checkpoint_path = output_dir / "best_model.pt"
    if not split_path.exists():
        print(f"Skipping post-training QA because {split_path} does not exist")
        return
    if not checkpoint_path.exists():
        print(f"Skipping post-training QA because {checkpoint_path} does not exist")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model_from_config(config, checkpoint["input_dim"], checkpoint["max_tracks"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = make_loader(split_path, train_cfg["batch_size"], train_cfg["num_workers"], False)
    metrics, y_true, logits, probs, pred = evaluate_model(
        model,
        loader,
        device,
        loss_fn,
        threshold=float(train_cfg.get("threshold", 0.5)),
    )
    write_json(output_dir / f"{split}_metrics.json", metrics)
    write_confusion_matrix_csv(output_dir / f"{split}_confusion_matrix.csv", metrics["confusion_matrix"])
    write_predictions_csv(output_dir / f"{split}_predictions.csv", split_path, y_true, logits, probs, pred)
    print(f"Wrote {split} metrics and predictions to {output_dir}")

    if train_cfg.get("qa_plots_after_train", True):
        from qa_plots import first_layer_attention, plot_attention, plot_confusion_matrix

        qa_dir = output_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        confusion_pdf = qa_dir / f"{split}_confusion_matrix.pdf"
        plot_confusion_matrix(metrics, confusion_pdf)

        event_index = int(train_cfg.get("qa_event_index", 0))
        max_attention_tracks = int(train_cfg.get("qa_max_attention_tracks", 64))
        data = np.load(split_path, allow_pickle=True)
        if 0 <= event_index < data["x"].shape[0]:
            model_cpu = model.to("cpu")
            x = torch.from_numpy(data["x"][event_index : event_index + 1]).float()
            mask = torch.from_numpy(data["mask"][event_index : event_index + 1]).bool()
            attn_weights, full_mask = first_layer_attention(model_cpu, x, mask)
            attention_pdf = qa_dir / f"{split}_event{event_index}_first_layer_attention.pdf"
            plot_attention(attn_weights, full_mask, event_index, attention_pdf, max_attention_tracks)
            print(f"Wrote QA plots: {confusion_pdf}, {attention_pdf}")
        else:
            print(f"Skipping attention QA: event index {event_index} outside {split} split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["project"].get("seed", 1337)))

    train_cfg = config["training"]
    dataset_dir = Path(train_cfg["dataset_dir"])
    output_dir = Path(train_cfg["output_dir"])
    ensure_dir(output_dir)
    print(f"Training output directory: {output_dir}")

    train_loader = make_loader(dataset_dir / "train.npz", train_cfg["batch_size"], train_cfg["num_workers"], True)
    val_loader = make_loader(dataset_dir / "val.npz", train_cfg["batch_size"], train_cfg["num_workers"], False)

    split_summaries = {}
    for split_name in ["train", "val", "holdout"]:
        split_path = dataset_dir / f"{split_name}.npz"
        if split_path.exists():
            split_summaries[split_name] = dataset_summary(split_path)
            print(
                f"{split_name} datapoints: "
                f"{split_summaries[split_name]['events']} events, "
                f"class_counts={split_summaries[split_name]['class_counts']}"
            )
    if "train" in split_summaries:
        print(f"Training datapoints: {split_summaries['train']['events']} events")

    sample = np.load(dataset_dir / "train.npz")
    validate_dataset_matches_config(config, dataset_dir, sample)
    input_dim = int(sample["x"].shape[-1])
    max_tracks = int(sample["x"].shape[1])

    device_name = train_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_model_from_config(config, input_dim=input_dim, max_tracks=max_tracks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = make_grad_scaler(device, enabled=bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda")

    best_loss = float("inf")
    best_epoch = -1
    history = []
    patience = int(train_cfg.get("early_stopping_patience", 12))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for x, mask, y in train_loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=scaler.is_enabled()):
                logits = model(x, mask)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if train_cfg.get("gradient_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics, _, _, _, _ = evaluate_model(model, val_loader, device, loss_fn, threshold=float(train_cfg.get("threshold", 0.5)))
        train_loss = float(np.mean(train_losses))
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_average_precision": val_metrics["average_precision"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "input_dim": input_dim,
                    "max_tracks": max_tracks,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "input_dim": input_dim,
            "max_tracks": max_tracks,
            "epoch": epoch,
        },
        output_dir / "last_model.pt",
    )
    write_history(output_dir / "history.csv", history)
    write_json(
        output_dir / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "dataset_summary": split_summaries,
        },
    )

    if train_cfg.get("export_onnx", False):
        metadata = export_checkpoint_to_onnx(
            output_dir / "best_model.pt",
            output_dir / "best_model.onnx",
            config_override=config,
            verify=True,
        )
        print(f"Exported ONNX: {metadata['onnx']}")

    if train_cfg.get("evaluate_after_train", True):
        run_holdout_outputs(config, output_dir, dataset_dir, train_cfg, device, loss_fn)


if __name__ == "__main__":
    main()
