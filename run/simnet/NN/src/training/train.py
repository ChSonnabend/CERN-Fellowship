#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_module(path):
    spec = importlib.util.spec_from_file_location("simnet_configurations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-type", choices=["NN", "BDT"], default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def read_config(path):
    with open(path) as f:
        return json.load(f)


def finite_frame(frame):
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.fillna(0.0)


def load_data(configurations, csv_override=None, max_rows=None):
    csv_path = Path(csv_override or configurations.data_path)
    nrows = max_rows or int(getattr(configurations, "num_datapoints", 0) or 0) or None
    data = pd.read_csv(csv_path, usecols=configurations.labels_x + [configurations.label_y], nrows=nrows)
    x = finite_frame(data[configurations.labels_x]).to_numpy(dtype=np.float32)
    y = data[configurations.label_y].to_numpy(dtype=np.float32)
    return x, y, csv_path


def split_data(x, y, configurations):
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x,
        y,
        test_size=configurations.test_size + configurations.validation_size,
        random_state=configurations.random_state,
        shuffle=configurations.shuffle,
        stratify=y,
    )
    rel_val = configurations.validation_size / (configurations.test_size + configurations.validation_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=1.0 - rel_val,
        random_state=configurations.random_state,
        shuffle=configurations.shuffle,
        stratify=y_tmp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_loaders(x, y, configurations, train_cfg):
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, configurations)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=batch_size)
    return train_loader, val_loader, test_loader, scaler, x_test[:1024], y_train


def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses = []
    ys = []
    scores = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(loss.item() * len(yb))
            ys.append(yb.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    y_pred = (y_score >= 0.5).astype(np.int32)
    metrics = {
        "loss": float(np.sum(losses) / len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def train_bdt(conf, configurations, args, x, y, csv_path, output_dir):
    for imp in conf["directory_settings"].get("classes", []):
        if imp not in sys.path:
            sys.path.append(imp)
    from BDTClass.BDTClasses.BDT_class import BDTClassifier

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, configurations)
    bdt_cfg = getattr(configurations, "bdt", {})
    model = BDTClassifier(
        features=configurations.labels_x,
        label=configurations.label_y,
        model_settings=bdt_cfg.get("model_settings", {}),
        scale_inputs=bdt_cfg.get("scale_inputs", False),
    )
    print(f"Training BDT on {len(x)} rows from {csv_path}")
    print(f"Features: {x.shape[1]}, positive fraction: {np.mean(y):.4f}")
    history = model.fit(x_train, y_train, x_val, y_val)
    test_metrics = model.evaluate(x_test, y_test)

    pd.DataFrame(history).to_csv(output_dir / "bdt_history.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "model_type": "BDT",
                "csv_path": str(csv_path),
                "features": configurations.labels_x,
                "label": configurations.label_y,
                "test_metrics": test_metrics,
                "bdt_config": bdt_cfg,
            },
            f,
            indent=2,
        )
    model.save(output_dir, {"csv_path": str(csv_path), "test_metrics": test_metrics})

    if conf["network_settings"].get("save_as_onnx", True):
        try:
            model.save_onnx(output_dir / "bdt.onnx", input_dim=x.shape[1])
        except RuntimeError as exc:
            message = str(exc)
            (output_dir / "onnx_export_error.txt").write_text(message + "\n")
            print(message)

    print(json.dumps(test_metrics, indent=2))


def train():
    args = parse_args()
    conf = read_config(args.config)
    configurations = load_module(conf["network_settings"]["configurations_file"])
    model_type = (args.model_type or conf["network_settings"].get("model_type", "NN")).upper()
    train_cfg = dict(configurations.training)
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    output_dir = Path(args.output_dir or Path.cwd() / "network")
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, csv_path = load_data(configurations, args.csv, args.max_rows)
    if model_type == "BDT":
        train_bdt(conf, configurations, args, x, y, csv_path, output_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader, test_loader, scaler, onnx_example, y_train = make_loaders(x, y, configurations, train_cfg)

    model = configurations.network(x.shape[1]).to(device)
    pos_weight = None
    if train_cfg.get("class_weighting", True):
        n_pos = float(np.sum(y_train == 1))
        n_neg = float(np.sum(y_train == 0))
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = math.inf
    best_state = None
    stale_epochs = 0
    history = []

    print(f"Training on {len(x)} rows from {csv_path}")
    print(f"Features: {x.shape[1]}, device: {device}, positive fraction: {np.mean(y):.4f}")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total_loss += loss.item() * len(yb)
            total_seen += len(yb)

        train_loss = total_loss / total_seen
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["loss"])
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}})
        print(f"epoch {epoch:04d} train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} val_auc={val_metrics.get('roc_auc', float('nan')):.5f}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(train_cfg["early_stopping_patience"]):
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    pd.DataFrame(history).to_csv(output_dir / "loss_history.csv", index=False)
    np.savez(output_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, labels_x=np.array(configurations.labels_x))

    model_info = {
        "csv_path": str(csv_path),
        "features": configurations.labels_x,
        "label": configurations.label_y,
        "test_metrics": test_metrics,
        "train_config": train_cfg,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(model_info, f, indent=2)

    if conf["network_settings"].get("save_as_pt", True):
        torch.save({"model_state_dict": model.state_dict(), "features": configurations.labels_x, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, output_dir / "net.pt")
    if conf["network_settings"].get("save_as_onnx", True):
        model.eval()
        example = torch.from_numpy(onnx_example.astype(np.float32)).to(device)
        torch.onnx.export(
            model,
            example,
            output_dir / "net.onnx",
            input_names=["input"],
            output_names=["logit"],
            dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
            opset_version=18,
            dynamo=False,
        )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    train()
