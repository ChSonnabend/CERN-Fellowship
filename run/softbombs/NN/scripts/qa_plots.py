#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import load_config
from softbombs.onnx_export import load_checkpoint_model


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def plot_confusion_matrix(metrics, output_path):
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["pred bg", "pred softbomb"])
    ax.set_yticks([0, 1], ["true bg", "true softbomb"])
    ax.set_title(
        "Holdout confusion matrix\n"
        f"acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics.get('roc_auc', float('nan')):.3f}"
    )
    for row in range(2):
        for col in range(2):
            value = int(cm[row, col])
            ax.text(col, row, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def first_layer_attention(model, x, mask):
    model.eval()
    with torch.no_grad():
        tokens = model.input_projection(x)
        cls = model.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask.bool()], dim=1)
        if model.position is not None:
            tokens = tokens + model.position[:, : tokens.shape[1], :]

        layer = model.encoder.layers[0]
        attn_input = layer.norm1(tokens) if layer.norm_first else tokens
        _, weights = layer.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=~full_mask,
            need_weights=True,
            average_attn_weights=False,
        )
    return weights.cpu().numpy(), full_mask.cpu().numpy()


def plot_attention(attn_weights, full_mask, event_index, output_path, max_tracks):
    weights = attn_weights[0]
    real_token_indices = np.nonzero(full_mask[0])[0]
    kept = real_token_indices[: max_tracks + 1]
    labels = ["CLS"] + [f"t{i}" for i in range(len(kept) - 1)]

    with PdfPages(output_path) as pdf:
        avg = weights.mean(axis=0)
        for title, matrix in [("first layer average over heads", avg)]:
            fig, ax = plt.subplots(figsize=(8, 7))
            shown = matrix[np.ix_(kept, kept)]
            image = ax.imshow(shown, cmap="viridis", vmin=0.0)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Event {event_index}: {title}")
            ax.set_xlabel("attended token")
            ax.set_ylabel("query token")
            tick_step = max(1, len(labels) // 16)
            ticks = np.arange(0, len(labels), tick_step)
            ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=90)
            ax.set_yticks(ticks, [labels[i] for i in ticks])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for head in range(weights.shape[0]):
            fig, ax = plt.subplots(figsize=(8, 7))
            shown = weights[head][np.ix_(kept, kept)]
            image = ax.imshow(shown, cmap="viridis", vmin=0.0)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Event {event_index}: first layer head {head}")
            ax.set_xlabel("attended token")
            ax.set_ylabel("query token")
            tick_step = max(1, len(labels) // 16)
            ticks = np.arange(0, len(labels), tick_step)
            ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=90)
            ax.set_yticks(ticks, [labels[i] for i in ticks])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--max-attention-tracks", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training_dir = Path(config["training"]["output_dir"])
    dataset_dir = Path(config["training"]["dataset_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else training_dir / "qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = training_dir / f"{args.split}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run scripts/evaluate.py first.")
    metrics = load_metrics(metrics_path)
    confusion_pdf = output_dir / f"{args.split}_confusion_matrix.pdf"
    plot_confusion_matrix(metrics, confusion_pdf)

    checkpoint = Path(args.checkpoint) if args.checkpoint else training_dir / "best_model.pt"
    model, _, _, _, _ = load_checkpoint_model(checkpoint, config_override=config, device="cpu")
    data = np.load(dataset_dir / f"{args.split}.npz", allow_pickle=True)
    if args.event_index < 0 or args.event_index >= data["x"].shape[0]:
        raise IndexError(f"event-index {args.event_index} outside split with {data['x'].shape[0]} events")

    x = torch.from_numpy(data["x"][args.event_index : args.event_index + 1]).float()
    mask = torch.from_numpy(data["mask"][args.event_index : args.event_index + 1]).bool()
    attn_weights, full_mask = first_layer_attention(model, x, mask)
    attention_pdf = output_dir / f"{args.split}_event{args.event_index}_first_layer_attention.pdf"
    plot_attention(attn_weights, full_mask, args.event_index, attention_pdf, args.max_attention_tracks)

    print(f"Wrote {confusion_pdf}")
    print(f"Wrote {attention_pdf}")


if __name__ == "__main__":
    main()

