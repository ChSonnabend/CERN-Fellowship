from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import write_json
from .model import build_model_from_config


class OnnxSoftbombWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tracks, mask):
        logits = self.model(tracks, mask)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities


def load_checkpoint_model(checkpoint_path, config_override=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config_override or checkpoint["config"]
    input_dim = int(checkpoint["input_dim"])
    max_tracks = int(checkpoint["max_tracks"])
    model = build_model_from_config(config, input_dim=input_dim, max_tracks=max_tracks)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, config, input_dim, max_tracks


def export_checkpoint_to_onnx(
    checkpoint_path,
    output_path,
    config_override=None,
    opset_version=17,
    verify=True,
):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, checkpoint, config, input_dim, max_tracks = load_checkpoint_model(
        checkpoint_path,
        config_override=config_override,
        device="cpu",
    )
    wrapper = OnnxSoftbombWrapper(model).eval()

    dummy_tracks = torch.zeros((1, max_tracks, input_dim), dtype=torch.float32)
    dummy_mask = torch.ones((1, max_tracks), dtype=torch.bool)

    torch.onnx.export(
        wrapper,
        (dummy_tracks, dummy_mask),
        output_path,
        input_names=["tracks", "mask"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "tracks": {0: "batch"},
            "mask": {0: "batch"},
            "logits": {0: "batch"},
            "probabilities": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    verification = None
    if verify:
        verification = verify_onnx(wrapper, output_path, dummy_tracks, dummy_mask)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(output_path),
        "input_dim": input_dim,
        "max_tracks": max_tracks,
        "epoch": checkpoint.get("epoch"),
        "opset_version": opset_version,
        "outputs": ["logits", "probabilities"],
        "verification": verification,
    }
    write_json(output_path.with_suffix(".onnx.json"), metadata)
    return metadata


def verify_onnx(wrapper, output_path, tracks, mask):
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime is not installed"}

    with torch.no_grad():
        torch_logits, torch_probs = wrapper(tracks, mask)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_probs = session.run(
        None,
        {
            "tracks": tracks.numpy(),
            "mask": mask.numpy(),
        },
    )
    logits_diff = float(np.max(np.abs(torch_logits.numpy() - onnx_logits)))
    probs_diff = float(np.max(np.abs(torch_probs.numpy() - onnx_probs)))
    return {
        "status": "ok",
        "max_abs_diff_logits": logits_diff,
        "max_abs_diff_probabilities": probs_diff,
    }

