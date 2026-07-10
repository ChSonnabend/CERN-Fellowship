import copy
import json
import os
from pathlib import Path


def deep_update(base, updates):
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    parent_name = config.pop("inherits", None)
    if parent_name:
        parent_path = (path.parent / parent_name).resolve()
        config = deep_update(load_config(parent_path), config)

    config["_config_path"] = str(path)
    apply_runtime_overrides(config)
    return config


def apply_runtime_overrides(config):
    training_output_dir = os.environ.get("SOFTBOMB_TRAINING_OUTPUT_DIR")
    if training_output_dir:
        config.setdefault("training", {})["output_dir"] = training_output_dir

    dataset_output_dir = os.environ.get("SOFTBOMB_DATASET_OUTPUT_DIR")
    if dataset_output_dir:
        config.setdefault("dataset", {})["output_dir"] = dataset_output_dir
        if "training" in config:
            config["training"]["dataset_dir"] = dataset_output_dir

    job_output_dir = os.environ.get("SOFTBOMB_JOB_OUTPUT_DIR")
    if job_output_dir:
        config.setdefault("runtime", {})["job_output_dir"] = job_output_dir


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def project_root(config):
    return Path(config["project"]["framework"]).resolve()


def expand_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
