import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import ensure_dir, write_json
from .root_io import (
    branch_names,
    find_best_tree,
    find_tree_by_token,
    list_root_files,
    open_root,
    prefer_merged_aod_files,
    read_tree_arrays,
    resolve_feature_branches,
    resolve_first,
)


BASE_FEATURES = ["tpc_dedx", "tof_beta", "momentum", "tan_lambda", "sin_phi"]


def discover_files(config):
    pattern = config["input"].get("file_pattern", "AO2D.root")
    excludes = config["input"].get("exclude_path_substrings", [])
    files_by_class = []
    for class_config in config["input"]["classes"]:
        files = list_root_files(class_config["local_dir"], pattern, class_config.get("max_files"))
        files = [file_name for file_name in files if not any(token in file_name for token in excludes)]
        if config["input"].get("prefer_merged_aod_per_job", True):
            files = prefer_merged_aod_files(files, class_config["local_dir"])
        files_by_class.append((class_config, files))
    return files_by_class


def _derive_momentum(arrays, spec):
    tan_lambda = arrays[spec["tan_lambda"]].astype(np.float32)
    if spec.get("pt"):
        pt = np.abs(arrays[spec["pt"]].astype(np.float32))
    else:
        signed_inv_pt = arrays[spec["signed_inverse_pt"]].astype(np.float32)
        pt = np.divide(1.0, np.abs(signed_inv_pt), out=np.zeros_like(signed_inv_pt), where=np.abs(signed_inv_pt) > 0)
    return pt * np.sqrt(1.0 + tan_lambda * tan_lambda)


def _materialize_features(arrays, resolved, include_multiplicity):
    columns = []
    for feature in BASE_FEATURES:
        if feature in resolved["features"]:
            columns.append(arrays[resolved["features"][feature]].astype(np.float32))
        elif feature == "momentum" and feature in resolved["derived"]:
            columns.append(_derive_momentum(arrays, resolved["derived"][feature]).astype(np.float32))
        else:
            raise RuntimeError(f"Missing required feature '{feature}' in resolved branch map: {resolved}")
    features = np.stack(columns, axis=1)
    return features


def _resolve_joined_branches(track_branches, extra_branches, dataset_config):
    resolved = {
        "track_tree_features": {},
        "extra_tree_features": {},
        "derived": {},
        "event_id": None,
        "tof_beta_is_fallback": False,
    }
    resolved["event_id"] = resolve_first(track_branches, dataset_config["event_id_branches"])

    for feature, spec in dataset_config["features"].items():
        direct_track = resolve_first(track_branches, spec.get("branches", []))
        direct_extra = resolve_first(extra_branches, spec.get("branches", []))
        if direct_track:
            resolved["track_tree_features"][feature] = direct_track
            continue
        if direct_extra:
            resolved["extra_tree_features"][feature] = direct_extra
            continue

        if feature == "tof_beta":
            fallback_extra = resolve_first(extra_branches, spec.get("fallback_branches", []))
            fallback_track = resolve_first(track_branches, spec.get("fallback_branches", []))
            if fallback_extra:
                resolved["extra_tree_features"][feature] = fallback_extra
                resolved["tof_beta_is_fallback"] = True
                continue
            if fallback_track:
                resolved["track_tree_features"][feature] = fallback_track
                resolved["tof_beta_is_fallback"] = True
                continue

        if feature == "momentum":
            derive = spec.get("derive_from", {})
            pt = resolve_first(track_branches, derive.get("pt", [])) or resolve_first(extra_branches, derive.get("pt", []))
            signed_inverse_pt = resolve_first(track_branches, derive.get("signed_inverse_pt", [])) or resolve_first(
                extra_branches, derive.get("signed_inverse_pt", [])
            )
            tan_lambda = resolve_first(track_branches, derive.get("tan_lambda", [])) or resolve_first(
                extra_branches, derive.get("tan_lambda", [])
            )
            if tan_lambda and (pt or signed_inverse_pt):
                resolved["derived"]["momentum"] = {
                    "pt": pt,
                    "signed_inverse_pt": signed_inverse_pt,
                    "tan_lambda": tan_lambda,
                }
    return resolved


def _collect_needed_branches(resolved):
    track_needed = set()
    extra_needed = set()
    if resolved["event_id"]:
        track_needed.add(resolved["event_id"])
    track_needed.update(resolved["track_tree_features"].values())
    extra_needed.update(resolved["extra_tree_features"].values())
    for spec in resolved["derived"].values():
        for value in spec.values():
            if value:
                track_needed.add(value)
                extra_needed.add(value)
    return track_needed, extra_needed


def _materialize_joined_features(track_arrays, extra_arrays, resolved):
    def get_array(branch):
        if branch in track_arrays:
            return track_arrays[branch]
        if branch in extra_arrays:
            return extra_arrays[branch]
        raise KeyError(branch)

    columns = []
    for feature in BASE_FEATURES:
        if feature in resolved["track_tree_features"]:
            columns.append(track_arrays[resolved["track_tree_features"][feature]].astype(np.float32))
        elif feature in resolved["extra_tree_features"]:
            columns.append(extra_arrays[resolved["extra_tree_features"][feature]].astype(np.float32))
        elif feature == "momentum" and feature in resolved["derived"]:
            spec = resolved["derived"][feature]
            merged_arrays = {key: get_array(key) for key in spec.values() if key}
            columns.append(_derive_momentum(merged_arrays, spec).astype(np.float32))
        else:
            raise RuntimeError(f"Missing required feature '{feature}' in resolved branch map: {resolved}")
    return np.stack(columns, axis=1)


def _build_events_from_file(file_name, class_label, config, tree_info):
    dataset_config = config["dataset"]
    track_tree = find_tree_by_token(file_name, dataset_config.get("track_tree_name", "O2track_iu"))
    extra_tree = find_tree_by_token(file_name, dataset_config.get("track_extra_tree_name", "O2trackextra"))
    if not track_tree or not extra_tree:
        raise RuntimeError(f"Could not find track/extra trees in {file_name}: track={track_tree}, extra={extra_tree}")

    with open_root(file_name) as root_file:
        track = root_file[track_tree]
        extra = root_file[extra_tree]
        if int(track.num_entries) != int(extra.num_entries):
            raise RuntimeError(
                f"Track and extra trees have different entries in {file_name}: "
                f"{track_tree}={track.num_entries}, {extra_tree}={extra.num_entries}"
            )
        track_branches = branch_names(track)
        extra_branches = branch_names(extra)

    resolved = _resolve_joined_branches(track_branches, extra_branches, dataset_config)
    missing = [
        feature
        for feature in BASE_FEATURES
        if feature not in resolved["track_tree_features"]
        and feature not in resolved["extra_tree_features"]
        and feature not in resolved["derived"]
    ]
    if missing:
        raise RuntimeError(
            f"Missing features {missing} in {file_name}:{track_tree}+{extra_tree}. "
            f"Run scripts/inspect_root.py and update the config aliases."
        )
    if not resolved["event_id"]:
        raise RuntimeError(
            f"No event id branch found in {file_name}:{track_tree}. "
            f"Configured candidates: {dataset_config['event_id_branches']}"
        )

    track_needed, extra_needed = _collect_needed_branches(resolved)
    track_branches = set(track_branches)
    extra_branches = set(extra_branches)
    track_arrays = read_tree_arrays(file_name, track_tree, sorted(track_needed & track_branches))
    extra_arrays = read_tree_arrays(file_name, extra_tree, sorted(extra_needed & extra_branches))
    event_ids = track_arrays[resolved["event_id"]]
    features = _materialize_joined_features(track_arrays, extra_arrays, resolved)

    if dataset_config.get("drop_nan_tracks", True):
        finite = np.isfinite(features).all(axis=1)
        event_ids = event_ids[finite]
        features = features[finite]

    unique_events, inverse = np.unique(event_ids, return_inverse=True)
    max_tracks = int(dataset_config["max_tracks"])
    min_tracks = int(dataset_config.get("min_tracks", 1))
    include_mult = bool(dataset_config.get("include_event_multiplicity", True))
    sort_feature = dataset_config.get("sort_tracks_by", "momentum")
    sort_desc = bool(dataset_config.get("sort_descending", True))
    sort_idx = BASE_FEATURES.index(sort_feature) if sort_feature in BASE_FEATURES else None

    events = []
    for event_pos, event_id in enumerate(unique_events):
        idx = np.nonzero(inverse == event_pos)[0]
        if idx.size < min_tracks:
            continue
        values = features[idx]
        if sort_idx is not None:
            order = np.argsort(values[:, sort_idx])
            if sort_desc:
                order = order[::-1]
            values = values[order]
        values = values[:max_tracks]

        multiplicity = float(idx.size)
        if include_mult:
            mult_column = np.full((values.shape[0], 1), multiplicity, dtype=np.float32)
            values = np.concatenate([values, mult_column], axis=1)

        padded = np.zeros((max_tracks, values.shape[1]), dtype=np.float32)
        mask = np.zeros((max_tracks,), dtype=bool)
        padded[: values.shape[0]] = values
        mask[: values.shape[0]] = True
        events.append(
            {
                "x": padded,
                "mask": mask,
                "y": int(class_label),
                "event_id": str(event_id),
                "source_file": str(file_name),
                "n_tracks": int(idx.size),
            }
        )
    return events, resolved


def _split_indices(n, val_fraction, test_fraction, rng):
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = indices[:n_test]
    val = indices[n_test : n_test + n_val]
    train = indices[n_test + n_val :]
    return train, val, test


def _pack(events, indices):
    selected = [events[i] for i in indices]
    x = np.stack([item["x"] for item in selected]).astype(np.float32)
    mask = np.stack([item["mask"] for item in selected])
    y = np.asarray([item["y"] for item in selected], dtype=np.float32)
    n_tracks = np.asarray([item["n_tracks"] for item in selected], dtype=np.int32)
    event_id = np.asarray([item["event_id"] for item in selected])
    source_file = np.asarray([item["source_file"] for item in selected])
    return x, mask, y, n_tracks, event_id, source_file


def _fit_scaler(x, mask):
    valid = x[mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return mean, std


def _fit_clip_bounds(x, mask, quantiles):
    if not quantiles:
        return None, None
    lo_q, hi_q = float(quantiles[0]), float(quantiles[1])
    valid = x[mask]
    lower = np.quantile(valid, lo_q, axis=0).astype(np.float32)
    upper = np.quantile(valid, hi_q, axis=0).astype(np.float32)
    return lower, upper


def _apply_clip(x, mask, lower, upper):
    if lower is None or upper is None:
        return x
    clipped = x.copy()
    clipped[mask] = np.clip(clipped[mask], lower, upper)
    clipped[~mask] = 0.0
    return clipped


def _apply_scaler(x, mask, mean, std):
    x_scaled = x.copy()
    x_scaled[mask] = (x_scaled[mask] - mean) / std
    x_scaled[~mask] = 0.0
    return x_scaled


def build_dataset(config):
    output_dir = Path(config["dataset"]["output_dir"])
    ensure_dir(output_dir)

    files_by_class = discover_files(config)
    for class_config, files in files_by_class:
        if not files:
            raise RuntimeError(f"No ROOT files found for class '{class_config['name']}' in {class_config['local_dir']}")

    all_probe_files = [file_name for _, files in files_by_class for file_name in files[:1]]
    tree_info = {"tree_path": "joined:O2track_iu+O2trackextra"}

    events_by_label = {}
    branch_resolution = {}
    skipped_files = []
    for class_config, files in files_by_class:
        class_events = []
        iterator = tqdm(files, desc=f"Reading {class_config['name']}", unit="file")
        for file_name in iterator:
            try:
                events, resolved = _build_events_from_file(file_name, class_config["label"], config, tree_info)
            except Exception as exc:
                if not config["dataset"].get("skip_bad_files", True):
                    raise
                skipped_files.append(
                    {
                        "class": class_config["name"],
                        "label": int(class_config["label"]),
                        "file": str(file_name),
                        "error": str(exc),
                    }
                )
                iterator.write(f"Skipping unreadable file: {file_name} ({exc})")
                continue
            branch_resolution[class_config["name"]] = resolved
            class_events.extend(events)
        events_by_label[int(class_config["label"])] = class_events

    rng = np.random.default_rng(int(config["dataset"].get("shuffle_seed", config["project"].get("seed", 1337))))
    min_count = min(len(events) for events in events_by_label.values())
    cap = config["dataset"].get("max_events_per_class")
    if cap is not None:
        min_count = min(min_count, int(cap))
    if min_count <= 0:
        raise RuntimeError("No events survived selection in at least one class.")

    split_events = {"train": [], "val": [], "holdout": []}
    holdout_fraction = float(config["dataset"].get("holdout_fraction", config["dataset"].get("test_fraction", 0.15)))
    for label, events in events_by_label.items():
        order = np.arange(len(events))
        rng.shuffle(order)
        selected = [events[i] for i in order[:min_count]]
        train_idx, val_idx, holdout_idx = _split_indices(
            len(selected),
            float(config["dataset"]["validation_fraction"]),
            holdout_fraction,
            rng,
        )
        split_events["train"].extend([selected[i] for i in train_idx])
        split_events["val"].extend([selected[i] for i in val_idx])
        split_events["holdout"].extend([selected[i] for i in holdout_idx])

    for split_name, events in split_events.items():
        order = np.arange(len(events))
        rng.shuffle(order)
        split_events[split_name] = [events[i] for i in order]

    if not split_events["train"]:
        raise RuntimeError("The training split is empty. Increase max_events_per_class or reduce validation/test fractions.")

    splits = {
        split_name: _pack(events, np.arange(len(events)))
        for split_name, events in split_events.items()
        if events
    }

    clip_lower, clip_upper = _fit_clip_bounds(
        splits["train"][0],
        splits["train"][1],
        config["dataset"].get("clip_quantiles"),
    )
    if clip_lower is not None:
        for split_name, payload in list(splits.items()):
            x, mask, y, n_tracks, event_id, source_file = payload
            splits[split_name] = (_apply_clip(x, mask, clip_lower, clip_upper), mask, y, n_tracks, event_id, source_file)

    if config["dataset"].get("normalize", True):
        mean, std = _fit_scaler(splits["train"][0], splits["train"][1])
        for split_name, payload in list(splits.items()):
            x, mask, y, n_tracks, event_id, source_file = payload
            splits[split_name] = (_apply_scaler(x, mask, mean, std), mask, y, n_tracks, event_id, source_file)
        np.savez(output_dir / "scaler.npz", mean=mean, std=std, clip_lower=clip_lower, clip_upper=clip_upper)
    else:
        feature_dim = splits["train"][0].shape[-1]
        mean = np.zeros((feature_dim,), dtype=np.float32)
        std = np.ones((feature_dim,), dtype=np.float32)
        np.savez(output_dir / "scaler.npz", mean=mean, std=std, clip_lower=clip_lower, clip_upper=clip_upper)

    for split_name, payload in splits.items():
        x, mask, y, n_tracks, event_id, source_file = payload
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            x=x,
            mask=mask,
            y=y,
            n_tracks=n_tracks,
            event_id=event_id,
            source_file=source_file,
        )
        if split_name == "holdout":
            np.savez_compressed(
                output_dir / "test.npz",
                x=x,
                mask=mask,
                y=y,
                n_tracks=n_tracks,
                event_id=event_id,
                source_file=source_file,
            )

    feature_names = BASE_FEATURES.copy()
    if config["dataset"].get("include_event_multiplicity", True):
        feature_names.append("event_multiplicity")

    if any(resolution.get("tof_beta_is_fallback") for resolution in branch_resolution.values()):
        feature_names[feature_names.index("tof_beta")] = "tof_pid_proxy"

    metadata = {
        "feature_names": feature_names,
        "max_tracks": int(config["dataset"]["max_tracks"]),
        "include_event_multiplicity": bool(config["dataset"].get("include_event_multiplicity", True)),
        "sort_tracks_by": config["dataset"].get("sort_tracks_by", "momentum"),
        "sort_descending": bool(config["dataset"].get("sort_descending", True)),
        "feature_note": "tof_pid_proxy is used when no direct TOF beta branch is available; default fallback is fTOFExpMom.",
        "tree_path": tree_info["tree_path"],
        "branch_resolution": branch_resolution,
        "class_counts_before_balance": {str(label): len(events) for label, events in events_by_label.items()},
        "skipped_files": skipped_files,
        "events_per_class_after_balance": min_count,
        "split_counts": {name: int(payload[2].shape[0]) for name, payload in splits.items()},
        "split_class_counts": {
            name: {
                str(label): int(np.sum(payload[2] == label))
                for label in sorted(events_by_label)
            }
            for name, payload in splits.items()
        },
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "clip_lower": None if clip_lower is None else clip_lower.tolist(),
        "clip_upper": None if clip_upper is None else clip_upper.tolist(),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata
