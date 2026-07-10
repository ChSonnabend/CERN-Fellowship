from pathlib import Path

import numpy as np
import uproot


def is_tree(obj):
    return hasattr(obj, "arrays") and hasattr(obj, "num_entries") and hasattr(obj, "keys")


def iter_trees(root_file):
    def visit(prefix, obj):
        if is_tree(obj):
            yield prefix, obj
            return
        if hasattr(obj, "keys"):
            for key in obj.keys():
                child_path = f"{prefix}/{key}" if prefix else key
                try:
                    child = obj[key]
                except Exception:
                    continue
                yield from visit(child_path, child)

    yield from visit("", root_file)


def open_root(path):
    return uproot.open(path, file_handler=uproot.MultithreadedFileSource, num_workers=4)


def strip_cycle(name):
    return name.rsplit(";", 1)[0] if ";" in name else name


def branch_names(tree):
    return [strip_cycle(str(branch)) for branch in tree.keys()]


def list_root_files(local_dir, pattern="AO2D.root", max_files=None):
    root = Path(local_dir)
    files = sorted(root.rglob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    return [str(path) for path in files]


def prefer_merged_aod_files(files, base_dir=None):
    """Prefer <job>/AO2D.root over <job>/tf*/AO2D.root when both exist."""
    grouped = {}
    passthrough = []
    base = Path(base_dir).resolve() if base_dir else None

    for file_name in sorted(files):
        path = Path(file_name)
        try:
            rel = path.resolve().relative_to(base) if base else path
        except ValueError:
            rel = path
        parts = rel.parts
        if len(parts) >= 2:
            grouped.setdefault(parts[0], []).append(file_name)
        else:
            passthrough.append(file_name)

    selected = []
    for _, candidates in sorted(grouped.items()):
        merged = [
            candidate
            for candidate in candidates
            if len(Path(candidate).parts) >= 2 and Path(candidate).name == "AO2D.root"
        ]
        if base:
            merged = [
                candidate
                for candidate in candidates
                if Path(candidate).resolve().relative_to(base).parts == (Path(candidate).resolve().relative_to(base).parts[0], "AO2D.root")
            ]
        if merged:
            selected.extend(sorted(merged)[:1])
        else:
            selected.extend(sorted(candidates))
    if not grouped:
        selected.extend(passthrough)
    return sorted(selected)


def find_tree_by_token(file_name, token):
    token = token.lower()
    with open_root(file_name) as root_file:
        candidates = []
        for tree_path, tree in iter_trees(root_file):
            clean = strip_cycle(tree_path).lower()
            if token in clean:
                candidates.append((tree_path, int(tree.num_entries)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def resolve_first(branches, aliases):
    branch_set = set(branches)
    for alias in aliases:
        if alias in branch_set:
            return alias
    lower_map = {branch.lower(): branch for branch in branches}
    for alias in aliases:
        hit = lower_map.get(alias.lower())
        if hit:
            return hit
    return None


def resolve_feature_branches(branches, feature_config, event_id_aliases):
    resolved = {"features": {}, "derived": {}, "event_id": None}
    resolved["event_id"] = resolve_first(branches, event_id_aliases)

    for feature, spec in feature_config.items():
        direct = resolve_first(branches, spec.get("branches", []))
        if direct:
            resolved["features"][feature] = direct
            continue

        if feature == "momentum":
            derive = spec.get("derive_from", {})
            pt = resolve_first(branches, derive.get("pt", []))
            signed_inverse_pt = resolve_first(branches, derive.get("signed_inverse_pt", []))
            tan_lambda = resolve_first(branches, derive.get("tan_lambda", []))
            if tan_lambda and (pt or signed_inverse_pt):
                resolved["derived"]["momentum"] = {
                    "pt": pt,
                    "signed_inverse_pt": signed_inverse_pt,
                    "tan_lambda": tan_lambda,
                }
    return resolved


def feature_score(resolved, feature_config):
    score = 0
    for feature in feature_config:
        if feature in resolved["features"] or feature in resolved["derived"]:
            score += 1
    if resolved["event_id"]:
        score += 1
    return score


def find_best_tree(files, config):
    requested = config["dataset"].get("tree_name")
    features = config["dataset"]["features"]
    event_ids = config["dataset"]["event_id_branches"]

    best = None
    for file_name in files:
        with open_root(file_name) as root_file:
            for tree_path, tree in iter_trees(root_file):
                if requested and requested not in tree_path:
                    continue
                branches = branch_names(tree)
                resolved = resolve_feature_branches(branches, features, event_ids)
                score = feature_score(resolved, features)
                item = {
                    "file": file_name,
                    "tree_path": tree_path,
                    "branches": branches,
                    "resolved": resolved,
                    "score": score,
                    "entries": int(tree.num_entries),
                }
                if best is None or item["score"] > best["score"]:
                    best = item
    return best


def read_tree_arrays(file_name, tree_name, branches):
    with open_root(file_name) as root_file:
        tree = root_file[tree_name]
        arrays = tree.arrays(branches, library="np")
    return {strip_cycle(str(key)): np.asarray(value) for key, value in arrays.items()}
