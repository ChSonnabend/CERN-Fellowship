#!/usr/bin/env python3
import argparse
import fnmatch
import posixpath
import shutil
import subprocess
from pathlib import Path

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import ensure_dir, load_config, write_json


def alien_command_available(command):
    return bool(command and shutil.which(command))


def already_in_alien_environment(alien_args):
    if not alien_args:
        return False
    return alien_command_available(alien_args[0])


def run_alien(config, alien_args):
    download = config["download"]
    if already_in_alien_environment(alien_args):
        cmd = alien_args
        cwd = None
    else:
        cmd = ["alienv", "enter", download["alienv_package"], "--"] + alien_args
        cwd = download["o2_workdir"]
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def normalize_alien_path(path):
    return path if path.startswith("alien://") else f"alien://{path}"


def strip_alien_url(path):
    return path.removeprefix("alien://")


def wanted_by_substrings(path, include_substrings=None, exclude_substrings=None):
    include_substrings = include_substrings or []
    exclude_substrings = exclude_substrings or []
    if include_substrings and not any(token in path for token in include_substrings):
        return False
    if exclude_substrings and any(token in path for token in exclude_substrings):
        return False
    return True


def has_wildcards(path):
    return any(char in path for char in "*?[")


def static_prefix_before_wildcard(path):
    path = strip_alien_url(path).rstrip("/")
    parts = path.split("/")
    prefix = []
    for part in parts:
        if has_wildcards(part):
            break
        prefix.append(part)
    out = "/".join(prefix)
    return out if out else "/"


def split_first_wildcard(path):
    path = strip_alien_url(path).rstrip("/")
    parts = path.split("/")
    for index, part in enumerate(parts):
        if has_wildcards(part):
            parent = "/".join(parts[:index]) or "/"
            pattern = part
            suffix = "/".join(parts[index + 1 :])
            return parent, pattern, suffix
    return None, None, None


def alien_ls(config, path):
    result = run_alien(config, ["alien_ls", path])
    return [line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip()]


def expand_alien_paths(config, alien_path):
    alien_path = strip_alien_url(alien_path).rstrip("/")
    if not has_wildcards(alien_path):
        return [alien_path]

    parent, pattern, suffix = split_first_wildcard(alien_path)
    matches = []
    for entry in alien_ls(config, parent):
        name = posixpath.basename(entry)
        if fnmatch.fnmatch(name, pattern):
            candidate = posixpath.join(parent, name)
            if suffix:
                candidate = posixpath.join(candidate, suffix)
            matches.extend(expand_alien_paths(config, candidate))
    return sorted(set(matches))


def discover(config, alien_path):
    download = config["download"]
    patterns = download.get("needed_file_patterns") or [config["input"].get("file_pattern", "AO2D.root")]
    include_substrings = download.get("include_path_substrings", [])
    exclude_substrings = download.get("exclude_path_substrings", config["input"].get("exclude_path_substrings", []))

    files = []
    seen = set()
    search_roots = expand_alien_paths(config, alien_path)
    for search_root in search_roots:
        for pattern in patterns:
            try:
                result = run_alien(config, ["alien_find", search_root, pattern])
            except subprocess.CalledProcessError as exc:
                print(f"  warning: alien_find failed for {search_root} {pattern}: {exc}")
                continue
            for line in result.stdout.splitlines():
                path = strip_alien_url(line.strip())
                if not path or path in seen:
                    continue
                if not wanted_by_substrings(path, include_substrings, exclude_substrings):
                    continue
                files.append(path)
                seen.add(path)
    return sorted(files)


def relative_to_alien_base(alien_file, alien_base):
    alien_file = strip_alien_url(alien_file).rstrip("/")
    alien_base = strip_alien_url(alien_base).rstrip("/")
    if alien_file == alien_base:
        return Path(posixpath.basename(alien_file))
    prefix = alien_base + "/"
    if alien_file.startswith(prefix):
        return Path(alien_file[len(prefix) :])
    return Path(posixpath.basename(alien_file))


def prefer_merged_aod_files(files, alien_base):
    grouped = {}
    passthrough = []
    for alien_file in sorted(files):
        rel = relative_to_alien_base(alien_file, alien_base)
        parts = rel.parts
        if len(parts) >= 2:
            grouped.setdefault(parts[0], []).append(alien_file)
        else:
            passthrough.append(alien_file)

    selected = []
    for _, candidates in sorted(grouped.items()):
        merged = []
        for candidate in candidates:
            rel = relative_to_alien_base(candidate, alien_base)
            if rel.parts and len(rel.parts) == 2 and rel.parts[-1] == "AO2D.root":
                merged.append(candidate)
        if merged:
            selected.extend(sorted(merged)[:1])
        else:
            selected.extend(sorted(candidates))
    if not grouped:
        selected.extend(passthrough)
    return sorted(selected)


def download_one(config, alien_file, local_file):
    local_file = Path(local_file)
    ensure_dir(local_file.parent)
    source = normalize_alien_path(alien_file)
    target = f"file://{local_file}"
    run_alien(config, ["alien_cp", source, target])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-files", action="store_true", help="Only print the files that would be copied")
    args = parser.parse_args()

    config = load_config(args.config)
    dry_run = args.dry_run or bool(config["download"].get("dry_run", False))
    manifest = {}

    for class_config in config["input"]["classes"]:
        files = discover(config, class_config["alien_path"])
        relative_base = class_config.get("alien_relative_base") or static_prefix_before_wildcard(class_config["alien_path"])
        if config["download"].get("prefer_merged_aod_per_job", True):
            files = prefer_merged_aod_files(files, relative_base)
        max_files = class_config.get("max_files")
        if max_files is not None:
            files = files[: int(max_files)]
        manifest[class_config["name"]] = []
        local_dir = Path(class_config["local_dir"])
        ensure_dir(local_dir)
        print(f"{class_config['name']}: {len(files)} files")
        for alien_file in files:
            if config["download"].get("preserve_structure", True):
                rel_path = relative_to_alien_base(alien_file, relative_base)
            else:
                rel_path = Path(alien_file.strip("/").replace("/", "__"))
            local_file = local_dir / rel_path
            manifest[class_config["name"]].append(
                {
                    "alien": alien_file,
                    "local": str(local_file),
                    "relative": str(rel_path),
                }
            )
            if local_file.exists() and not config["download"].get("overwrite", False):
                print(f"  exists, skip: {local_file}")
                continue
            print(f"  {alien_file} -> {local_file}")
            if not dry_run and not args.print_files:
                download_one(config, alien_file, local_file)

    manifest_path = Path(config["project"]["output_dir"]) / "alien_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
