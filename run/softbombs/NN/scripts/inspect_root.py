#!/usr/bin/env python3
import argparse

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import load_config
from softbombs.dataset import discover_files
from softbombs.root_io import branch_names, iter_trees, open_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-branches", type=int, default=120)
    args = parser.parse_args()

    config = load_config(args.config)
    files_by_class = discover_files(config)
    for class_config, files in files_by_class:
        print(f"\n=== {class_config['name']} label={class_config['label']} ===")
        for file_name in files[: args.max_files]:
            print(f"\nFILE {file_name}")
            with open_root(file_name) as root_file:
                for tree_path, tree in iter_trees(root_file):
                    branches = branch_names(tree)
                    print(f"  TREE {tree_path} entries={tree.num_entries} branches={len(branches)}")
                    shown = branches[: args.max_branches]
                    print("    " + ", ".join(shown))
                    if len(branches) > len(shown):
                        print(f"    ... {len(branches) - len(shown)} more")


if __name__ == "__main__":
    main()

