#!/usr/bin/env python3
import argparse

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import load_config
from softbombs.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = build_dataset(config)
    print("Dataset creation complete")
    print(f"Tree: {metadata['tree_path']}")
    print(f"Split counts: {metadata['split_counts']}")
    print(f"Events per class after balance: {metadata['events_per_class_after_balance']}")


if __name__ == "__main__":
    main()

