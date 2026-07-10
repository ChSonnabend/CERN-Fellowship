#!/usr/bin/env python3
import argparse

from _bootstrap import add_project_src

add_project_src()

from softbombs.config import load_config
from softbombs.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/test_config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = build_dataset(config)
    print("Test dataset creation complete")
    print(f"Split counts: {metadata['split_counts']}")


if __name__ == "__main__":
    main()

