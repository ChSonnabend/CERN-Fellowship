#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("-s", "--skip-q", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        conf = json.load(f)

    day = datetime.now().strftime("%d-%m-%Y")
    out_dir = Path(conf["exec_settings"]["output_folder"]) / day / conf["exec_settings"]["opt_foldername"]
    if out_dir.exists() and not args.skip_q:
        response = input(f"Jobs directory ({out_dir}) exists. Overwrite it? (y/n) ")
        if response != "y":
            print("Stopping.")
            return
    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "network").mkdir(parents=True)
    (out_dir / "QA").mkdir()
    shutil.copy2(args.config, out_dir / "config.json")
    shutil.copy2(conf["network_settings"]["training_file"], out_dir / "train.py")
    shutil.copy2(conf["network_settings"]["configurations_file"], out_dir / "configurations.py")

    copied_config = out_dir / "config.json"
    with copied_config.open() as f:
        local_conf = json.load(f)
    local_conf["network_settings"]["configurations_file"] = str(out_dir / "configurations.py")
    local_conf["network_settings"]["training_file"] = str(out_dir / "train.py")
    with copied_config.open("w") as f:
        json.dump(local_conf, f, indent=2)

    print(out_dir)


if __name__ == "__main__":
    main()
