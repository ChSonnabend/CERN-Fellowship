#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        conf = json.load(f)

    day = datetime.now().strftime("%d-%m-%Y")
    training_dir = Path(conf["exec_settings"]["training_dir"])
    out_dir = Path(conf["exec_settings"]["output_folder"]) / day / conf["exec_settings"]["opt_foldername"]
    subprocess.run(
        [
            "python3",
            str(training_dir / "src" / "framework" / "shell_script_creation.py"),
            "--job-script",
            str(out_dir / "train.py"),
            "--submission-dir",
            str(out_dir),
            "--config",
            str(out_dir / "config.json"),
            "--job-config",
            json.dumps(conf["job_settings"]["TRAIN"]),
        ],
        check=True,
    )
    optional_args = conf["job_settings"].get("optional_args", "")
    cmd = f"sbatch --output={out_dir / 'network' / 'job.out'} --error={out_dir / 'network' / 'job.err'} {optional_args} {out_dir / 'TRAIN.sh'}"
    print(subprocess.check_output(cmd, shell=True).decode().strip())


if __name__ == "__main__":
    main()
