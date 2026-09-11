#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-script", required=True)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-config", default="{}")
    return parser.parse_args()


def slurm_key(key):
    return {
        "name": "job-name",
        "memory": "mem",
        "notify": "mail-type",
        "email": "mail-user",
    }.get(key, key)


def main():
    args = parse_args()
    job = json.loads(args.job_config)
    special = job.pop("special-args", {})
    device = special.get("device", "CPU")

    with open(args.config) as f:
        conf = json.load(f)

    if device in ("AMD_GPU", "CUDA"):
        job["gres"] = f"gpu:{int(special.get('ngpus', 1))}"

    lines = ["#!/bin/bash"]
    job["chdir"] = args.submission_dir
    for key, value in job.items():
        if value:
            lines.append(f"#SBATCH --{slurm_key(key)}={value}")

    if device == "CPU":
        interpreter = f"apptainer exec {conf['directory_settings']['rocm_container']} python3"
    elif device == "AMD_GPU":
        interpreter = f"apptainer exec --rocm {conf['directory_settings']['rocm_container']} python3"
    elif device == "CUDA":
        interpreter = f"apptainer exec --nv {conf['directory_settings']['cuda_container']} python3"
    else:
        interpreter = "python3"

    lines += [
        "",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        f"time srun {interpreter} {args.job_script} --config {args.config} --output-dir {Path(args.submission_dir) / 'network'}",
        "",
    ]
    out = Path(args.submission_dir) / "TRAIN.sh"
    out.write_text("\n".join(lines))
    out.chmod(0o755)
    print(out)


if __name__ == "__main__":
    main()
