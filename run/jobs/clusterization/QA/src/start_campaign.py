#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_job_id(text: str) -> Optional[int]:
    import re
    m = re.search(r"Submitted batch job (\d+)", text)
    return int(m.group(1)) if m else None


def submit_controller_job(
    output_root: Path,
    controller_script: Path,
    python_exe: str,
    iteration: int,
) -> int:
    sbatch_path = output_root / f"controller_iter_{iteration:03d}.sbatch"
    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=interp_iter_{iteration:03d}
#SBATCH --output={output_root / "controller_logs" / f"iter_{iteration:03d}_%j.out"}
#SBATCH --error={output_root / "controller_logs" / f"iter_{iteration:03d}_%j.err"}
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

{python_exe} "{controller_script}" \\
  --campaign-root "{output_root}" \\
  --iteration {iteration}
"""
    sbatch_path.write_text(sbatch_text)
    sbatch_path.chmod(0o755)

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    print(text, end="")

    job_id = extract_job_id(text)
    if job_id is None:
        raise RuntimeError(f"Could not extract controller job ID for iteration {iteration}")
    return job_id


def initialize_new_campaign(
    campaign_config_src: Path,
    campaign_json: Dict[str, Any],
    output_root: Path,
) -> Path:
    if output_root.exists():
        shutil.rmtree(output_root)

    mkdir(output_root)
    mkdir(output_root / "iterations")
    mkdir(output_root / "configurations")
    mkdir(output_root / "job_outputs")
    mkdir(output_root / "controller_logs")
    mkdir(output_root / "state")

    campaign_config_dst = output_root / "campaign_config.json"
    shutil.copy2(campaign_config_src, campaign_config_dst)

    state = {
        "campaign_root": str(output_root),
        "campaign_config": str(campaign_config_dst),
        "current_iteration": 0,
        "finished": False,
        "history": []
    }
    dump_json(output_root / "state" / "campaign_state.json", state)

    return campaign_config_dst


def resume_campaign(
    campaign_config_src: Path,
    campaign_json: Dict[str, Any],
    output_root: Path,
    resume_iteration: int,
) -> Path:
    if not output_root.exists():
        raise RuntimeError(
            f"Cannot resume: output_root does not exist: {output_root}"
        )

    state_path = output_root / "state" / "campaign_state.json"
    if not state_path.exists():
        raise RuntimeError(
            f"Cannot resume: missing state file: {state_path}"
        )

    campaign_config_dst = output_root / "campaign_config.json"
    if not campaign_config_dst.exists():
        shutil.copy2(campaign_config_src, campaign_config_dst)

    # Make sure base directories exist
    mkdir(output_root / "iterations")
    mkdir(output_root / "configurations")
    mkdir(output_root / "job_outputs")
    mkdir(output_root / "controller_logs")
    mkdir(output_root / "state")

    state = load_json(state_path)

    if Path(state.get("campaign_root", "")).resolve() != output_root.resolve():
        raise RuntimeError(
            "State file campaign_root does not match requested output_root"
        )

    # Reset campaign status so it can continue
    state["finished"] = False
    state.pop("stop_reason", None)
    state.pop("final_iteration", None)
    state.pop("final_x_opt", None)
    state["current_iteration"] = resume_iteration

    dump_json(state_path, state)
    return campaign_config_dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-config", required=True, help="JSON config for the campaign")
    ap.add_argument(
        "--resume-iteration",
        type=int,
        default=None,
        help=(
            "Resume an existing campaign by submitting controller iteration N. "
            "Example: if iteration 0 completed and iteration 1 failed, use --resume-iteration 1."
        ),
    )
    args = ap.parse_args()

    campaign_config_src = Path(args.campaign_config).expanduser().resolve()
    campaign_json_input = load_json(campaign_config_src)

    output_root = Path(campaign_json_input["output_root"]).expanduser().resolve()

    # If resuming, prefer config from output directory
    if args.resume_iteration is not None:
        campaign_config_dst = output_root / "campaign_config.json"

        if not campaign_config_dst.exists():
            raise RuntimeError(
                f"Cannot resume: missing campaign_config.json in {output_root}"
            )

        campaign_json = load_json(campaign_config_dst)
        print(f"[INFO] Using campaign config from output directory: {campaign_config_dst}")
    else:
        campaign_json = campaign_json_input
    controller_script = Path(campaign_json["controller_script"]).expanduser().resolve()
    python_exe = str(campaign_json.get("python", "python3"))

    if args.resume_iteration is None:
        campaign_config_dst = initialize_new_campaign(
            campaign_config_src=campaign_config_src,
            campaign_json=campaign_json,
            output_root=output_root,
        )
        first_iteration = 0
        print(f"[INFO] Initialized new campaign under: {output_root}")
    else:
        if args.resume_iteration < 0:
            raise ValueError("--resume-iteration must be >= 0")

        campaign_config_dst = resume_campaign(
            campaign_config_src=campaign_config_src,
            campaign_json=campaign_json,
            output_root=output_root,
            resume_iteration=args.resume_iteration,
        )
        first_iteration = args.resume_iteration
        print(f"[INFO] Resuming existing campaign under: {output_root}")
        print(f"[INFO] Resubmitting controller from iteration: {first_iteration}")

    job_id = submit_controller_job(
        output_root=output_root,
        controller_script=controller_script,
        python_exe=python_exe,
        iteration=first_iteration,
    )

    if first_iteration == 0 and args.resume_iteration is None:
        print(f"[INFO] Submitted first controller job: {job_id}")
    else:
        print(f"[INFO] Submitted resumed controller job for iteration {first_iteration}: {job_id}")

    print(f"[INFO] Campaign config in use: {campaign_config_dst}")


if __name__ == "__main__":
    main()