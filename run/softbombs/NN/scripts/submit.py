#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from _bootstrap import add_project_src

project_root = add_project_src()

from softbombs.config import ensure_dir, load_config


STAGE_COMMANDS = {
    "download": "python3 -u scripts/download_alien.py --config {config}",
    "dataset": "python3 -u scripts/build_dataset.py --config {config}",
    "test-dataset": "python3 -u scripts/make_test_dataset.py --config {config}",
    "train": "python3 -u scripts/train.py --config {config}",
    "evaluate": "python3 -u scripts/evaluate.py --config {config}",
    "export-onnx": "python3 -u scripts/export_onnx.py --config {config}",
    "qa-plots": "python3 -u scripts/qa_plots.py --config {config}",
}


def effective_slurm_config(config):
    slurm = dict(config["slurm"])
    device = slurm.get("device", "NVIDIA_H200_GPU")

    if device == "NVIDIA_H200_GPU":
        slurm["partition"] = slurm.get("partition") or "nvidia_gpu"
        slurm["constraint"] = slurm.get("constraint") or "h200"
        slurm["container"] = slurm.get("cuda_container") or slurm.get("container")
        slurm["apptainer_gpu_flag"] = "--nv"
    elif device == "AMD_MI100_GPU":
        slurm["partition"] = "amd_gpu"
        slurm["constraint"] = "mi100"
        slurm["container"] = slurm.get("rocm_container") or slurm.get("container")
        slurm["apptainer_gpu_flag"] = ""
    elif device == "CPU":
        slurm["partition"] = slurm.get("cpu_partition") or "debug"
        slurm["constraint"] = slurm.get("cpu_constraint")
        slurm["gres"] = None
        slurm["container"] = slurm.get("cpu_container") or slurm.get("cuda_container") or slurm.get("container")
        slurm["apptainer_gpu_flag"] = ""
    elif device == "EPN":
        slurm["partition"] = slurm.get("epn_partition") or "prod"
        slurm["constraint"] = slurm.get("epn_constraint")
        slurm["container"] = None
        slurm["apptainer_gpu_flag"] = ""
        slurm["python"] = slurm.get("python") or "python3.9"
        slurm["gres"] = slurm.get("gres") or f"gpu:{int(slurm.get('ngpus', 1))}"
        slurm["nodes"] = int(slurm.get("nodes", 1))
        slurm["ntasks_per_node"] = int(slurm.get("ntasks_per_node", slurm.get("ngpus", 1)))
        slurm["use_srun"] = slurm.get("use_srun", True)
    else:
        raise ValueError(f"Unknown slurm.device: {device}")

    return slurm


def runtime_setup_lines(config, slurm, stage, stage_label):
    project_output = Path(config["project"]["output_dir"]).resolve()
    cache_base = Path(slurm.get("cache_dir") or Path(config["project"]["output_dir"]) / "runtime")
    cache_base = cache_base.resolve()
    lines = [
        'export SOFTBOMB_JOB_STAMP="$(date +%Y%m%d)_${SLURM_JOB_ID:-local_$$}"',
        f'export SOFTBOMB_JOB_OUTPUT_DIR="{project_output}/jobs/${{SOFTBOMB_JOB_STAMP}}"',
        f'export SOFTBOMB_RUNTIME_ROOT="{cache_base}"',
        f'export SOFTBOMB_JOB_RUNTIME="{cache_base}/{stage_label}_${{SLURM_JOB_ID:-local}}"',
        'export SOFTBOMB_SHORT_TMP="/tmp/softbomb_${USER:-user}_${SLURM_JOB_ID:-local}"',
        'mkdir -p "$SOFTBOMB_JOB_OUTPUT_DIR"',
        'mkdir -p "$SOFTBOMB_JOB_OUTPUT_DIR/slurm"',
        f'export SOFTBOMB_JOB_STDOUT="$SOFTBOMB_JOB_OUTPUT_DIR/slurm/{stage_label}_${{SLURM_JOB_ID:-local}}.out"',
        f'export SOFTBOMB_JOB_STDERR="$SOFTBOMB_JOB_OUTPUT_DIR/slurm/{stage_label}_${{SLURM_JOB_ID:-local}}.err"',
        'echo "Redirecting stdout to $SOFTBOMB_JOB_STDOUT"',
        'echo "Redirecting stderr to $SOFTBOMB_JOB_STDERR"',
        'exec > "$SOFTBOMB_JOB_STDOUT"',
        'exec 2> "$SOFTBOMB_JOB_STDERR"',
        'echo "Job output directory: $SOFTBOMB_JOB_OUTPUT_DIR"',
        'echo "Job stdout: $SOFTBOMB_JOB_STDOUT"',
        'echo "Job stderr: $SOFTBOMB_JOB_STDERR"',
        'mkdir -p "$SOFTBOMB_JOB_RUNTIME"/{tmp,apptainer_cache,xdg_cache,torch,triton,miopen,rocm,comgr,pycache}',
        'mkdir -p "$SOFTBOMB_SHORT_TMP"',
        'cleanup_softbomb_runtime() {',
        '  local status=$?',
        '  trap - EXIT INT TERM',
        f'  if [[ -n "${{SOFTBOMB_JOB_RUNTIME:-}}" && -n "${{SOFTBOMB_RUNTIME_ROOT:-}}" && "$SOFTBOMB_JOB_RUNTIME" == "$SOFTBOMB_RUNTIME_ROOT"/{stage_label}_* ]]; then',
        '    echo "Cleaning runtime directory: $SOFTBOMB_JOB_RUNTIME"',
        '    rm -rf -- "$SOFTBOMB_JOB_RUNTIME"',
        '  else',
        '    echo "Skipping runtime cleanup because path guard did not match: ${SOFTBOMB_JOB_RUNTIME:-unset}"',
        '  fi',
        '  if [[ -n "${SOFTBOMB_SHORT_TMP:-}" && "$SOFTBOMB_SHORT_TMP" == /tmp/softbomb_* ]]; then',
        '    echo "Cleaning temporary directory: $SOFTBOMB_SHORT_TMP"',
        '    rm -rf -- "$SOFTBOMB_SHORT_TMP"',
        '  else',
        '    echo "Skipping temporary cleanup because path guard did not match: ${SOFTBOMB_SHORT_TMP:-unset}"',
        '  fi',
        '  return "$status"',
        '}',
        'trap cleanup_softbomb_runtime EXIT',
        "trap 'exit 130' INT",
        "trap 'exit 143' TERM",
        'export TMPDIR="$SOFTBOMB_SHORT_TMP"',
        'export TEMP="$TMPDIR"',
        'export TMP="$TMPDIR"',
        'export APPTAINER_TMPDIR="$SOFTBOMB_JOB_RUNTIME/tmp"',
        'export APPTAINER_CACHEDIR="$SOFTBOMB_JOB_RUNTIME/apptainer_cache"',
        'export SINGULARITY_TMPDIR="$SOFTBOMB_JOB_RUNTIME/tmp"',
        'export SINGULARITY_CACHEDIR="$SOFTBOMB_JOB_RUNTIME/apptainer_cache"',
        'export XDG_CACHE_HOME="$SOFTBOMB_JOB_RUNTIME/xdg_cache"',
        'export TORCH_HOME="$SOFTBOMB_JOB_RUNTIME/torch"',
        'export TRITON_CACHE_DIR="$SOFTBOMB_JOB_RUNTIME/triton"',
        'export MIOPEN_USER_DB_PATH="$SOFTBOMB_JOB_RUNTIME/miopen"',
        'export MIOPEN_CUSTOM_CACHE_DIR="$SOFTBOMB_JOB_RUNTIME/miopen"',
        'export ROCM_CACHE_DIR="$SOFTBOMB_JOB_RUNTIME/rocm"',
        'export HIP_CACHE_DIR="$SOFTBOMB_JOB_RUNTIME/rocm"',
        'export AMD_COMGR_CACHE_DIR="$SOFTBOMB_JOB_RUNTIME/comgr"',
        'export PYTHONPYCACHEPREFIX="$SOFTBOMB_JOB_RUNTIME/pycache"',
        'export APPTAINERENV_TMPDIR="$TMPDIR"',
        'export APPTAINERENV_TEMP="$TEMP"',
        'export APPTAINERENV_TMP="$TMP"',
        'export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"',
        'export APPTAINERENV_TORCH_HOME="$TORCH_HOME"',
        'export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"',
        'export APPTAINERENV_MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"',
        'export APPTAINERENV_MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR"',
        'export APPTAINERENV_ROCM_CACHE_DIR="$ROCM_CACHE_DIR"',
        'export APPTAINERENV_HIP_CACHE_DIR="$HIP_CACHE_DIR"',
        'export APPTAINERENV_AMD_COMGR_CACHE_DIR="$AMD_COMGR_CACHE_DIR"',
        'export APPTAINERENV_PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX"',
        'export SINGULARITYENV_TMPDIR="$TMPDIR"',
        'export SINGULARITYENV_TEMP="$TEMP"',
        'export SINGULARITYENV_TMP="$TMP"',
        'export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"',
        'export SINGULARITYENV_TORCH_HOME="$TORCH_HOME"',
        'export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"',
        'export SINGULARITYENV_MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"',
        'export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR"',
        'export SINGULARITYENV_ROCM_CACHE_DIR="$ROCM_CACHE_DIR"',
        'export SINGULARITYENV_HIP_CACHE_DIR="$HIP_CACHE_DIR"',
        'export SINGULARITYENV_AMD_COMGR_CACHE_DIR="$AMD_COMGR_CACHE_DIR"',
        'export SINGULARITYENV_PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX"',
    ]
    if stage == "train":
        lines.extend(
            [
                'export SOFTBOMB_TRAINING_OUTPUT_DIR="$SOFTBOMB_JOB_OUTPUT_DIR/training"',
                'mkdir -p "$SOFTBOMB_TRAINING_OUTPUT_DIR"',
            ]
        )
    elif stage in {"dataset", "test-dataset"} and slurm.get("isolate_dataset_output", False):
        lines.extend(
            [
                'export SOFTBOMB_DATASET_OUTPUT_DIR="$SOFTBOMB_JOB_OUTPUT_DIR/data"',
                'mkdir -p "$SOFTBOMB_DATASET_OUTPUT_DIR"',
            ]
        )
    return lines


def wrapped_command(config, command):
    slurm = effective_slurm_config(config)
    if slurm.get("device") == "EPN":
        python_cmd = slurm.get("python", "python3.9")
        if command.startswith("python3 "):
            command = command.replace("python3 ", f"{python_cmd} ", 1)
        elif command.startswith("python3 -u "):
            command = command.replace("python3 -u ", f"{python_cmd} -u ", 1)
        return f"srun {command}" if slurm.get("use_srun", True) else command

    container = slurm.get("container")
    if not container:
        return command
    runtime = slurm.get("container_runtime", "apptainer")
    gpu_flag = slurm.get("apptainer_gpu_flag", "")
    flag_part = f" {gpu_flag}" if gpu_flag else ""
    return f"{runtime} exec{flag_part} {container} {command}"


def build_script(config, stage, command):
    slurm = effective_slurm_config(config)
    if slurm.get("device") == "AMD_MI100_GPU":
        stage_label = f"{stage}_amd"
    elif slurm.get("device") == "EPN":
        stage_label = f"{stage}_epn"
    elif slurm.get("device") == "CPU":
        stage_label = f"{stage}_cpu"
    else:
        stage_label = stage
    output_dir = Path(config["project"]["output_dir"]) / "slurm"
    ensure_dir(output_dir)
    if slurm.get("keep_central_slurm_logs", False):
        slurm_stdout = f"{output_dir}/{stage_label}_%j.out"
        slurm_stderr = f"{output_dir}/{stage_label}_%j.err"
    else:
        slurm_stdout = "/dev/null"
        slurm_stderr = "/dev/null"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=softbomb_{stage_label}",
        f"#SBATCH --chdir={project_root}",
        f"#SBATCH --time={slurm['time']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --output={slurm_stdout}",
        f"#SBATCH --error={slurm_stderr}",
    ]
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("constraint"):
        lines.append(f"#SBATCH --constraint={slurm['constraint']}")
    if slurm.get("gres"):
        lines.append(f"#SBATCH --gres={slurm['gres']}")
    if slurm.get("nodes"):
        lines.append(f"#SBATCH --nodes={slurm['nodes']}")
    if slurm.get("ntasks_per_node"):
        lines.append(f"#SBATCH --ntasks-per-node={slurm['ntasks_per_node']}")
    if slurm.get("mail_type"):
        lines.append(f"#SBATCH --mail-type={slurm['mail_type']}")
    if slurm.get("mail_user"):
        lines.append(f"#SBATCH --mail-user={slurm['mail_user']}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {project_root}",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        ]
    )
    lines.extend(slurm.get("setup_commands", []))
    lines.extend(runtime_setup_lines(config, slurm, stage, stage_label))
    lines.extend(["", wrapped_command(config, command), ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGE_COMMANDS))
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    jobs_dir = Path(config["slurm"]["jobs_dir"])
    ensure_dir(jobs_dir)
    command = STAGE_COMMANDS[args.stage].format(config=str(Path(args.config).resolve()))
    script = build_script(config, args.stage, command)
    slurm = effective_slurm_config(config)
    if slurm.get("device") == "AMD_MI100_GPU":
        stage_label = f"{args.stage}_amd"
    elif slurm.get("device") == "EPN":
        stage_label = f"{args.stage}_epn"
    elif slurm.get("device") == "CPU":
        stage_label = f"{args.stage}_cpu"
    else:
        stage_label = args.stage
    script_path = jobs_dir / f"{stage_label}.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    print(f"Wrote {script_path}")
    if args.dry_run:
        print(script)
        return
    result = subprocess.check_output(["sbatch", str(script_path)], text=True)
    print(result.strip())


if __name__ == "__main__":
    main()
