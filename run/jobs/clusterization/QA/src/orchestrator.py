#!/usr/bin/env python3

import argparse
import copy
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional


GPU_PROC_NN_PATH = ["reco_task", "input-digits", "configKeyValues", "GPU_proc_nn"]
JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


# ============================================================
# Generic helpers
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def deep_get(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def deep_set(d: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def sanitize_float_for_name(x: float) -> str:
    return f"{x:.12g}".replace(".", "p").replace("-", "m")


def extract_job_ids(text: str) -> List[int]:
    return [int(x) for x in JOB_ID_RE.findall(text)]


# ============================================================
# Parsing helpers
# ============================================================

def parse_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value]
    raise TypeError(f"Expected string or list, got {type(value)}")


def parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    raise TypeError(f"Expected string or list, got {type(value)}")


def parse_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(f"Expected string or list, got {type(value)}")


def parse_boolish_list(value: Any) -> List[bool]:
    if value is None:
        return []
    if isinstance(value, str):
        return [bool(int(x.strip())) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            if isinstance(x, bool):
                out.append(x)
            elif isinstance(x, int):
                out.append(bool(x))
            elif isinstance(x, str):
                xl = x.strip().lower()
                if xl in ("1", "true", "yes", "y"):
                    out.append(True)
                elif xl in ("0", "false", "no", "n"):
                    out.append(False)
                else:
                    raise ValueError(f"Cannot interpret boolean value '{x}'")
            else:
                raise TypeError(f"Unsupported boolean-like value type: {type(x)}")
        return out
    raise TypeError(f"Expected string or list, got {type(value)}")


def parse_bounds(value: Any, ndim: Optional[int] = None) -> List[List[float]]:
    if isinstance(value, str):
        flat = [float(x.strip()) for x in value.split(",") if x.strip()]
        if len(flat) % 2 != 0:
            raise ValueError("Bounds string must contain an even number of values")
        arr = [flat[i:i+2] for i in range(0, len(flat), 2)]
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            arr = []
        elif isinstance(value[0], (list, tuple)):
            arr = [[float(x[0]), float(x[1])] for x in value]
        else:
            flat = [float(x) for x in value]
            if len(flat) % 2 != 0:
                raise ValueError("Flat bounds list must contain an even number of values")
            arr = [flat[i:i+2] for i in range(0, len(flat), 2)]
    else:
        raise TypeError(f"Expected bounds as string or list, got {type(value)}")

    if ndim is not None and len(arr) != ndim:
        raise ValueError(f"Expected {ndim} bounds rows, got {len(arr)}")
    return arr


def parse_scales(value: Any, ndim: Optional[int] = None) -> List[str]:
    scales = parse_str_list(value)
    valid = {"linear", "log"}
    for s in scales:
        if s not in valid:
            raise ValueError(f"Unknown scale '{s}'")
    if ndim is not None and len(scales) != ndim:
        raise ValueError(f"Expected {ndim} scales, got {len(scales)}")
    return scales


# ============================================================
# Sampling helpers without numpy
# ============================================================

def build_axis(vmin: float, vmax: float, n: int, scale: str) -> List[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return [float(vmin)]

    if scale == "linear":
        step = (vmax - vmin) / (n - 1)
        return [vmin + i * step for i in range(n)]

    if scale == "log":
        if vmin <= 0 or vmax <= 0:
            raise ValueError("Log scale requires positive bounds")
        lo = math.log10(vmin)
        hi = math.log10(vmax)
        step = (hi - lo) / (n - 1)
        return [10 ** (lo + i * step) for i in range(n)]

    raise ValueError(f"Unknown scale: {scale}")


def build_cartesian_grid(
    bounds: Sequence[Sequence[float]],
    counts: Sequence[int],
    scales: Sequence[str],
) -> List[List[float]]:
    axes = [build_axis(bounds[i][0], bounds[i][1], counts[i], scales[i]) for i in range(len(counts))]
    return [list(p) for p in itertools.product(*axes)]


def lhs_in_box(
    bounds: Sequence[Sequence[float]],
    n_samples: int,
    scales: Sequence[str],
    seed: int = 42,
) -> List[List[float]]:
    import random

    d = len(bounds)
    if len(scales) != d:
        raise ValueError("scales length must match dimensionality")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = random.Random(seed)
    samples = [[0.0 for _ in range(d)] for _ in range(n_samples)]

    for i in range(d):
        low = float(bounds[i][0])
        high = float(bounds[i][1])
        if high <= low:
            raise ValueError(f"Invalid bounds in dim {i}: low={low}, high={high}")

        strata = [(j + rng.random()) / n_samples for j in range(n_samples)]
        rng.shuffle(strata)

        for j, u in enumerate(strata):
            if scales[i] == "log":
                if low <= 0 or high <= 0:
                    raise ValueError("Log-space LHS needs positive bounds")
                lo = math.log10(low)
                hi = math.log10(high)
                samples[j][i] = 10 ** (lo + u * (hi - lo))
            elif scales[i] == "linear":
                samples[j][i] = low + u * (high - low)
            else:
                raise ValueError(f"Unknown scale: {scales[i]}")

    return samples


def lhs_around_center(
    center: Sequence[float],
    half_widths: Sequence[float],
    n_samples: int,
    global_bounds: Sequence[Sequence[float]],
    log_space: Sequence[bool],
    seed: int = 42,
    include_center: bool = True,
) -> List[List[float]]:
    import random

    center = [float(x) for x in center]
    half_widths = [float(x) for x in half_widths]
    d = len(center)

    if len(log_space) != d:
        raise ValueError("log_space length must match dimensionality")

    # low = [max(center[i] - half_widths[i], global_bounds[i][0]) for i in range(d)]
    # high = [min(center[i] + half_widths[i], global_bounds[i][1]) for i in range(d)]

    low = [center[i] - half_widths[i] for i in range(d)]
    high = [center[i] + half_widths[i] for i in range(d)]

    low = [max(low[i], global_bounds[i][0]) for i in range(d)]
    high = [min(high[i], global_bounds[i][1]) for i in range(d)]

    for i in range(d):
        if high[i] <= low[i]:
            raise ValueError(f"Invalid LHS box in dim {i}: low={low[i]}, high={high[i]}")

    rng = random.Random(seed)
    samples = [[0.0 for _ in range(d)] for _ in range(n_samples)]

    for i in range(d):
        strata = [(j + rng.random()) / n_samples for j in range(n_samples)]
        rng.shuffle(strata)

        for j in range(n_samples):
            u = strata[j]
            if log_space[i]:
                if low[i] <= 0 or high[i] <= 0:
                    raise ValueError("Log-space LHS needs positive bounds")
                lo = math.log10(low[i])
                hi = math.log10(high[i])
                samples[j][i] = 10 ** (lo + u * (hi - lo))
            else:
                samples[j][i] = low[i] + u * (high[i] - low[i])

    if include_center:
        return [center] + samples
    return samples


# ============================================================
# Config generation
# ============================================================

def set_scan_values_in_config(cfg: Dict[str, Any], scan_params: Sequence[str], point: Sequence[float]) -> None:
    gpu_cfg = deep_get(cfg, GPU_PROC_NN_PATH)
    for p, v in zip(scan_params, point):
        gpu_cfg[p] = float(v)


def generate_configs_for_points(
    points: Sequence[Sequence[float]],
    base_config: Dict[str, Any],
    configs_dir: Path,
    output_base_dir: Path,
    scan_params: Sequence[str],
    mode: int,
    nn_model_path: str,
) -> Tuple[List[Path], List[Path]]:
    mkdir(configs_dir)
    mkdir(output_base_dir)

    cfg_paths = []
    output_dirs = []

    for point in points:
        cfg = copy.deepcopy(base_config)
        set_scan_values_in_config(cfg, scan_params, point)

        gpu_cfg = deep_get(cfg, GPU_PROC_NN_PATH)
        gpu_cfg["nnUseClusterErrorNetwork"] = mode
        gpu_cfg["nnClusterErrorModelPath"] = nn_model_path

        leaf = "__".join(
            f"{name}_{sanitize_float_for_name(float(val))}"
            for name, val in zip(scan_params, point)
        )

        this_cfg_dir = configs_dir / leaf
        this_output_dir = output_base_dir / leaf
        mkdir(this_cfg_dir)
        mkdir(this_output_dir)

        deep_set(cfg, ["exec_settings", "output_dir"], str(this_output_dir))

        cfg_path = this_cfg_dir / "config_qa.json"
        dump_json(cfg_path, cfg)
        shutil.copy2(cfg_path, this_output_dir / "config_qa.json")

        cfg_paths.append(cfg_path)
        output_dirs.append(this_output_dir)

    return cfg_paths, output_dirs


def prepare_submission_config_copy(
    submission_template: Dict[str, Any],
    destination: Path,
    new_configurations_dir: Path,
) -> Path:
    cfg = copy.deepcopy(submission_template)
    cfg["submission"]["configurations_dir"] = str(new_configurations_dir)
    dump_json(destination, cfg)
    return destination


# ============================================================
# SLURM helpers
# ============================================================

def run_capture(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    ret = proc.wait()
    return ret, "".join(lines)


def submit_iteration_jobs(
    create_jobs_script: Path,
    run_jobs_script: Path,
    submission_cfg_copy: Path,
) -> List[int]:
    ret1, _ = run_capture([
        "python3",
        str(create_jobs_script),
        "--avoid-question", "1",
        "--config", str(submission_cfg_copy),
    ])
    if ret1 != 0:
        raise RuntimeError("create_jobs.py failed")

    ret2, out2 = run_capture([
        "python3",
        str(run_jobs_script),
        "--config", str(submission_cfg_copy),
        "--submit", "1",
        "--options", "reco,combine",
    ])
    if ret2 != 0:
        raise RuntimeError("run_jobs.py failed")

    return extract_job_ids(out2)


WATCHDOG_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import json
import subprocess
import time

TERMINAL_PREFIXES = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
)

def squeue_state(job_id: int):
    res = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T|%M"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if res.returncode != 0:
        return None
    line = res.stdout.strip()
    if not line:
        return None
    state, elapsed = line.split("|", 1)
    return state.strip(), elapsed.strip()

def sacct_terminal_state(job_id: int):
    res = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=JobIDRaw,State", "-P", "-n"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 2 and fields[0] == str(job_id):
            return fields[1].strip()
    return None

def elapsed_to_seconds(s: str) -> int:
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, sec = parts
    else:
        raise ValueError(f"Unsupported elapsed format: {s}")
    return days * 86400 + h * 3600 + m * 60 + sec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-ids-file", required=True)
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--max-runtime-seconds", type=int, default=3600)
    ap.add_argument("--max-requeues", type=int, default=3)
    args = ap.parse_args()

    with open(args.job_ids_file, "r") as f:
        job_ids = json.load(f)

    try:
        with open(args.state_file, "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        state = {str(j): {"requeues": 0} for j in job_ids}

    while True:
        all_done = True

        for jid in job_ids:
            s_jid = str(jid)
            if s_jid not in state:
                state[s_jid] = {"requeues": 0}

            sq = squeue_state(jid)

            if sq is not None:
                all_done = False
                status, elapsed = sq
                if status == "RUNNING":
                    runtime_sec = elapsed_to_seconds(elapsed)
                    if runtime_sec > args.max_runtime_seconds:
                        if state[s_jid]["requeues"] >= args.max_requeues:
                            subprocess.run(["scancel", str(jid)], check=False)
                        else:
                            subprocess.run(["scontrol", "requeue", str(jid)], check=False)
                            state[s_jid]["requeues"] += 1

                with open(args.state_file, "w") as f:
                    json.dump(state, f, indent=2)
                continue

            st = sacct_terminal_state(jid)
            if st is None or not st.startswith(TERMINAL_PREFIXES):
                all_done = False

        if all_done:
            with open(args.state_file, "w") as f:
                json.dump(state, f, indent=2)
            break

        time.sleep(args.poll_seconds)

if __name__ == "__main__":
    main()
'''


def write_and_submit_watchdog(
    iteration_dir: Path,
    job_ids: Sequence[int],
    poll_seconds: int,
    max_runtime_seconds: int,
    max_requeues: int,
) -> Optional[int]:
    py_path = iteration_dir / "watchdog.py"
    sbatch_path = iteration_dir / "watchdog.sbatch"
    job_ids_path = iteration_dir / "job_ids.json"
    state_path = iteration_dir / "watchdog_state.json"

    py_path.write_text(WATCHDOG_SCRIPT)
    py_path.chmod(0o755)

    with open(job_ids_path, "w") as f:
        json.dump([int(x) for x in job_ids], f, indent=2)

    sbatch_path.write_text(f"""#!/bin/bash
#SBATCH --job-name=watchdog_{iteration_dir.name}
#SBATCH --output={iteration_dir / "watchdog_%j.out"}
#SBATCH --error={iteration_dir / "watchdog_%j.err"}
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

python3 "{py_path}" \\
  --job-ids-file "{job_ids_path}" \\
  --state-file "{state_path}" \\
  --poll-seconds {poll_seconds} \\
  --max-runtime-seconds {max_runtime_seconds} \\
  --max-requeues {max_requeues}
""")
    sbatch_path.chmod(0o755)

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    print(text, end="")
    ids = extract_job_ids(text)
    return ids[0] if ids else None


def submit_next_controller(
    campaign_root: Path,
    controller_script: Path,
    python_exe: str,
    next_iteration: int,
    dependency_job_id: Optional[int],
) -> Optional[int]:
    logs_dir = campaign_root / "controller_logs"
    mkdir(logs_dir)

    sbatch_path = campaign_root / f"controller_iter_{next_iteration:03d}.sbatch"

    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=interp_iter_{next_iteration:03d}
#SBATCH --output={logs_dir / f"iter_{next_iteration:03d}_%j.out"}
#SBATCH --error={logs_dir / f"iter_{next_iteration:03d}_%j.err"}
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

{python_exe} "{controller_script}" \\
  --campaign-root "{campaign_root}" \\
  --iteration {next_iteration}
"""
    sbatch_path.write_text(sbatch_text)
    sbatch_path.chmod(0o755)

    cmd = ["sbatch"]
    if dependency_job_id is not None:
        cmd.extend(["--dependency", f"afterany:{dependency_job_id}"])
    cmd.append(str(sbatch_path))

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    print(text, end="")
    ids = extract_job_ids(text)
    return ids[0] if ids else None


# ============================================================
# Campaign state
# ============================================================

def load_campaign(campaign_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    campaign_cfg = load_json(campaign_root / "campaign_config.json")
    state = load_json(campaign_root / "state" / "campaign_state.json")
    return campaign_cfg, state


def save_state(campaign_root: Path, state: Dict[str, Any]) -> None:
    dump_json(campaign_root / "state" / "campaign_state.json", state)


# ============================================================
# Apptainer inner-fit launcher
# ============================================================

def run_inner_fit_in_apptainer(
    campaign_root: Path,
    iteration: int,
    image: str,
    inner_script: Path,
    bind_paths: Sequence[str],
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cmd = ["apptainer", "exec"]

    for b in bind_paths:
        cmd.extend(["--bind", b])

    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    else:
        env = None

    cmd.extend([
        image,
        "python3",
        str(inner_script),
        "--campaign-root",
        str(campaign_root),
        "--iteration",
        str(iteration),
    ])

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
    )
    print(res.stdout, end="")

    if res.returncode != 0:
        raise RuntimeError(f"inner_fit.py failed in apptainer with exit code {res.returncode}")

    prev_iteration_dir = campaign_root / "iterations" / f"iter_{iteration-1:03d}"
    return load_json(prev_iteration_dir / "interpolation_summary.json")


# ============================================================
# Main iteration logic
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    campaign_root = Path(args.campaign_root).expanduser().resolve()
    iteration = args.iteration

    campaign_cfg, state = load_campaign(campaign_root)

    scan_params = parse_str_list(campaign_cfg["scan_params"])
    ndim = len(scan_params)

    lhs_bounds = parse_bounds(campaign_cfg["lhs_bounds"], ndim=ndim)
    initial_half_widths = parse_float_list(campaign_cfg["initial_half_widths"])
    lhs_log_space = parse_boolish_list(campaign_cfg.get("lhs_log_space", [False] * ndim))

    if len(initial_half_widths) != ndim:
        raise ValueError("initial_half_widths length must match scan_params")
    if len(lhs_log_space) != ndim:
        raise ValueError("lhs_log_space length must match scan_params")

    base_config_template = Path(campaign_cfg["base_config_template"]).expanduser().resolve()
    submission_config_template = Path(campaign_cfg["submission_config_template"]).expanduser().resolve()
    create_jobs_script = Path(campaign_cfg["create_jobs_script"]).expanduser().resolve()
    run_jobs_script = Path(campaign_cfg["run_jobs_script"]).expanduser().resolve()
    controller_script = Path(campaign_cfg["controller_script"]).expanduser().resolve()
    python_exe = str(campaign_cfg.get("python", "python3"))

    mode = int(campaign_cfg["mode"])
    nn_model_path = str(campaign_cfg["nn_model_path"])

    initial_lhs_samples = int(campaign_cfg.get("initial_lhs_samples", campaign_cfg.get("lhs_samples", 80)))
    initial_lhs_seed = int(campaign_cfg.get("initial_lhs_seed", campaign_cfg.get("lhs_seed", 42)))

    lhs_samples = int(campaign_cfg.get("lhs_samples", 80))
    lhs_seed = int(campaign_cfg.get("lhs_seed", 42))
    include_center = bool(campaign_cfg.get("include_center", 1))
    max_iterations = int(campaign_cfg.get("max_iterations", 20))

    watchdog_poll_seconds = int(campaign_cfg.get("watchdog_poll_seconds", 120))
    watchdog_max_runtime_seconds = int(campaign_cfg.get("watchdog_max_runtime_seconds", 3600))
    watchdog_max_requeues = int(campaign_cfg.get("watchdog_max_requeues", 3))

    apptainer_image = str(campaign_cfg["apptainer_image"])
    inner_fit_script = Path(campaign_cfg["inner_fit_script"]).expanduser().resolve()
    apptainer_bind_paths = parse_str_list(campaign_cfg.get("apptainer_bind_paths", []))

    base_config = load_json(base_config_template)
    submission_template = load_json(submission_config_template)

    iteration_dir = campaign_root / "iterations" / f"iter_{iteration:03d}"
    iteration_configs_dir = campaign_root / "configurations" / f"iter_{iteration:03d}"
    iteration_outputs_dir = campaign_root / "job_outputs" / f"iter_{iteration:03d}"
    mkdir(iteration_dir)
    mkdir(iteration_configs_dir)
    mkdir(iteration_outputs_dir)

    prev_fit = None
    if iteration > 0:
        prev_fit = run_inner_fit_in_apptainer(
            campaign_root=campaign_root,
            iteration=iteration,
            image=apptainer_image,
            inner_script=inner_fit_script,
            bind_paths=apptainer_bind_paths,
        )

        if prev_fit["converged"]:
            state["finished"] = True
            state["stop_reason"] = "converged"
            state["final_iteration"] = iteration - 1
            state["final_x_opt"] = prev_fit["x_opt"]
            state["current_iteration"] = iteration
            save_state(campaign_root, state)
            print(f"[INFO] Convergence reached at iteration {iteration - 1}. Campaign finished.")
            return

    if iteration >= max_iterations:
        state["finished"] = True
        state["stop_reason"] = "max_iterations_reached"
        state["final_iteration"] = iteration - 1 if iteration > 0 else None
        if prev_fit is not None:
            state["final_x_opt"] = prev_fit.get("x_opt")
        state["current_iteration"] = iteration
        save_state(campaign_root, state)
        print("[INFO] Max iterations reached. Stopping.")
        return

    if iteration == 0:
        initial_grid_scales = []
        for i in range(ndim):
            if lhs_log_space[i]:
                initial_grid_scales.append("log")
            else:
                initial_grid_scales.append("linear")
        points = lhs_in_box(
            bounds=lhs_bounds,
            n_samples=initial_lhs_samples,
            scales=initial_grid_scales,
            seed=initial_lhs_seed,
        )
        current_center = None
        current_half_widths = initial_half_widths
        source_iteration = None
    else:
        assert prev_fit is not None
        x_opt = prev_fit["x_opt"]
        next_half_widths = prev_fit["next_half_widths"]

        points = lhs_around_center(
            center=x_opt,
            half_widths=next_half_widths,
            n_samples=lhs_samples,
            global_bounds=lhs_bounds,
            log_space=lhs_log_space,
            seed=lhs_seed + iteration,
            include_center=include_center,
        )
        current_center = x_opt
        current_half_widths = next_half_widths
        source_iteration = iteration - 1

    dump_json(
        iteration_dir / "sampling_plan.json",
        {
            "iteration": iteration,
            "scan_params": scan_params,
            "source_iteration": source_iteration,
            "center": current_center,
            "half_widths": current_half_widths,
            "points": points,
        },
    )

    cfg_paths, output_dirs = generate_configs_for_points(
        points=points,
        base_config=base_config,
        configs_dir=iteration_configs_dir,
        output_base_dir=iteration_outputs_dir,
        scan_params=scan_params,
        mode=mode,
        nn_model_path=nn_model_path,
    )

    sampling_data = load_json(iteration_dir / "sampling_plan.json")
    sampling_data["generated_config_paths"] = [str(p) for p in cfg_paths]
    sampling_data["generated_output_dirs"] = [str(p) for p in output_dirs]
    dump_json(iteration_dir / "sampling_plan.json", sampling_data)

    submission_cfg_copy = prepare_submission_config_copy(
        submission_template=submission_template,
        destination=iteration_dir / "config.json",
        new_configurations_dir=iteration_configs_dir,
    )

    job_ids = submit_iteration_jobs(
        create_jobs_script=create_jobs_script,
        run_jobs_script=run_jobs_script,
        submission_cfg_copy=submission_cfg_copy,
    )

    watchdog_job_id = None
    if job_ids:
        watchdog_job_id = write_and_submit_watchdog(
            iteration_dir=iteration_dir,
            job_ids=job_ids,
            poll_seconds=watchdog_poll_seconds,
            max_runtime_seconds=watchdog_max_runtime_seconds,
            max_requeues=watchdog_max_requeues,
        )

    next_controller_job_id = submit_next_controller(
        campaign_root=campaign_root,
        controller_script=controller_script,
        python_exe=python_exe,
        next_iteration=iteration + 1,
        dependency_job_id=watchdog_job_id,
    )

    state["current_iteration"] = iteration + 1
    state["history"].append(
        {
            "iteration": iteration,
            "source_iteration": source_iteration,
            "reco_job_ids": job_ids,
            "watchdog_job_id": watchdog_job_id,
            "controller_submitted_next": next_controller_job_id,
            "converged": False,
        }
    )
    save_state(campaign_root, state)

    print(f"[INFO] Iteration {iteration} setup done.")
    print(f"[INFO] Submitted reco jobs: {job_ids}")
    print(f"[INFO] Submitted watchdog: {watchdog_job_id}")
    print(f"[INFO] Submitted next controller: {next_controller_job_id}")


if __name__ == "__main__":
    main()