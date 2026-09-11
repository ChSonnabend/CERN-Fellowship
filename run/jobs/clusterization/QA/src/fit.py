#!/usr/bin/env python3

import argparse
import glob
import json
import os, sys
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import uproot

sys.path.append("/lustre/alice/users/csonnab/cern-fellowship/classes")
from HypersurfaceOptimization.HO.ho_core import HO

GPU_PROC_NN_PATH = ["reco_task", "input-digits", "configKeyValues", "GPU_proc_nn"]


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


def deep_get(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def parse_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value]
    raise TypeError(f"Expected string or list, got {type(value)}")


def parse_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(f"Expected string or list, got {type(value)}")


def parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
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


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        xl = value.strip().lower()
        if xl in ("1", "true", "yes", "y"):
            return True
        if xl in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"Cannot interpret boolean value '{value}'")
    raise TypeError(f"Expected bool/int/str, got {type(value)}")


def parse_bounds(value: Any, ndim: Optional[int] = None) -> np.ndarray:
    if isinstance(value, str):
        flat = [float(x.strip()) for x in value.split(",") if x.strip()]
        arr = np.array(flat, dtype=float)
        if arr.size % 2 != 0:
            raise ValueError("Bounds string must contain an even number of values")
        arr = arr.reshape(-1, 2)
    elif isinstance(value, (list, tuple)):
        arr = np.array(value, dtype=float)
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                raise ValueError("Flat bounds list must contain an even number of values")
            arr = arr.reshape(-1, 2)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            raise ValueError("Bounds must be shape (N,2) or a flat list of length 2N")
    else:
        raise TypeError(f"Expected bounds as string or list, got {type(value)}")

    if ndim is not None and arr.shape[0] != ndim:
        raise ValueError(f"Expected {ndim} bounds rows, got {arr.shape[0]}")
    return arr


# ============================================================
# Config / result handling
# ============================================================

def get_scan_values_from_config(cfg: Dict[str, Any], scan_params: Sequence[str]) -> List[float]:
    gpu_cfg = deep_get(cfg, GPU_PROC_NN_PATH)
    return [float(gpu_cfg[p]) for p in scan_params]


def parse_reco_log(log_path: Path) -> Dict[str, float]:
    num_tracks = None
    num_corr_attached = None
    num_fake_attached = None

    with open(log_path, "r") as f:
        for line in f:
            if "track(s)" in line and "found" in line:
                num_tracks = int(line.split("found")[-1].split("track(s)")[0].strip())
            elif "Correctly Attached non-fake normalized" in line:
                num_corr_attached = float(line.split(":")[-1].split("(")[0].strip())
            elif "Fake attached clusters" in line:
                num_fake_attached = float(line.split(":")[-1].split("(")[0].strip())

    delta_attached = 0
    if num_corr_attached is None:
        print(f"Could not parse 'Correctly Attached non-fake normalized' from {log_path}")
    elif num_fake_attached is None:
        print(f"Could not parse 'Fake attached clusters' from {log_path}")
    else:
        delta_attached = num_corr_attached - num_fake_attached
    return {
        "num_tracks": num_tracks if num_tracks is not None else -1,
        "num_corr_attached": num_corr_attached,
        "num_fake_attached": num_fake_attached,
        "delta_attached": delta_attached,
    }

def parse_histograms(log_path: Path, histogram_name: str = "tracksRecAllPrimVsY_eff") -> Dict[str, Any]:
    f = uproot.open(log_path)
    h = f[histogram_name].values()
    el = f[histogram_name].errors("low")
    eh = f[histogram_name].errors("high")
    e = np.maximum(el, eh)
    return np.sum(h[1] / e[1]**2) / np.sum(1 / e[1]**2)


def find_config_for_output_dir(output_dir: Path) -> Path:
    direct = output_dir / "config_qa.json"
    if direct.exists():
        return direct
    candidates = list(output_dir.glob("config*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not uniquely locate config JSON in output dir: {output_dir}")


def collect_completed_results_from_output_dirs(
    output_dirs: Sequence[Path],
    scan_params: Sequence[str],
    metric_scale: float,
    reco_log_name: str = "job_RECO_0.out",
    histogram_name: str = "histograms_0.root",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    points = []
    values = []
    rows = []

    for output_dir in output_dirs:
        log_paths = glob.glob(str(output_dir) + "/**/" + reco_log_name, recursive=True)
        histogram_names = []
        for log in log_paths:
            hn = log.replace(reco_log_name, histogram_name)
            if os.path.exists(hn):
                histogram_names.append(hn)
            else:
                histogram_names.append(None)

        for i, log_path in enumerate(log_paths):
            if not Path(log_path).exists():
                continue

            cfg_path = find_config_for_output_dir(output_dir)
            cfg = load_json(cfg_path)

            try:
                coord = get_scan_values_from_config(cfg, scan_params)
                parsed = parse_reco_log(log_path)
                h_E = parse_histograms(histogram_names[i], "tracksRecAllPrimVsY_eff") if histogram_names[i] else None
                h_C = parse_histograms(histogram_names[i], "tracksCloneAllPrimVsY_eff") if histogram_names[i] else None
                h_F = parse_histograms(histogram_names[i], "tracksFakeAllPrimVsY_eff") if histogram_names[i] else None
                # Reference values from default reconstruction
                r_E = 0.9800413413714134
                r_C = 0.10765200764438837
                r_F = 0.14585534400886513
                if h_E is not None:
                    score = (h_E / r_E) * (((r_F / h_F) * (r_C / h_C)) if (h_E > r_E) else 1) # parsed["delta_attached"] * h_E / metric_scale
                else:
                    score = None
            except:
                print(f"Error processing log {log_path}:")
                import traceback
                traceback.print_exc()
                continue

            if score is not None:
                if score > 0:
                    points.append(coord)
                    values.append(score)
                    rows.append(
                        {
                            "output_dir": str(output_dir),
                            "config_path": str(cfg_path),
                            "point": coord,
                            "score": score,
                            **parsed,
                        }
                    )

    if not points:
        raise RuntimeError("No completed outputs with readable logs were found")

    return np.asarray(points, dtype=float), np.asarray(values, dtype=float), rows


# ============================================================
# Interpolation helpers
# ============================================================

def interpolation_dict_from_points(points: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    out = {}
    for point, val in zip(points, values):
        if val > 0:
            key = "_".join(f"{x:.12g}" for x in point)
            out[key] = float(val)
    return out


def estimate_next_half_widths(
    points: np.ndarray,
    values: np.ndarray,
    center: np.ndarray,
    prev_half_widths: np.ndarray,
    global_bounds: np.ndarray,
    min_half_widths: np.ndarray,
    shrink_factor: float = 0.5,
    expansion_factor: float = 1.25,
    top_k: int = 8,
) -> np.ndarray:
    idx = np.argsort(values)[::-1]
    best_points = points[idx[:min(top_k, len(points))]]
    span = np.max(np.abs(best_points - center.reshape(1, -1)), axis=0)

    candidate = np.maximum(span * expansion_factor, prev_half_widths * shrink_factor)
    candidate = np.maximum(candidate, min_half_widths)

    # left_room = center - global_bounds[:, 0]
    # right_room = global_bounds[:, 1] - center
    # candidate = np.minimum(candidate, np.minimum(left_room, right_room))
    # candidate = np.maximum(candidate, min_half_widths)

    return candidate


def converged(
    prev_center: np.ndarray,
    new_center: np.ndarray,
    prev_half_widths: np.ndarray,
    next_half_widths: np.ndarray,
    min_half_widths: np.ndarray,
    shift_frac: float = 1.,
    rtol: float = 1e-12,
) -> bool:
    small_shift = np.all(np.abs(new_center - prev_center) < shift_frac * prev_half_widths)
    min_box = np.all(next_half_widths <= min_half_widths * (1.0 + rtol))
    return bool(small_shift or min_box)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    return ap.parse_args()

def make_unit_vector(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=float)
    v[idx] = 1.0
    return v

def collect_output_dirs_up_to_iteration(campaign_root: Path, last_iteration: int) -> List[Path]:
    output_dirs: List[Path] = []

    for it in range(last_iteration + 1):
        it_dir = campaign_root / "iterations" / f"iter_{it:03d}"
        sampling_path = it_dir / "sampling_plan.json"
        if not sampling_path.exists():
            continue

        sampling = load_json(sampling_path)
        for p in sampling.get("generated_output_dirs", []):
            output_dirs.append(Path(p))

    return output_dirs


def collect_history_fit_errors(
    campaign_root: Path,
    last_iteration_inclusive: int,
    metric_preference: str = "val_rmse_then_rmse",
) -> List[float]:
    values: List[float] = []

    for it in range(last_iteration_inclusive + 1):
        summary_path = campaign_root / "iterations" / f"iter_{it:03d}" / "interpolation_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = load_json(summary_path)
        except Exception:
            continue

        metric = None
        if metric_preference == "val_rmse_then_rmse":
            val_rmse = summary.get("fit_val_rmse", None)
            if val_rmse is not None:
                metric = float(val_rmse)
            else:
                rmse = summary.get("fit_rmse", None)
                if rmse is not None:
                    metric = float(rmse)
        elif metric_preference == "rmse":
            rmse = summary.get("fit_rmse", None)
            if rmse is not None:
                metric = float(rmse)

        if metric is None:
            continue
        if np.isfinite(metric):
            values.append(metric)

    return values

def save_all_pair_surfaces(
    ho_analysis: HO,
    result: Dict[str, Any],
    scan_params: Sequence[str],
    output_dir: Path,
    grid_size: int = 200,
    mode: str = "slice",
    plot_space: str = "raw",
    center_data: bool = False,
    percentile_span: float = 1.0,
    scale_axis: Tuple[str, str] = ("linear", "linear"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    d = len(scan_params)

    extremum_result = result.get("extremum_result", None)
    fit_info = ho_analysis.settings["fit_info"]

    # Choose center:
    # 1) extremum if available
    # 2) otherwise the sampled point with maximum value
    if extremum_result is not None:
        center_point = np.asarray(extremum_result["x_opt"], dtype=float)
        show_extremum = True
        extremum_to_plot = extremum_result
    else:
        vals = np.asarray(fit_info["vals"], dtype=float)
        X_raw = np.asarray(fit_info["X_raw"], dtype=float)

        if len(vals) == 0:
            raise ValueError("fit_info['vals'] is empty")
        if len(X_raw) != len(vals):
            raise ValueError("fit_info['X_raw'] and fit_info['vals'] must have the same length")

        imax = np.argmax(vals)
        center_point = np.asarray(X_raw[imax], dtype=float)

        # Optional: fabricate a marker result so the chosen max sample is shown
        extremum_to_plot = {
            "x_opt": center_point,
            "value_opt": vals[imax],
            "mode": "max sample",
        }
        show_extremum = True

    if plot_space == "modelspace":
        center_for_plot = ho_analysis._points_to_model_space(
            fit_info,
            center_point.reshape(1, -1),
        )[0]
    else:
        center_for_plot = center_point

    for i, j in itertools.combinations(range(d), 2):
        axis_u = make_unit_vector(d, i)
        axis_v = make_unit_vector(d, j)

        save_path = output_dir / f"surface_{scan_params[i]}__vs__{scan_params[j]}.pdf"

        ho_analysis.settings["plot_surface"] = {
            "axis_u": axis_u.tolist(),
            "axis_v": axis_v.tolist(),
            "mode": mode,
            "plot_space": plot_space,
            "center": center_for_plot,
            "grid_size": grid_size,
            "center_data": center_data,
            "percentile_span": percentile_span,

            "show_contours": True,
            "show_extremum": show_extremum,
            "extremum_result": extremum_to_plot,

            "xlabel": scan_params[i],
            "ylabel": scan_params[j],
            "title": f"{mode}: {scan_params[i]} vs {scan_params[j]}",
            "scale_axis": scale_axis,

            "savefig": str(save_path),
            "showfig": False,
        }

        ho_analysis.plot_surface_2d()

def main() -> None:
    args = parse_args()

    campaign_root = Path(args.campaign_root).expanduser().resolve()
    iteration = args.iteration

    campaign_cfg = load_json(campaign_root / "campaign_config.json")
    scan_params = parse_str_list(campaign_cfg["scan_params"])
    ndim = len(scan_params)

    lhs_bounds = parse_bounds(campaign_cfg["lhs_bounds"], ndim=ndim)
    min_half_widths = np.asarray(parse_float_list(campaign_cfg.get("min_half_widths", [1e-4] * ndim)), dtype=float)
    hidden_sizes = tuple(parse_int_list(campaign_cfg.get("hidden_sizes", [64, 64])))

    metric_scale = float(campaign_cfg.get("metric_scale", 1.4e7))
    lr = float(campaign_cfg.get("lr", 1e-2))
    n_epochs = int(campaign_cfg.get("n_epochs", 10000))
    extremum_mode = str(campaign_cfg.get("extremum_mode", "max"))
    interpolation_log_space = parse_boolish(campaign_cfg.get("interpolation_log_space", 0))
    interpolation_use_validation_model = parse_boolish(campaign_cfg.get("interpolation_use_validation_model", 1))
    interpolation_validation_split = float(campaign_cfg.get("interpolation_validation_split", 0.2))

    prev_iteration_dir = campaign_root / "iterations" / f"iter_{iteration-1:03d}"
    prev_sampling = load_json(prev_iteration_dir / "sampling_plan.json")

    all_output_dirs = collect_output_dirs_up_to_iteration(
        campaign_root=campaign_root,
        last_iteration=iteration - 1,
    )

    observed_points, observed_values, rows = collect_completed_results_from_output_dirs(
        output_dirs=all_output_dirs,
        scan_params=scan_params,
        metric_scale=metric_scale,
    )

    i_best_sample = int(np.argmax(observed_values))
    best_sample_point = observed_points[i_best_sample]
    best_sample_value = float(observed_values[i_best_sample])
    print(
        "[INFO] Highest observed sampled value: "
        f"{best_sample_value:.8g} at point {np.array2string(best_sample_point, precision=6)}"
    )

    dump_json(prev_iteration_dir / "collected_rows.json", {"rows": rows})

    interp_dict = interpolation_dict_from_points(observed_points, observed_values)
    settings = {
        "parser": "part_2",
        "interpolate": {
            "dict_analyze": interp_dict,
            "hidden_sizes": hidden_sizes,
            "lr": lr,
            "n_epochs": n_epochs,
            "do_plot": False,
            "find_extremum": True,
            "extremum_mode": extremum_mode,
            "log_space": interpolation_log_space,
            "use_validation_model": interpolation_use_validation_model,
            "validation_split": interpolation_validation_split,
        }
    }

    ho_analysis = HO(settings)

    result = ho_analysis.grid_interpolate_nn_nd()

    extremum_result = result.get("extremum_result")
    if extremum_result is not None:
        x_opt = np.asarray(extremum_result["x_opt"], dtype=float)
        value_opt = float(extremum_result["value_opt"])
        value_at_best_sample_by_nn = float(ho_analysis.evaluate_nn_surface_nd(best_sample_point))

        # Diagnostic: compare optimizer maximum with best sampled point in both raw and transformed spaces
        dist_raw = float(np.linalg.norm(x_opt - best_sample_point))
        try:
            fit_info = ho_analysis.settings["fit_info"]
            best_sample_t = ho_analysis._points_to_model_space(fit_info, best_sample_point.reshape(1, -1))[0]
            x_opt_t = ho_analysis._points_to_model_space(fit_info, x_opt.reshape(1, -1))[0]
            dist_model = float(np.linalg.norm(x_opt_t - best_sample_t))
        except Exception:
            dist_model = None

        print(
            "[INFO] NN extremum: "
            f"value={value_opt:.8g}, nn_at_best_sample={value_at_best_sample_by_nn:.8g}, "
            f"raw_distance_to_best_sample={dist_raw:.6g}"
            + (
                ""
                if dist_model is None
                else f", modelspace_distance_to_best_sample={dist_model:.6g}"
            )
        )

    plots_dir = prev_iteration_dir / "surface_plots"
    save_all_pair_surfaces(
        ho_analysis=ho_analysis,
        result=result,
        scan_params=scan_params,
        output_dir=plots_dir,
        grid_size=int(campaign_cfg.get("plot_surface_grid_size", 200)),
        mode=str(campaign_cfg.get("plot_surface_mode", "slice")),
        plot_space=str(campaign_cfg.get("plot_surface_space", "raw")),
        center_data=parse_boolish(campaign_cfg.get("plot_surface_center_data", 0)),
        percentile_span=float(campaign_cfg.get("plot_surface_percentile_span", 1.0)),
        scale_axis=tuple(campaign_cfg.get("plot_surface_scale_axis", ["linear", "linear"])),
    )

    x_opt_nn = np.asarray(result["extremum_result"]["x_opt"], dtype=float)
    fit_info = result["fit_info"]
    fit_rmse = float(fit_info.get("rmse"))
    fit_train_rmse = float(fit_info.get("train_rmse", fit_rmse))
    fit_val_rmse = fit_info.get("val_rmse", None)
    fit_val_rmse = None if fit_val_rmse is None else float(fit_val_rmse)

    fit_quality_metric_name = str(campaign_cfg.get("fit_quality_metric", "val_rmse_then_rmse"))
    current_fit_error = fit_val_rmse if (fit_quality_metric_name == "val_rmse_then_rmse" and fit_val_rmse is not None) else fit_rmse

    fit_history = collect_history_fit_errors(
        campaign_root=campaign_root,
        last_iteration_inclusive=max(iteration - 2, -1),
        metric_preference=fit_quality_metric_name,
    )

    fit_error_history_min_points = int(campaign_cfg.get("fit_error_history_min_points", 3))
    fit_error_bad_factor = float(campaign_cfg.get("fit_error_bad_factor", 1.8))
    fallback_to_sample_max_if_bad_fit = parse_boolish(campaign_cfg.get("fallback_to_sample_max_if_bad_fit", 1))

    fit_error_reference = None
    fit_error_is_bad = False
    if len(fit_history) >= fit_error_history_min_points:
        fit_error_reference = float(np.median(np.asarray(fit_history, dtype=float)))
        fit_error_is_bad = bool(current_fit_error > fit_error_bad_factor * fit_error_reference)

    print(
        "[INFO] Fit quality: "
        f"metric={fit_quality_metric_name}, current={current_fit_error:.6g}, "
        f"history_n={len(fit_history)}"
        + (
            ""
            if fit_error_reference is None
            else f", history_median={fit_error_reference:.6g}, bad_factor={fit_error_bad_factor:.6g}, is_bad={fit_error_is_bad}"
        )
    )

    sampled_lo = np.min(observed_points, axis=0)
    sampled_hi = np.max(observed_points, axis=0)
    sampled_span = sampled_hi - sampled_lo
    sampled_span_safe = np.where(sampled_span > 0.0, sampled_span, 1.0)

    nn_to_sample_delta = x_opt_nn - best_sample_point
    nn_to_sample_distance_raw = float(np.linalg.norm(nn_to_sample_delta))
    nn_to_sample_distance_box = float(np.linalg.norm(nn_to_sample_delta / sampled_span_safe))

    fallback_distance_box = float(campaign_cfg.get("fallback_distance_box", 0.75))
    fallback_to_sample_max_if_far = parse_boolish(campaign_cfg.get("fallback_to_sample_max_if_far", 1))

    use_sample_max_far = fallback_to_sample_max_if_far and (nn_to_sample_distance_box > fallback_distance_box)
    use_sample_max_bad_fit = fallback_to_sample_max_if_bad_fit and fit_error_is_bad
    use_sample_max = use_sample_max_far or use_sample_max_bad_fit
    x_opt = best_sample_point.copy() if use_sample_max else x_opt_nn.copy()
    if use_sample_max_far and use_sample_max_bad_fit:
        x_opt_source = "sample_max_far_and_bad_fit"
    elif use_sample_max_far:
        x_opt_source = "sample_max_far"
    elif use_sample_max_bad_fit:
        x_opt_source = "sample_max_bad_fit"
    else:
        x_opt_source = "nn_extremum"

    print(
        "[INFO] Center selection: "
        f"source={x_opt_source}, nn_to_sample_distance_raw={nn_to_sample_distance_raw:.6g}, "
        f"nn_to_sample_distance_box={nn_to_sample_distance_box:.6g}, "
        f"threshold={fallback_distance_box:.6g}, fit_error_is_bad={fit_error_is_bad}"
    )

    prev_center = np.asarray(prev_sampling["center"], dtype=float)
    prev_half = np.asarray(prev_sampling["half_widths"], dtype=float)

    next_half_widths = estimate_next_half_widths(
        points=observed_points,
        values=observed_values,
        center=x_opt,
        prev_half_widths=prev_half,
        global_bounds=lhs_bounds,
        min_half_widths=min_half_widths,
    )

    converged_flag = False
    if prev_center.size > 0:
        converged_flag = converged(
        prev_center=prev_center,
        new_center=x_opt,
        prev_half_widths=prev_half,
        next_half_widths=next_half_widths,
        min_half_widths=min_half_widths,
        shift_frac=float(campaign_cfg.get("convergence_shift_frac", .2)),
    )

    dump_json(
        prev_iteration_dir / "interpolation_summary.json",
        {
            "scan_params": scan_params,
            "x_opt": x_opt.tolist(),
            "x_opt_nn": x_opt_nn.tolist(),
            "x_opt_source": x_opt_source,
            "best_sample_point": best_sample_point.tolist(),
            "best_sample_value": best_sample_value,
            "fit_rmse": fit_rmse,
            "fit_train_rmse": fit_train_rmse,
            "fit_val_rmse": fit_val_rmse,
            "fit_quality_metric": fit_quality_metric_name,
            "fit_error_current": current_fit_error,
            "fit_error_history": fit_history,
            "fit_error_history_median": fit_error_reference,
            "fit_error_bad_factor": fit_error_bad_factor,
            "fit_error_is_bad": fit_error_is_bad,
            "nn_to_sample_distance_raw": nn_to_sample_distance_raw,
            "nn_to_sample_distance_box": nn_to_sample_distance_box,
            "fallback_distance_box": fallback_distance_box,
            "fallback_to_sample_max_if_bad_fit": fallback_to_sample_max_if_bad_fit,
            "half_widths_before": prev_half.tolist(),
            "next_half_widths": next_half_widths.tolist(),
            "converged": converged_flag,
        },
    )

    print(json.dumps(
        {
            "iteration": iteration - 1,
            "x_opt": x_opt.tolist(),
            "x_opt_source": x_opt_source,
            "x_opt_nn": x_opt_nn.tolist(),
            "best_sample_point": best_sample_point.tolist(),
            "fit_error_is_bad": fit_error_is_bad,
            "fit_error_current": current_fit_error,
            "nn_to_sample_distance_box": nn_to_sample_distance_box,
            "fallback_distance_box": fallback_distance_box,
            "next_half_widths": next_half_widths.tolist(),
            "converged": converged_flag,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()