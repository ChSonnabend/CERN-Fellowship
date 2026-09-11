import json
import os, glob
import argparse
import uproot
import numpy as np

parser = argparse.ArgumentParser(description="Parse log files to extract latest cluster information")
parser.add_argument("--log-dir", type=str, help="Directory containing log files")
args = parser.parse_args()

GPU_PROC_NN_PATH = ["reco_task", "input-digits", "configKeyValues", "GPU_proc_nn"]

def deep_get(d, keys):
    cur = d
    for k in keys:
        cur = cur[k]
    return cur

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_scan_values_from_config(cfg, scan_params):
    gpu_cfg = deep_get(cfg, GPU_PROC_NN_PATH)
    return [float(gpu_cfg[p]) for p in scan_params]

def parse_reco_log(log_path):
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
        # print(f"Could not parse 'Correctly Attached non-fake normalized' from {log_path}")
        pass
    elif num_fake_attached is None:
        # print(f"Could not parse 'Fake attached clusters' from {log_path}")
        pass
    else:
        delta_attached = num_corr_attached - num_fake_attached
    return {
        "num_tracks": num_tracks if num_tracks is not None else -1,
        "num_corr_attached": num_corr_attached,
        "num_fake_attached": num_fake_attached,
        "delta_attached": delta_attached,
    }

def find_config_for_output_dir(output_dir):
    direct = output_dir + "/config_qa.json"
    if os.path.exists(direct):
        return direct
    candidates = glob.glob(os.path.join(output_dir, "config*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not uniquely locate config JSON in output dir: {output_dir}")

def parse_histograms(log_path, histogram_name: str = "tracksRecAllPrimVsY_eff"):
    f = uproot.open(log_path)
    h = f[histogram_name].values()
    el = f[histogram_name].errors("low")
    eh = f[histogram_name].errors("high")
    e = np.maximum(el, eh)
    return np.sum(h[1] / e[1]**2) / np.sum(1 / e[1]**2)

def parse_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value]
    raise TypeError(f"Expected string or list, got {type(value)}")

scores = dict()
campaign_cfg = load_json(glob.glob(os.path.join(args.log_dir, "**/campaign_config.json"), recursive=True)[0])
scan_params = parse_str_list(campaign_cfg["scan_params"])
for log_file in glob.glob(os.path.join(args.log_dir, "**/job_RECO_0.out"), recursive=True):
    try:
        hn = log_file.replace("job_RECO_0.out", "histograms_0.root")
        cfg_path = find_config_for_output_dir(os.path.dirname(os.path.dirname(log_file)))
        cfg = load_json(cfg_path)
        coord = get_scan_values_from_config(cfg, scan_params)
        parsed = parse_reco_log(log_file)
        h_E = parse_histograms(hn, "tracksRecAllPrimVsY_eff") if hn else None
        h_C = parse_histograms(hn, "tracksCloneAllPrimVsY_eff") if hn else None
        h_F = parse_histograms(hn, "tracksFakeAllPrimVsY_eff") if hn else None
        # Reference values from default reconstruction
        r_E = 0.9800413413714134
        r_C = 0.10765200764438837
        r_F = 0.14585534400886513

        if h_E is not None:
            score = h_E/r_E * (((r_F / h_F) * (r_C / h_C)) if (h_E > r_E) else 1) #parsed["delta_attached"] * h_E * (((r_F / h_F) * (r_C / h_C)) if (h_E > r_E) else 1) # parsed["delta_attached"] * h_E / metric_scale
        else:
            score = None

        scores[score] = {
            "coord": coord,
            "path": log_file,
            "h_E": h_E,
            "h_C": h_C,
            "h_F": h_F
        }
    except Exception as e:
        continue

print("Best score:", max(scores.keys()))
print("Best config:", scores[max(scores.keys())])
