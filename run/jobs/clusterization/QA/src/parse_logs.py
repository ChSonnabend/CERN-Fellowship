import os, glob
import argparse

parser = argparse.ArgumentParser(description="Parse log files to extract latest cluster information")
parser.add_argument("--log-dir", type=str, help="Directory containing log files")
args = parser.parse_args()

clusters_data = dict()
for log_file in glob.glob(os.path.join(args.log_dir, "**/job_RECO_0.out"), recursive=True):
    try:
        num_corr_attached = None
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Correctly Attached non-fake normalized" in line:
                    num_corr_attached = float(line.split(":")[-1].split("(")[0].strip())
        if num_corr_attached is not None:
            clusters_data[log_file] = num_corr_attached
    except Exception as e:
        continue
    
for log_file, num_corr_attached in clusters_data.items():
    print(f"{log_file}: {num_corr_attached}")