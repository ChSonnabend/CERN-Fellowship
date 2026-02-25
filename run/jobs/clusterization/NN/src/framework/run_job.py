import json
import sys
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--log-dir", default=".", help="Log directory")
parser.add_argument("-sd", "--submission-dir", default=".", help="Submission directory with TRAIN.sh")
args = parser.parse_args()

configs_file = open("config.json", "r")
CONF = json.load(configs_file)

### directory settings
training_dir    = CONF["exec_settings"]["training_dir"]
output_folder   = CONF["exec_settings"]["output_folder"]

### network settings
training_file   = CONF["network_settings"]["training_file"]

### job settings
optional_args   = CONF["job_settings"]["optional_args"]

configs_file.close()


job_ids = [-1]

### Submit job for clusterization
out = subprocess.check_output("sbatch --output={0}/job.out --error={0}/job.err {2} {1}/TRAIN.sh {0}".format(args.log_dir, args.submission_dir, optional_args), shell=True).decode().strip('\n')
print(out)