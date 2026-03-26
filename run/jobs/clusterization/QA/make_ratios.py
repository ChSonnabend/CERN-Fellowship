from sys import exit
import os
import json
from datetime import date
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.json", help="JSON file with settings for jobs")
parser.add_argument("-o", "--output-dir", default=";;", help="QA directory (overwrite to the output directory in config_qa.json)")
parser.add_argument("-d", "--dependency", type=int, default=-1, help="Depndency afterok")
parser.add_argument("--gpucf", type=str, default=";;", help="GPU CF directory name")
parser.add_argument("-f", "--file", type=str, default="histograms.root", help="Histograms file from GPUQA.cxx")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)
configs_file.close()

### directory settings
configurations_dir              = CONF["submission"]["configurations_dir"]

#################################

slurm_dict = {**CONF["job_settings"], **CONF["directory_settings"]}
if "memory" not in slurm_dict.keys():
    slurm_dict["memory"] = "80G"
if "kernelsPerJob" not in slurm_dict.keys():
    slurm_dict["kernelsPerJob"] = "20"

baseline_slurm = """#!/bin/bash
#SBATCH --job-name=RATIO                            # Job name
#SBATCH --time=%(time)s                             # Run time limit
#SBATCH --mem=%(memory)s                            # job memory
#SBATCH --partition=%(partition)s                   # job partition (debug, main)
#SBATCH --cpus-per-task=%(kernelsPerJob)s           # job partition (debug, main)

unset http_proxy
unset https_proxy

apptainer shell -B /lustre -B /scratch %(O2_container)s<<\EOF
export JALIEN_TOKEN_CERT=/%(token_dir)s/tokencert_9898.pem
export JALIEN_TOKEN_KEY=/%(token_dir)s/tokenkey_9898.pem
alienv -w /scratch/alice/csonnab/MyO2/sw enter %(O2_env)s
""" % slurm_dict

def longest_common_substring(strings):
    if not strings:
        return ""

    # Take the shortest string in the list, since the common substring can't be longer than this
    shortest_str = min(strings, key=len)
    length = len(shortest_str)

    # Start with the longest possible substring and gradually reduce the size
    for sub_len in range(length, 0, -1):
        for start in range(length - sub_len + 1):
            substring = shortest_str[start:start + sub_len]
            if all(substring in s for s in strings):
                return substring

    return ""

def check_path(path, overwrite=True):
    return_value = False
    if os.path.exists(path):
        response = input("Jobs directory ({}) exists. Overwrite it? (y/n) ".format(path))
        if response == 'y':
            os.system('rm -rf {0}'.format(path))
            os.makedirs(path)
            return_value = True
        else:
            if overwrite:
                print("Stopping macro!")
                exit()
            else:
                print("Directory not overwritten!")
    else:
        os.makedirs(path)
    return return_value

directories = list()
for i, config_qa in enumerate(glob.glob(configurations_dir + "/*.json")):

    cf = open(config_qa, "r")
    SUBMIT = json.load(cf)
    cf.close()

    directories.append(SUBMIT["exec_settings"]["output_dir"])

if args.output_dir != ";;":
    output_dir = args.output_dir
else:
    output_dir = longest_common_substring(directories)

if args.gpucf == ";;":
    for i, ratio_dir in enumerate(glob.glob(output_dir + "/**/" + args.file, recursive=True)):
        if "gpu_cf" in ratio_dir:
            print("--> Jobs for {0}".format(ratio_dir))
            for i, ratio_dir_2 in enumerate(glob.glob(ratio_dir.split("/gpu_cf")[0] + "/**/" + args.file, recursive=True)):
                if "gpu_cf" not in ratio_dir_2:

                    ratio_sh = baseline_slurm
                    ratio_sh += 'root -l -q \'{0}("{1}", "{2}")\'\n'.format(CONF["submission"]["ratio_script"], ratio_dir_2, ratio_dir)
                    ratio_sh += "EOF\n"

                    sh_script = os.path.join(ratio_dir_2.split("/" + args.file)[0], "RATIO.sh")
                    bash_file = open(sh_script, "w")
                    bash_file.write(ratio_sh)
                    bash_file.close()
                    submission_string = "sbatch --output=ratio.out --error=ratio.err --chdir={1} "
                    if args.dependency > 0:
                        submission_string += "--dependency=afterok:{0} ".format(args.dependency)
                    submission_string += "{0}".format(sh_script, ratio_dir_2.split("/" + args.file)[0])
                    os.system("sbatch --output=ratio.out --error=ratio.err --chdir={1} {0}".format(sh_script, ratio_dir_2.split("/" + args.file)[0]))
else:
    for i, ratio_dir in enumerate(glob.glob(output_dir + "/**/" + args.file, recursive=True)):
        gpucf_histogram = glob.glob(os.path.join(args.gpucf, "**", args.file), recursive=True)[0]
        if "gpu_cf" not in ratio_dir:
            ratio_sh = baseline_slurm
            ratio_sh += 'root -l -q \'{0}("{1}", "{2}")\'\n'.format(CONF["submission"]["ratio_script"], ratio_dir, gpucf_histogram)
            ratio_sh += "EOF\n"

            sh_script = os.path.join(ratio_dir.split("/" + args.file)[0], "RATIO.sh")
            bash_file = open(sh_script, "w")
            bash_file.write(ratio_sh)
            bash_file.close()
            submission_string = "sbatch --output=ratio.out --error=ratio.err --chdir={1} "
            if args.dependency > 0:
                submission_string += "--dependency=afterok:{0} ".format(args.dependency)
            submission_string += "{0}".format(sh_script, ratio_dir.split("/" + args.file)[0])
            os.system("sbatch --output=ratio.out --error=ratio.err --chdir={1} {0}".format(sh_script, ratio_dir.split("/" + args.file)[0]))