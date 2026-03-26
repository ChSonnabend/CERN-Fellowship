from sys import exit
import os
import json
from datetime import date
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.json", help="JSON file with settings for jobs")
parser.add_argument("-avoid-q", "--avoid-question", default=0, help="Whether to overwrite existing jobs directory without asking")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)
configs_file.close()

### directory settings
configurations_dir              = CONF["submission"]["configurations_dir"]

#################################

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
        if not args.avoid_question:
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
for i, config_qa in enumerate(glob.glob(configurations_dir + "/**/*.json", recursive=True)):

    # print(config_qa)
    cf = open(config_qa, "r")
    SUBMIT = json.load(cf)
    cf.close()
    run_analysis = SUBMIT["data_settings"]["real-data"] and ("mode" in SUBMIT["analysis"].keys()) and (SUBMIT["analysis"]["mode"] != 0)

    if not run_analysis:
        directories.append(SUBMIT["exec_settings"]["output_dir"])

longest_common_dir = longest_common_substring(directories)
if longest_common_dir:
    check_path(longest_common_dir, False)


for i, config_qa in enumerate(glob.glob(configurations_dir + "/**/*.json", recursive=True)):

    if not "do_not_run" in config_qa:
        cf = open(config_qa, "r")
        SUBMIT = json.load(cf)
        cf.close()
        run_analysis = False
        if (SUBMIT["data_settings"]["real-data"] and ("mode" in SUBMIT["analysis"].keys()) and (SUBMIT["analysis"]["mode"] != 0)):
            run_analysis = SUBMIT["analysis"]["mode"]

        output_dir = SUBMIT["exec_settings"]["output_dir"]

        if SUBMIT["data_settings"]["real-data"] and SUBMIT["exec_settings"]["bad_runs"] != ";;":
            if os.path.isfile(SUBMIT["exec_settings"]["bad_runs"]):
                os.system("cp {0} {1}".format(SUBMIT["exec_settings"]["bad_runs"], os.path.join(output_dir, "badRuns.txt")))
            else:
                print("Bad runs file does not exist: {0}".format(SUBMIT["exec_settings"]["bad_runs"]))
                print("Skipping copying bad runs file.")

        if run_analysis:
            os.system("cp {0} {1}".format(SUBMIT["analysis"]["configuration_pid"], os.path.join(output_dir, "configurations_" + run_analysis + ".json")))
            os.system("cp {0} {1}".format(SUBMIT["analysis"]["output_director"], os.path.join(output_dir, "OutputDirector" + run_analysis.upper() + ".json")))
            if SUBMIT["data_settings"]["PID"]["fetch-from-ccdb"]==0:
                os.system("cp {0} {1}".format(SUBMIT["data_settings"]["PID"]["nn-path"], os.path.join(output_dir, "nnpid.onnx")))
        else:
            if longest_common_dir != output_dir:
                check_path(output_dir, False)

            os.system("cp {0} {1}".format(config_qa, os.path.join(output_dir, "config_qa.json")))
            os.system("cp {0} {1}".format(args.config, os.path.join(output_dir, "config.json")))
            if (SUBMIT["data_settings"]["real-data"]) and (SUBMIT["analysis"]["run"] > 0):
                os.system("cp {0} {1}".format(SUBMIT["analysis"]["configuration_pid"], os.path.join(output_dir, "configurations_pid.json")))
                os.system("cp {0} {1}".format(SUBMIT["analysis"]["output_director"], os.path.join(output_dir, "OutputDirector.json")))

