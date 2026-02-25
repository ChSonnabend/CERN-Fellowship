from sys import exit
import os
import json
from datetime import datetime
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.json", help="JSON file with settings for jobs")
parser.add_argument("-su", "--suite", default="suite.json", help="JSON file with suite settings")
parser.add_argument("-s", "--skip-q", default=0, help="Skip directory check")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)
now = datetime.now()
day = now.strftime("%d-%m-%Y")

### execution settings
training_dir        = CONF["exec_settings"]["training_dir"]
output_folder       = os.path.join(CONF["exec_settings"]["output_folder"], day)
enable_qa           = CONF["exec_settings"]["enable_qa"]

### suite submission
suite_submission    = CONF["suite_submission"]["suite_submission"]
suite_configs       = CONF["suite_submission"]["configurations_dir"]

### network settings
configurations      = CONF["network_settings"]["configurations_file"]
training_file       = CONF["network_settings"]["training_file"]

configs_file.close()

#################################

def check_path(path, overwrite=True):
    return_value = False
    if os.path.exists(path):
        response = input("Jobs directory ({}) exists. Overwrite it? (y/n) ".format(path))
        if response == 'y':
            os.system('rm -rf {0}'.format(path))
            return_value = True
        else:
            if overwrite:
                print("Stopping macro!")
                exit()
            else:
                print("Directory not overwritten!")
    else:
        return_value = True
    return return_value

def overwrite_subdirs():
    return_value = False
    response = input("Can subdirectories be overwritten? (y/n) ")
    if response == 'y':
        return_value = True
    return return_value

if suite_submission:

    if not args.skip_q:
        overwrite_full = check_path(output_folder, True)

    for suite_file in glob.glob(os.path.join(suite_configs, "**", args.suite), recursive=True):
        suite_path = os.path.dirname(suite_file)
        suite_configs_file = open(suite_file, "r")
        SUITE = json.load(suite_configs_file)

        for k_subsuite in SUITE["network-submission"].keys():

            for tr_config in SUITE["common"]["training-data"].keys():

                # print("Processing {0} - {1}".format(tr_config, k_subsuite))

                v_subsuite = SUITE["network-submission"][k_subsuite]
                final_out_folder = os.path.join(output_folder, tr_config, v_subsuite["output_folder"])
                if os.path.exists(final_out_folder):
                    os.system('rm -rf {0}'.format(final_out_folder))
                os.makedirs(final_out_folder, exist_ok=True)

                os.system('cp {0} {1}'.format(args.config, final_out_folder))
                os.system('cp {0} {1}'.format(training_file, os.path.join(final_out_folder, "train.py")))
                os.system('cp {0} {1}'.format(k_subsuite, os.path.join(final_out_folder, "configurations.py")))
                if not os.path.exists(os.path.join(final_out_folder, "network")):
                    os.makedirs(os.path.join(final_out_folder, "network"))
                if not os.path.exists(os.path.join(final_out_folder, "QA")):
                    os.makedirs(os.path.join(final_out_folder, "QA"))

                with open(os.path.join(final_out_folder, "configurations.py"), 'r') as file:
                    filedata = file.read()
                for kr, vr in v_subsuite["replace"].items():
                    if "data_path" in kr:
                        vr = os.path.join(SUITE["common"]["training-data"][tr_config], vr)
                    filedata = filedata.replace(kr + ' = ";;"', kr + ' = "{}"'.format(vr))
                filedata = filedata.replace('os.path.join(";;", "config.json")', 'os.path.join("{}", "config.json")'.format(final_out_folder))
                with open(os.path.join(final_out_folder, "configurations.py"), 'w') as file:
                    file.write(filedata)

else:
    if not args.skip_q:
        check_path(os.path.join(training_dir, output_folder), True)
    os.makedirs(os.path.join(training_dir, output_folder, "network"))
    if(enable_qa):
        os.makedirs(os.path.join(training_dir, output_folder,"QA"))
    os.system('cp {0} {1}/configurations.py'.format(configurations, output_folder))
    os.system('cp {0} {1}'.format(args.config, output_folder))
    os.system('cp {0} {1}/train.py'.format(training_file, output_folder))