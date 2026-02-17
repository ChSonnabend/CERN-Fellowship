import json
import os
from datetime import date
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.json", help="Local directory for training of the neural network")
parser.add_argument("-su", "--suite", default="suite.json", help="JSON file with suite settings")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)

### execution settings
training_dir    = CONF["exec_settings"]["training_dir"]
output_folder   = CONF["exec_settings"]["output_folder"]

suite_submission    = CONF["suite_submission"]["suite_submission"]
suite_configs       = CONF["suite_submission"]["configurations_dir"]

configs_file.close()

if suite_submission:
    for suite_file in glob.glob(os.path.join(suite_configs, "**", args.suite), recursive=True):
        suite_configs_file = open(suite_file, "r")
        SUITE = json.load(suite_configs_file)

        for k_subsuite in SUITE["network-submission"].keys():

            for tr_config in SUITE["common"]["training-data"].keys():

                v_subsuite = SUITE["network-submission"][k_subsuite]
                final_out_folder = os.path.join(output_folder, tr_config, v_subsuite["output_folder"])

                if SUITE["network-submission"][k_subsuite]["submit"]:
                    os.system("python3 {0}/framework/shell_script_creation.py --job-script {1} --submission-dir {2} --config {3}".format(os.path.join(training_dir, "src"), os.path.join(training_dir, final_out_folder, "train.py"), final_out_folder, args.config))
                    os.system("python3 {0}/framework/run_job.py --log-dir {1} --submission-dir {2}".format(os.path.join(training_dir, "src"), os.path.join(training_dir, final_out_folder, 'network'), final_out_folder))

        suite_configs_file.close()

else:
    os.system("python3 {0}/framework/shell_script_creation.py --job-script {1} --submission-dir {2} --config {3}".format(os.path.join(training_dir, "src"), os.path.join(training_dir, output_folder, "train.py"), os.path.join(training_dir, output_folder), args.config))
    os.system("python3 {0}/framework/run_job.py --log-dir {1} --submission-dir {2}".format(os.path.join(training_dir, "src"), os.path.join(training_dir, output_folder, 'network'), os.path.join(training_dir, output_folder)))