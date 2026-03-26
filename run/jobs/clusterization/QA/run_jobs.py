from sys import exit
import os
import json
from datetime import date
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.json", help="JSON file with settings for jobs")
parser.add_argument("-o", "--options", default="all", help="Options: reco (runs only the reconstruction. Reconstruciton is always run), qa (runs reco + QA), afterburner (runs reco + afterburner qa), all (runs everything)")
parser.add_argument("-s", "--submit", default=1, type=int, help="Submit the job")
parser.add_argument("-l", "--limit", default=-1, type=int, help="Limit the number of jobs submitted")
parser.add_argument("-perf-cpu", "--performance-test-cpu", type=int, default=0, help="Runs the CPU perfromance benchmarking")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)
configs_file.close()

### directory settings
configurations_dir              = CONF["submission"]["configurations_dir"]

# if not os.path.exists(".max_session.tmp"):
#     with open(".max_session.tmp", 'w') as f:
#         f.write('0')

for i, config_qa in enumerate(glob.glob(configurations_dir + "/**/*.json", recursive=True)):

    if not "do_not_run" in config_qa:
        print("\n---> Submission for", config_qa)
        cf = open(config_qa, "r")
        SUBMIT = json.load(cf)
        cf.close()
        os.system("python3 {0} --config {1} --submission-config {2} --output-dir {3} --remove-files {4} --id {5} --from-aod {6} --options {7} --submit {8} --limit {9} --performance-test-cpu {10}".format(
            CONF["submission"]["qa_script"], args.config, config_qa,
            SUBMIT["exec_settings"]["output_dir"], SUBMIT["data_settings"]["remove-misc-files"], str(i),
            int(SUBMIT["data_settings"]["from-aod"]), args.options, args.submit,
            args.limit, args.performance_test_cpu)
        )

# print("\nAll submissions done! Max session ID was: {0}".format(open(".max_session.tmp", 'r').readlines()[0]))
# os.system("rm .max_session.tmp")