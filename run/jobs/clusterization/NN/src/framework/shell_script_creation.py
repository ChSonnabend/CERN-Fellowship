import sys
import os
import json
import argparse
from os import path

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="/lustre/alice/users/csonnab/PhD/jobs/clusterization/NN/config.json", help="JSON file with settings for data conversion jobs")
parser.add_argument("-js", "--job-script", default=".", help="Path to job script")
parser.add_argument("-jc", "--job-config", default = "{}", help="Job settings (dict) with which to update the default configs")
parser.add_argument("-sd", "--submission-dir", default=".", help="Path to which the bash script gets written and from where it will be executed")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)
configs_file.close()

for imp in CONF["directory_settings"]["classes"]:
    sys.path.append(imp)
from GeneralPurposeClass.deep_update_json import deep_update

### directory settings
rocm_container  = CONF["directory_settings"]["rocm_container"]
cuda_container  = CONF["directory_settings"]["cuda_container"]
bash_path = path.join(args.submission_dir, "TRAIN.sh")

### job settings
global_slurm_defaults = {
    "name": "JOB",
    "time": 60,
    "chdir": args.submission_dir,
    "notify": "END,FAIL,INVALID_DEPEND",
    "email": "",
    "special-args": {
        "optional_args": "--exclusive"
    }
}

ext_job_config = json.loads(args.job_config)
job_settings = deep_update(global_slurm_defaults, ext_job_config, verbose=False)

slurm_replacements = {
    "name": "job-name",
    "kernelsPerJob": "cpus-per-task",
    "memory": "mem",
    "notify": "mail-type",
    "email": "mail-user",
    "folder": "chdir",
}

# Replace invalid key names with actual slurm option names
for k in list(job_settings):
    if k in slurm_replacements:
        job_settings[slurm_replacements[k]] = job_settings.pop(k)

device = ext_job_config["special-args"].get("device", None)
ngpus = ext_job_config["special-args"].get("ngpus", 1)
local_slurm_defaults = {} #default
interpreters = None
extra_workflow = ""

if device == "EPN": ### Setup to submit to EPN nodes

    local_slurm_defaults = {
        "partition": "prod"
    }
    interpreters = {
        "py": "/bin/python3.9"
    }

    if ngpus:
        nodes = ngpus // 8
        ntasks_per_node = 8 if ngpus > 8 else ngpus

        local_slurm_defaults["nodes"] = nodes
        local_slurm_defaults["gres"] = "gpu:" + str(ntasks_per_node)
        local_slurm_defaults["ntasks-per-node"] = ntasks_per_node

        extra_workflow = "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}"

elif device == "MI100_GPU":

    local_slurm_defaults = {
        "partition": "gpu",
        "time": 480,
        "constraint": "mi100"
    }
    interpreters = {
        "py": "apptainer exec " + rocm_container
    }

    if ngpus:
        nodes = ngpus // 8
        ntasks_per_node = 8 if ngpus > 8 else ngpus

        local_slurm_defaults["nodes"] = nodes
        local_slurm_defaults["gres"] = "gpu:" + str(ntasks_per_node)
        local_slurm_defaults["ntasks-per-node"] = ntasks_per_node

        extra_workflow = "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}"

elif device == "CPU":
    local_slurm_defaults = {
        "time": 480
    }
    interpreters = {
        "py": "apptainer exec " + rocm_container
    }

else:
    print("Choose a given device (GPU or CPU)!")
    print("Stopping.")
    exit()


### Dictionary creation

job_settings = deep_update(job_settings, local_slurm_defaults, verbose=False)
job_settings.pop("special-args", None)

### Script creation

def create_slurm_header(config_dict, workflow):
    script = "#!/bin/bash\n"
    for k, v in config_dict.items():
        if v:
            script += "#SBATCH --" + str(k) + "=" + str(v) + "\n"
    script += "\n" + workflow
    return script

choose_interpreter = interpreters.get(args.job_script.split(".")[-1], "")
workflow = extra_workflow + "\n" + "time srun " + choose_interpreter + " " + args.job_script + " --config " + args.config
final_script = create_slurm_header(job_settings, workflow)

bash_file = open(bash_path, "w")
bash_file.write(final_script)
bash_file.close()