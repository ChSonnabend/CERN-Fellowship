import sys
import os
import json
import argparse
from os import path

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="/lustre/alice/users/csonnab/PhD/jobs/clusterization/NN/config.json", help="JSON file with settings for data conversion jobs")
parser.add_argument("-jbscript", "--job-script", default=".", help="Path to job script")
parser.add_argument("-smdir", "--submission-dir", default=".", help="Path to output / submission directory where the file is written")
args = parser.parse_args()

configs_file = open(args.config, "r")
CONF = json.load(configs_file)

### directory settings
rocm_container  = CONF["directory_settings"]["rocm_container"]
cuda_container  = CONF["directory_settings"]["cuda_container"]

### network settings

### job settings
name            = CONF["job_settings"]["name"]
partition       = CONF["job_settings"]["partition"]
time            = CONF["job_settings"]["time"]
device          = CONF["job_settings"]["device"]
kernelsPerJob   = CONF["job_settings"]["kernelsPerJob"]
memory          = CONF["job_settings"]["memory"]
notify          = CONF["job_settings"]["notify"]
email           = CONF["job_settings"]["email"]

configs_file.close()

job_dict = {'user': os.environ['USER'],
'name': name,
'time': time,
'kJ': int(kernelsPerJob),
'pj': args.submission_dir,
'mem': memory,
'part': partition,
'cuda_container': cuda_container,
'rocm_container': rocm_container,
'job_script': str(args.job_script),
'notify': notify,
'email': email}

if "ngpus" in CONF["job_settings"].keys():
    job_dict['ngpus'] = CONF["job_settings"]["ngpus"]

if "EPN" in device: ### Setup to submit to EPN nodes

    bash_path = path.join(args.submission_dir, "TRAIN.sh")
    script = """#!/bin/bash
#SBATCH --job-name=%(name)s                                                 # Task name
#SBATCH --chdir=%(pj)s                                                      # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=%(part)s                                                # job partition (debug, main)
#SBATCH --mail-type=%(notify)s                                              # notify via email
#SBATCH --mail-user=%(email)s                                               # recipient
""" % job_dict

    if "ngpus" in job_dict.keys() and int(job_dict['ngpus']) > 8:
        job_dict['nodes'] = int(job_dict['ngpus']) // 8
        job_dict['ntasks_per_node'] = 8
        script += """#SBATCH --nodes=%(nodes)s                              # number of nodes
#SBATCH --gres=gpu:8   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=%(ntasks_per_node)s                               # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun /bin/python3.9 %(job_script)s --local-training-dir "$1"
""" % job_dict

    elif int(job_dict['ngpus']) > 1:
        script += """#SBATCH --nodes=1                                      # number of nodes
#SBATCH --gres=gpu:%(ngpus)s   		                                        # reservation for GPU
#SBATCH --ntasks-per-node=%(ngpus)s                                         # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun /bin/python3.9 %(job_script)s --local-training-dir "$1"

""" % job_dict

    else:
        script += """#SBATCH --nodes=1                                      # number of nodes
#SBATCH --gres=gpu:1   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=1                                                 # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time /bin/python3.9 %(job_script)s --local-training-dir "$1"

""" % job_dict

    bash_file = open(bash_path, "w")
    bash_file.write(script)
    bash_file.close()

elif "PRIV" in device:

    bash_file = open(path.join(args.submission_dir, "TRAIN.sh"), "w")
    bash_file.write(
"""#!/bin/bash
#SBATCH --job-name=%(name)s                                                 # Task name
#SBATCH --chdir=%(pj)s                                                      # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=cluster                                                 # job partition (debug, main)
#SBATCH --mail-type=%(notify)s                                              # notify via email
#SBATCH --mail-user=%(email)s                                               # recipient

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time python3 %(job_script)s --local-training-dir "$1"

""" % job_dict

    )
    bash_file.close()

elif device == "MI100_GPU":

    bash_path = path.join(args.submission_dir, "TRAIN.sh")
    script = """#!/bin/bash
#SBATCH --job-name=%(name)s                                                 # Task name
#SBATCH --chdir=%(pj)s                                                      # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=%(part)s                                                # job partition (debug, main)
#SBATCH --mail-type=%(notify)s                                              # notify via email
#SBATCH --mail-user=%(email)s                                               # recipient
#SBATCH --constraint=mi100   		                                        # reservation for GPU
""" % job_dict

    if "ngpus" in job_dict.keys() and int(job_dict['ngpus']) > 8:
        job_dict['nodes'] = int(job_dict['ngpus']) // 8
        job_dict['ntasks_per_node'] = 8
        script += """#SBATCH --nodes=%(nodes)s                              # number of nodes
#SBATCH --gres=gpu:8   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=%(ntasks_per_node)s                               # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --local-training-dir "$1"
""" % job_dict

    elif int(job_dict['ngpus']) > 1:
        script += """#SBATCH --nodes=1                                      # number of nodes
#SBATCH --gres=gpu:%(ngpus)s   		                                        # reservation for GPU
#SBATCH --ntasks-per-node=%(ngpus)s                                         # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --local-training-dir "$1"

""" % job_dict

    else:
        script += """#SBATCH --nodes=1                                      # number of nodes
#SBATCH --gres=gpu:1   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=1                                                 # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time singularity exec %(rocm_container)s python3 %(job_script)s --local-training-dir "$1"

""" % job_dict

    bash_file = open(bash_path, "w")
    bash_file.write(script)
    bash_file.close()

elif device == "MI50_GPU":

    bash_path = path.join(args.submission_dir, "TRAIN.sh")
    script = """#!/bin/bash
#SBATCH --job-name=%(name)s                                                 # Task name
#SBATCH --chdir=%(pj)s                                                      # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --partition=%(part)s                                                # job partition (debug, main)
#SBATCH --mail-type=%(notify)s                                              # notify via email
#SBATCH --mail-user=%(email)s                                               # recipient
#SBATCH --constraint=mi50   		                                        # reservation for GPU
""" % job_dict

    if "ngpus" in job_dict.keys() and int(job_dict['ngpus']) > 8:
        job_dict['nodes'] = int(job_dict['ngpus']) // 8
        job_dict['ntasks_per_node'] = 8
        script += """#SBATCH --nodes=%(nodes)s                              # number of nodes
#SBATCH --gres=gpu:8   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=%(ntasks_per_node)s                               # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --local-training-dir "$1"
""" % job_dict

    elif int(job_dict['ngpus']) > 1:
        script += """#SBATCH --nodes=1                                      # number of nodes
#SBATCH --gres=gpu:%(ngpus)s   		                                        # reservation for GPU
#SBATCH --ntasks-per-node=%(ngpus)s                                         # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time srun singularity exec %(rocm_container)s python3 %(job_script)s --local-training-dir "$1"

""" % job_dict

    else:
        script += """#SBATCH --nodes=1                                      # number of nodes
#SBATCH --gres=gpu:1   		                                                # reservation for GPU
#SBATCH --ntasks-per-node=1                                                 # number of tasks, for MULTI-GPU training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time singularity exec %(rocm_container)s python3 %(job_script)s --local-training-dir "$1"

""" % job_dict

    bash_file = open(bash_path, "w")
    bash_file.write(script)
    bash_file.close()

elif device == "CPU":

    bash_file = open(path.join(args.submission_dir, "TRAIN.sh"), "w")
    bash_file.write(
"""#!/bin/bash

#SBATCH --job-name=%(name)s                                                 # Task name
#SBATCH --chdir=%(pj)s                                                      # Working directory on shared storage
#SBATCH --time=%(time)s                                                     # Run time limit
#SBATCH --mem=%(mem)s                                                       # job memory
#SBATCH --cpus-per-task=%(kJ)s 	                                            # cpus per task
#SBATCH --partition=%(part)s                                                # job partition (debug, main)
#SBATCH --mail-type=%(notify)s                                              # notify via email
#SBATCH --mail-user=%(email)s                                               # recipient

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

time python3 %(job_script)s --local-training-dir "$1"

""" % job_dict

    )
    bash_file.close()

else:
    print("Choose a given device (GPU or CPU)!")
    print("Stopping.")
    exit()