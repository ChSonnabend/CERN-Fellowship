import os, sys, json
import numpy as np
import argparse
import subprocess
import copy
import glob
import re

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--submit", default=1, type=int)
parser.add_argument("-opt", "--options", default="all")
parser.add_argument("-o", "--output-dir", default=";;")
parser.add_argument("-c", "--config", default="config.json")
parser.add_argument("-sc", "--submission-config", default="config_qa.json")
parser.add_argument("-w", "--write-scripts", default=1)
parser.add_argument("-rd", "--reco-digits", default=1)
parser.add_argument("-rm", "--remove-files", default=1)
parser.add_argument("-real", "--real-data", default=None)
parser.add_argument("-ad", "--alien-dir", default=None)
parser.add_argument("-d", "--download", default=1)
parser.add_argument("-aod", "--produce-aod", default=0)
parser.add_argument("-id", "--id", default=1, type=int)
parser.add_argument("-fa", "--from-aod", default=0, type=int)
parser.add_argument("-saonce", "--submit-qa-only-once", default=1, type=int)
parser.add_argument("--combine-script", default="/lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/combine_histograms.C", type=str, help="Path to the script for combining histograms")
parser.add_argument("--limit", default=-1, type=int, help="Limit the number of jobs submitted")
parser.add_argument("--performance-test-cpu", default=0, type=int, help="If set to 1, run the CPU performance test instead of the standard QA")
args = parser.parse_args()

### CONFIGURATIONS
configs_file = open(args.config, "r")
CONF = json.load(configs_file)
configs_file.close()

submit_file = open(args.submission_config, "r")
SUBMIT = json.load(submit_file)
submit_file.close()

try:
    sim_json_file = open(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "config_sim.json"), "r")
    SIM_JSON = json.load(sim_json_file)
    sim_json_file.close()
except:
    SIM_JSON = {}

slurm_dict = CONF
slurm_dict["job_settings"]["chdir"] = SUBMIT["exec_settings"]["output_dir"]
# mode = SUBMIT["qa_task"]["mode"]

if not args.real_data:
    args.real_data = SUBMIT["data_settings"]["real-data"]

if(args.output_dir != ";;"):
    output_dir = args.output_dir
else:
    output_dir = os.path.join(SUBMIT["exec_settings"]["simulation_dir"], SUBMIT["qa_task"]["mode"] + "_QA")
    output_dir = output_dir.replace(",", "_")


### SLURM & utils

job_ids = [-1]
end_of_file = "EOF"

def getMCworkflow(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Command" in line:
                task = line.split('Command "')[1].split('" successfully finished.')[0]
                break
    return task

def getHBFUtilsConfig(task):
    """Pull the HBFUtils.* settings out of a command scraped from the MC production logs.

    The reader stage is not taken from those logs but still needs the same HBFUtils
    configuration as the reconstruction stages, otherwise the DPL timer injects a wrong
    first orbit and every CCDB object is queried for the wrong timestamp.
    """
    match = re.search(r'--configKeyValues\s+"([^"]*)"', task)
    if not match:
        return ""
    return "".join(kv + ";" for kv in match.group(1).split(";") if kv.strip().startswith("HBFUtils."))

def stripRun(stage):
    """Drop "--run" so the stage serialises its topology into the pipe instead of running it.

    Only the last stage of a piped DPL chain may carry "--run": a driver that has it executes
    its own topology rather than dumping it to stdout for the next stage (see the dump branch
    in Framework/Core/src/runDataProcessing.cxx). The commands scraped from the production
    logs are standalone invocations and all carry it.
    """
    return re.sub(r"\s--run(?=\s|$)", "", stage.rstrip())

def disableRootInput(stage):
    """Suppress the ROOT readers InputHelper would inject into a downstream stage.

    In a merged workflow those readers collide with the devices that actually produce the
    data: an injected itstpc-track-reader publishing GLO/TPCITS next to the itstpc-track-matcher
    that computes it makes DPL reject the graph with "Found duplicate outputs". The flag is all
    or nothing, so everything the chain genuinely reads from disk is supplied once by the
    o2-global-track-cluster-reader stage at the head of the pipe. The reader options are
    registered by the reader specs themselves and become unrecognised once those are gone, so
    they have to be dropped along with them.
    """
    stage = re.sub(r'\s--tpc-native-cluster-reader\s+"[^"]*"', "", stage)
    stage = re.sub(r"\s--tpc-track-reader\s+\S+", "", stage)
    ### The combine flags pack a stage's reader devices into one process, so with the readers
    ### gone they leave an empty combined device behind (PV-Input-Reader, SV-Input-Reader,
    ### TOF-readers). Having no inputs to wait on, those get their run() called over and over,
    ### and it writes "Processing Combined with N threads" straight to std::cerr with no log
    ### level (DPLWorkflowUtils.h), which floods the job log.
    stage = re.sub(r"\s--combine-source-devices(?=\s|$)", "", stage)
    stage = re.sub(r"\s--combine-devices(?=\s|$)", "", stage)
    return stage + " --disable-root-input"

def write_bash_script(sd, job_name, task, submit=False, job_array=[-1], odir = output_dir, afterany=False, mode=0, cvmfs_tag="VO_ALICE@O2Physics::daily-20250920-0000-1", overwrite_name=None, overwrite_log_name=None, write_output_error=True, requeue=True, trapped_requeue=False):

    if isinstance(task, str):
        task = {"before_eof": task, "after_eof": ""}

    trap_requeue = ""
    if trapped_requeue:
        trap_requeue = "trap 'echo \"Time limit approaching — requeuing job...\"; scontrol requeue $SLURM_JOB_ID; exit 0' SIGUSR1"

    logsname = "{}/job_{}".format(sd["job_settings"]["chdir"], job_name)
    if overwrite_log_name:
        logsname = "{}/job_{}".format(sd["job_settings"]["chdir"], overwrite_log_name)

    script_name = job_name
    if overwrite_name:
        script_name = overwrite_name

    slurm_script = "#!/bin/bash\n"
    for opt, val in sd["job_settings"].items():
        slurm_script += "#SBATCH --{0}={1}\n".format(opt, val)
    slurm_script += "#SBATCH --job-name={}\n".format(job_name)
    if trapped_requeue:
        slurm_script += "#SBATCH --signal=B:SIGUSR1@60\n" # Send SIGUSR1 60 seconds before time limit
    if write_output_error:
        slurm_script += "#SBATCH --output={}.out\n".format(logsname)
        slurm_script += "#SBATCH --error={}.err\n".format(logsname)
    if requeue:
        slurm_script += "#SBATCH --requeue"

    ### O2 environment
    if mode == 0:
        slurm_script += """
MAX_RETRIES=3
RETRY_COUNT=${RETRY_COUNT:-0}

%(trap_requeue)s

echo "Retry #$RETRY_COUNT for job $SLURM_JOB_ID"

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "Max retries reached. Stopping."
    exit 1
fi

unset http_proxy
unset https_proxy

apptainer shell -B /scratch -B /lustre %(O2_container)s<<\EOF
export JALIEN_TOKEN_CERT=/%(token_dir)s/tokencert_9898.pem
export JALIEN_TOKEN_KEY=/%(token_dir)s/tokenkey_9898.pem
export ALIEN_SITE=CERN
alienv -w %(O2_dir)s enter %(O2_env)s

export ALIEN_PROC_ID=$SLURM_JOB_ID
export FAIRMQ_IPC_PREFIX=$SLURM_JOB_ID
%(task)s

TASK_STATUS=\$?
# Default to 0 if TASK_STATUS is not numeric or empty
if [ -z "\$TASK_STATUS" ] || ! [[ "\$TASK_STATUS" =~ ^[0-9]+\$ ]]; then
    TASK_STATUS=0
fi
exit $TASK_STATUS

EOF
status=$?

%(task2)s

""" % {**sd["directory_settings"], "task": task["before_eof"], "task2": task["after_eof"], "trap_requeue": trap_requeue}

        if requeue:
            slurm_script += """
if [ "${status:-1}" -ne 0 ]; then
    export RETRY_COUNT=$((RETRY_COUNT+1))
    scontrol requeue ${SLURM_JOB_ID}
fi

exit $status
""" % {**sd["directory_settings"], "task": task["before_eof"], "task2": task["after_eof"]}

    ### O2Physics environment with container
    elif mode == 1:
        slurm_script += """

MAX_RETRIES=3
RETRY_COUNT=${RETRY_COUNT:-0}

%(trap_requeue)s

echo "Retry #$RETRY_COUNT for job $SLURM_JOB_ID"

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "Max retries reached. Stopping."
    exit 1
fi

unset http_proxy
unset https_proxy

apptainer shell -B /scratch -B /lustre -B /cvmfs %(O2_container)s<<\EOF
export JALIEN_TOKEN_CERT=/%(token_dir)s/tokencert_9898.pem
export JALIEN_TOKEN_KEY=/%(token_dir)s/tokenkey_9898.pem
export ALIEN_SITE=CERN
source /cvmfs/alice.cern.ch/etc/login.sh
/cvmfs/alice.cern.ch/bin/alienv enter %(cvmfs_tag)s

export ALIEN_PROC_ID=$SLURM_JOB_ID
export FAIRMQ_IPC_PREFIX=$SLURM_JOBID
%(task)s

TASK_STATUS=\$?
# Default to 0 if TASK_STATUS is not numeric or empty
if [ -z "\$TASK_STATUS" ] || ! [[ "\$TASK_STATUS" =~ ^[0-9]+\$ ]]; then
    TASK_STATUS=0
fi
exit $TASK_STATUS

EOF
status=$?

%(task2)s

""" % {**sd["directory_settings"], "task": task["before_eof"], "task2": task["after_eof"],"cvmfs_tag": cvmfs_tag, "trap_requeue": trap_requeue}

        if requeue:
            slurm_script += """
if [ "${status:-1}" -ne 0 ]; then
    export RETRY_COUNT=$((RETRY_COUNT+1))
    scontrol requeue ${SLURM_JOB_ID}
fi

exit $status
""" % {**sd["directory_settings"], "task": task["before_eof"], "task2": task["after_eof"],"cvmfs_tag": cvmfs_tag}

    elif mode == 2:

        slurm_script += """
%(task)s
%(task2)s
""" % {**sd["directory_settings"], "task": task["before_eof"], "task2": task["after_eof"],"cvmfs_tag": cvmfs_tag}

    sh_script = os.path.join(odir, "{}.sh".format(script_name))
    bash_file = open(sh_script, "w")
    bash_file.write(slurm_script)
    bash_file.close()

    if submit:
        submission = "sbatch "
        if job_array[-1] != -1:
            if afterany:
                submission += " --dependency=afterany:{0}".format(job_array[-1])
            else:
                submission += " --dependency=afterok:{0}".format(job_array[-1])
        submission += " " + sh_script
        out = subprocess.check_output(submission, shell=True).decode().strip('\n')
        print("({}) ".format(job_name) + out)
        job_array.append(int(str(out.split(" ")[-1])))

def conv_slurm_mem(mem_string):
    if "K" in mem_string:
        return int(mem_string.replace("K", "")) * 1024
    elif "M" in mem_string:
        return int(mem_string.replace("M", "")) * 1024**2
    elif "G" in mem_string:
        return int(mem_string.replace("G", "")) * 1024**3
    else:
        print("SLURM mem_string: Returning input value")
        return int(mem_string)

### TASKS
def find_and_replace(cmd: str, key: str, new_value: str) -> str:
    """
    Replace or remove an option in a shell command string.

    CLI:
      --key
      --key value
      --key=value
    configKeyValues:
      key=value;
    key argument may be 'key', 'key=', '--key', '--key='.
    If new_value == "": remove the whole option (and its value).
    """
    key_norm = key.rstrip('=')

    if key_norm.startswith('--'):
        k = re.escape(key_norm)
        replaced = False

        # --key=VALUE
        pat_eq = re.compile(rf'(?<![\w-])({k})=(?P<val>"[^"]*"|\'[^\']*\'|[^\s"]+)')
        def repl_eq(m):
            nonlocal replaced
            replaced = True
            return f"{m.group(1)}={new_value}" if new_value else ''
        cmd, n1 = pat_eq.subn(repl_eq, cmd)

        # --key VALUE
        pat_sp = re.compile(
            rf'(?<![\w-])({k})(?:\s+)(?P<val>"[^"]*"|\'[^\']*\'|[^\s"-][^\s]*)'
        )
        def repl_sp(m):
            nonlocal replaced
            replaced = True
            return f"{m.group(1)} {new_value}" if new_value else ''
        cmd, n2 = pat_sp.subn(repl_sp, cmd)

        # Standalone flag (only if not already replaced and not followed by a value)
        # Ensure next token is another option (starts with --) or end-of-string
        if not replaced:
            pat_flag = re.compile(rf'(?<![\w-])({k})(?=(?:\s+--|$))')
            def repl_flag(m):
                return f"{m.group(1)} {new_value}" if new_value else ''
            cmd = pat_flag.sub(repl_flag, cmd)

        cmd = re.sub(r'\s{2,}', ' ', cmd).strip()
        return cmd

    # configKeyValues style
    k = re.escape(key_norm)
    if new_value:
        pat_replace = re.compile(rf'(?<![\w.-])({k})=([^;"]*)(?=;|")')
        cmd = pat_replace.sub(lambda m: f"{m.group(1)}={new_value}", cmd)
    else:
        pat_semicolon = re.compile(rf'(?<![\w.-]){k}=[^;"]*;')
        cmd = pat_semicolon.sub('', cmd)
        pat_end = re.compile(rf'(?<![\w.-]){k}=[^;"]*(?="|$)')
        cmd = pat_end.sub('', cmd)
        cmd = re.sub(r';{2,}', ';', cmd).replace(';"', '"')

    return cmd

task_dict = {
    "VARS": {
        "processing_vars": """
export DPL_REPORT_PROCESSING=1
SKIP_BAD_FILES="{2}"
ARGSALL="--session {0} --severity info --shm-segment-id {0} --shm-segment-size {1} --early-forward-policy noraw --monitoring-backend no-op:// --fairmq-rate-logging 0 --timeframes-rate-limit 2 --timeframes-rate-limit-ipcid {0}"
TRACKTUNETPCINNER="trackTuneParams.sourceLevelTPC=true;trackTuneParams.tpcCovInnerType=2;trackTuneParams.tpcCovInner[0]=0.05;trackTuneParams.tpcCovInner[1]=0.2;trackTuneParams.tpcCovInner[2]=0.0003;trackTuneParams.tpcCovInner[3]=0.0013;trackTuneParams.tpcCovInner[4]=0.0059300284;trackTuneParams.tpcCovInnerSlope[0]=1.4794650254467986e-08;trackTuneParams.tpcCovInnerSlope[1]=5.9178601017871944e-08;trackTuneParams.tpcCovInnerSlope[2]=8.87679015268079e-11;trackTuneParams.tpcCovInnerSlope[3]=3.846609066161676e-10;trackTuneParams.tpcCovInnerSlope[4]=1.7546539235412473e-09;"
TRACKTUNETPCOUTER="trackTuneParams.tpcCovOuterType=2;trackTuneParams.tpcCovOuter[0]=0.05;trackTuneParams.tpcCovOuter[1]=0.2;trackTuneParams.tpcCovOuter[2]=0.0003;trackTuneParams.tpcCovOuter[3]=0.0013;trackTuneParams.tpcCovOuter[4]=0.0059300284;trackTuneParams.tpcCovOuterSlope[0]=1.4794650254467986e-08;trackTuneParams.tpcCovOuterSlope[1]=5.9178601017871944e-08;trackTuneParams.tpcCovOuterSlope[2]=8.87679015268079e-11;trackTuneParams.tpcCovOuterSlope[3]=3.846609066161676e-10;trackTuneParams.tpcCovOuterSlope[4]=1.7546539235412473e-09;"
TPCTUNE=";GPU_rec_tpc.clusterError2AdditionalYSeeding=0.1;GPU_rec_tpc.clusterError2AdditionalZSeeding=0.15;$TRACKTUNETPCINNER;$TRACKTUNETPCOUTER;"
SVEXTRA=";svertexer.createFullV0s=true;svertexer.createFullCascades=true;"
ITSSETTINGS=";;ITSClustererParam.maxBCDiffToMaskBias=-10;ITSClustererParam.maxBCDiffToSquashBias=10;ITSCATrackerParam.deltaRof=0;ITSVertexerParam.clusterContributorsCut=16;ITSVertexerParam.lowMultBeamDistCut=0;ITSCATrackerParam.nROFsPerIterations=12;ITSCATrackerParam.perPrimaryVertexProcessing=false;ITSCATrackerParam.fataliseUponFailure=false;ITSCATrackerParam.dropTFUponFailure=true;ITSVertexerParam.nIterations=2;ITSCATrackerParam.doUPCIteration=true"
ITSTRACKSETTINGS=";ITSCATrackerParam.sysErrY2[0]=100e-8;ITSCATrackerParam.sysErrZ2[0]=100e-8;ITSCATrackerParam.sysErrY2[1]=100e-8;ITSCATrackerParam.sysErrZ2[1]=100e-8;ITSCATrackerParam.sysErrY2[2]=100e-8;ITSCATrackerParam.sysErrZ2[2]=100e-8;ITSCATrackerParam.sysErrY2[3]=100e-8;ITSCATrackerParam.sysErrZ2[3]=100e-8;ITSCATrackerParam.sysErrY2[4]=100e-8;ITSCATrackerParam.sysErrZ2[4]=100e-8;ITSCATrackerParam.sysErrY2[5]=100e-8;ITSCATrackerParam.sysErrZ2[5]=100e-8;ITSCATrackerParam.sysErrY2[6]=100e-8;ITSCATrackerParam.sysErrZ2[6]=100e-8"
ITSTPCMATCHER=";tpcitsMatch.askMinTPCRow[0]=78;tpcitsMatch.askMinTPCRow[1]=78;tpcitsMatch.askMinTPCRow[2]=78;tpcitsMatch.askMinTPCRow[3]=78;tpcitsMatch.askMinTPCRow[4]=78;tpcitsMatch.askMinTPCRow[5]=78;tpcitsMatch.askMinTPCRow[6]=78;tpcitsMatch.askMinTPCRow[7]=78;tpcitsMatch.askMinTPCRow[8]=78;tpcitsMatch.askMinTPCRow[9]=78;tpcitsMatch.askMinTPCRow[10]=78;tpcitsMatch.askMinTPCRow[11]=78;tpcitsMatch.askMinTPCRow[12]=78;tpcitsMatch.askMinTPCRow[13]=78;tpcitsMatch.askMinTPCRow[14]=78;tpcitsMatch.askMinTPCRow[15]=78;tpcitsMatch.askMinTPCRow[16]=78;tpcitsMatch.askMinTPCRow[17]=78;tpcitsMatch.askMinTPCRow[18]=78;tpcitsMatch.askMinTPCRow[19]=78;tpcitsMatch.askMinTPCRow[20]=78;tpcitsMatch.askMinTPCRow[21]=78;tpcitsMatch.askMinTPCRow[22]=78;tpcitsMatch.askMinTPCRow[23]=78;tpcitsMatch.askMinTPCRow[24]=78;tpcitsMatch.askMinTPCRow[25]=78;tpcitsMatch.askMinTPCRow[26]=78;tpcitsMatch.askMinTPCRow[27]=78;tpcitsMatch.askMinTPCRow[28]=78;tpcitsMatch.askMinTPCRow[29]=78;tpcitsMatch.askMinTPCRow[30]=78;tpcitsMatch.askMinTPCRow[31]=78;tpcitsMatch.askMinTPCRow[32]=78;tpcitsMatch.askMinTPCRow[33]=78;tpcitsMatch.askMinTPCRow[34]=78;tpcitsMatch.askMinTPCRow[35]=78;tpcitsMatch.globalTimeExtraErrorMUS=0.2"
"""
    },
    "O2": {
        "tf_to_aod": """
o2-raw-tf-reader-workflow $ARGSALL $SKIP_BAD_FILES --delay 0 --raw-only-det all --loop 0  --input-data {2}  --onlyDet ITS,TPC,TRD,TOF,FT0,CTP    --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-tfidinfo-writer-workflow $ARGSALL   --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-itsmft-stf-decoder-workflow $ARGSALL --nthreads 2 --raw-data-dumps 0 --pipeline its-stf-decoder:1  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;;VerbosityConfig.rawParserSeverity=warn;;" -b | \\
o2-ft0-flp-dpl-workflow $ARGSALL  --pipeline ft0-datareader-dpl:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-tof-compressor $ARGSALL --tof-compressor-paranoid --pipeline tof-compressor-0:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-trd-datareader $ARGSALL  --sortDigits --pipeline trd-datareader:1   -b | \\
o2-ctp-reco-workflow $ARGSALL   --ntf-to-average 1 --pipeline ctp-raw-decoder:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-tpc-scaler-workflow $ARGSALL   --enable-M-shape-correction  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-gpu-reco-workflow $ARGSALL $TPCCORROPT --gpu-reconstruction "--severity info" --input-type=zsraw --disable-mc --output-type tracks,clusters,tpc-triggers,send-clusters-per-sector --pipeline gpu-reconstruction:1,gpu-reconstruction-prepare:1   --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;GPU_global.deviceType=CPU;GPU_proc.debugLevel=0;GPU_proc.ompThreads={3};GPU_proc.deviceNum=-2;;;TPCCorrMap.lumiInstFactor=2.414;$RECOGPU;GPU_global.rundEdx=1;" -b | \\
o2-tof-reco-workflow $ARGSALL  --local-cmp --input-type raw --output-type clusters,digits --use-ccdb --disable-root-input  --disable-mc --pipeline tof-compressed-decoder:1,TOFClusterer:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-its-reco-workflow $ARGSALL --trackerCA  --tracking-mode async --disable-mc --clusters-from-upstream  --pipeline its-tracker:1,its-clusterer:1  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;ITSVertexerParam.lowMultBeamDistCut=0;ITSCATrackerParam.nROFsPerIterations=12;$ITSSETTINGS;$ITSTRACKSETTINGS;;" | \\
o2-ft0-reco-workflow $ARGSALL --disable-root-input  --disable-mc --pipeline ft0-reconstructor:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-trd-tracklet-transformer $ARGSALL --disable-irframe-reader --disable-root-input  --disable-mc  --pipeline TRDTRACKLETTRANSFORMER:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-tpcits-match-workflow $ARGSALL $TPCCORROPT --disable-root-input  --disable-mc --produce-calibration-data  --nthreads 1 --pipeline itstpc-track-matcher:1  --use-ft0 --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;;ft0tag.minAmplitudeA=5;ft0tag.minAmplitudeC=5;ft0tag.minAmplitudeAC=20;;;;TPCCorrMap.lumiInstFactor=2.414;$ITSTPCMATCHER;$ITSSETTINGS;$ITSTRACKSETTINGS;" -b | \\
o2-trd-global-tracking $ARGSALL $TPCCORROPT --disable-root-input  --disable-mc  --enable-vdexb-calib --enable-ph  --track-sources TPC,ITS-TPC --pipeline trd-globaltracking_TPC_ITS-TPC_:1,trd-globaltracking_TPC_FT0_ITS-TPC_:1,trd-globaltracking_TPC_FT0_ITS-TPC_CTP_:1  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;;ft0tag.minAmplitudeA=5;ft0tag.minAmplitudeC=5;ft0tag.minAmplitudeAC=20;;;;TPCCorrMap.lumiInstFactor=2.414;;" -b | \\
o2-tof-matcher-workflow $ARGSALL $TPCCORROPT --use-fit --disable-root-input  --disable-mc --enable-dia --tof-lanes 1 --track-sources TPC,ITS-TPC,TPC-TRD,ITS-TPC-TRD --pipeline tof-matcher:1  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;;;TPCCorrMap.lumiInstFactor=2.414;;" -b | \\
o2-tpc-reco-workflow $ARGSALL $TPCCORROPT --input-type pass-through --output-type clusters,tpc-triggers,tracks,send-clusters-per-sector --disable-mc  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-primary-vertexing-workflow $ARGSALL --disable-mc --disable-root-input   --vertexing-sources ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --vertex-track-matching-sources ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --pipeline primary-vertexing:1,pvertex-track-matching:1  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;pvertexer.acceptableScale2=9;pvertexer.minScale2=2;pvertexer.timeMarginVertexTime=1.3;pvertexer.addTimeSigma2Debris=1e-2;pvertexer.meanVertexExtraErrSelection=0.03;pvertexer.maxITSOnlyFraction=0.85;pvertexer.maxTDiffDebris=1.5;pvertexer.maxZDiffDebris=0.3;pvertexer.addZSigma2Debris=0.09;pvertexer.addTimeSigma2Debris=2.25;pvertexer.maxChi2TZDebris=100;pvertexer.maxMultRatDebris=1.;pvertexer.maxTDiffDebrisExtra=-1.;pvertexer.dbscanDeltaT=-0.55;pvertexer.maxTMAD=1.;pvertexer.maxZMAD=0.04;ft0tag.minAmplitudeA=5;ft0tag.minAmplitudeC=5;ft0tag.minAmplitudeAC=20;" -b | \\
o2-secondary-vertexing-workflow $ARGSALL $TPCCORROPT --disable-mc  --disable-root-input  --vertexing-sources ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --threads 2 --pipeline secondary-vertexing:1  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;TPCCorrMap.lumiInstFactor=2.414;$SVEXTRA;" -b | \\
o2-calibration-ft0-time-spectra-processor $ARGSALL   --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-ft0-integrate-cluster-workflow $ARGSALL   --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-tof-integrate-cluster-workflow $ARGSALL   --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-aod-producer-workflow $ARGSALL    --info-sources ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --disable-root-input --aod-writer-keep dangling --aod-writer-resfile "AO2D" --aod-writer-resmode UPDATE --disable-mc --pipeline aod-producer-workflow:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-dpl-run $ARGSALL
""",
# o2-tpc-miptrack-filter $ARGSALL --processEveryNthTF 40 --maxTracksPerTF 1000 -b | \\
# Workflow above can add before aod-producer: o2-eve-export-workflow $ARGSALL --display-tracks ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --display-clusters ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --skipOnEmptyInput --disable-root-input  --jsons-folder jsons --disable-mc  --configKeyValues "keyval.input_dir={0};keyval.output_dir={1};;;;eveconf.onlyNthEvent=2;;eveconf.maxTracks=5000;;eveconf.PVMode=true;;" -b | \\
# ctf_to_aod: CTFINPUT=1 WORKFLOWMODE=print BEAMTYPE=pp WORKFLOW_PARAMETERS=AOD /scratch/alice/csonnab/MyO2/O2/prodtests/full-system-test/dpl-workflow.sh
        "ctf_to_aod": """
o2-ctf-reader-workflow $ARGSALL --ans-version 1.0 --ctf-dict none --delay 0 --loop 0  --ctf-input {2}   --onlyDet ITS,TPC,TRD,TOF,EMC,CTP,FT0,FV0,FDD,MCH,MFT,MID,HMP --emcal-decoded-subspec 1   --pipeline tpc-entropy-decoder:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-its-reco-workflow $ARGSALL --trackerCA  --tracking-mode async --ccdb-meanvertex-seed --disable-mc --clusters-from-upstream --disable-root-output --pipeline its-tracker:1,its-clusterer:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;ITSVertexerParam.phiCut=0.5;ITSVertexerParam.clusterContributorsCut=3;ITSVertexerParam.tanLambdaCut=0.2;ITSVertexerParam.nThreads=2;ITSCATrackerParam.nThreads=2;;;$ITSSETTINGS;$ITSTRACKSETTINGS;\" -b | \\
o2-gpu-reco-workflow $ARGSALL --gpu-reconstruction \"--severity info\" --input-type=compressed-clusters-ctf --disable-mc --output-type tracks,clusters  --lumi-type 1 --pipeline gpu-reconstruction:1,gpu-reconstruction-prepare:1   --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;GPU_global.deviceType=CPU;GPU_proc.debugLevel=0;GPU_proc.tpcInputWithClusterRejection=1;GPU_proc.ompThreads=-1;GPU_proc.deviceNum=-2;;;;\" -b | \\
o2-tof-reco-workflow $ARGSALL  --local-cmp --input-type digits --output-type clusters --disable-root-input --disable-root-output --disable-mc --pipeline tof-compressed-decoder:1,TOFClusterer:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-ft0-reco-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc --pipeline ft0-reconstructor:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-trd-tracklet-transformer $ARGSALL --disable-irframe-reader --disable-root-input --disable-root-output --disable-mc  --pipeline TRDTRACKLETTRANSFORMER:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-tpcits-match-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc    --lumi-type 1 --nthreads 2 --pipeline itstpc-track-matcher:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;;;;;$ITSTPCMATCHER;$ITSSETTINGS;$ITSTRACKSETTINGS;\" -b | \\
o2-trd-global-tracking $ARGSALL --disable-root-input --disable-root-output --disable-mc  --enable-ph   --lumi-type 1 --track-sources TPC,ITS-TPC --pipeline trd-globaltracking_TPC_ITS-TPC_:1,trd-globaltracking_TPC_FT0_ITS-TPC_:1,trd-globaltracking_TPC_FT0_ITS-TPC_CTP_:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;;;;;;\" -b | \\
o2-tof-matcher-workflow $ARGSALL --use-fit --disable-root-input --disable-root-output --disable-mc  --lumi-type 1 --tof-lanes 1 --track-sources TPC,ITS-TPC,TPC-TRD,ITS-TPC-TRD --pipeline tof-matcher:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;;;;;\" -b | \\
o2-mid-reco-workflow $ARGSALL --disable-root-output --disable-mc --pipeline MIDClusterizer:1,MIDTracker:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-mch-reco-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc --pipeline mch-track-finder:1,mch-cluster-finder:1,mch-cluster-transformer:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-mft-reco-workflow $ARGSALL --disable-mc --clusters-from-upstream --disable-root-output  --nThreads 2 --pipeline mft-tracker:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;;;\" -b | \\
o2-fdd-reco-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-fv0-reco-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-hmpid-digits-to-clusters-workflow $ARGSALL --disable-root-input --disable-root-output --pipeline HMP-Clusterization:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-hmpid-matcher-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc --track-sources ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF,TPC-TRD,TPC-TOF,TPC-TRD-TOF --pipeline hmp-matcher:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-muon-tracks-matcher-workflow $ARGSALL --disable-root-input --disable-mc --disable-root-output --pipeline muon-track-matcher:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-globalfwd-matcher-workflow $ARGSALL --disable-root-input --disable-root-output --disable-mc --pipeline globalfwd-track-matcher:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;FwdMatching.useMIDMatch=true;;\" -b | \\
o2-emcal-cell-recalibrator-workflow $ARGSALL --input-subspec 1 --output-subspec 0 --redirect-led  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-emcal-cell-writer-workflow $ARGSALL --disable-mc --subspec 10 --cell-writer-name emcal-led-cells-writer --emcal-led-cells-writer \"--outfile emcledcells.root\"  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-primary-vertexing-workflow $ARGSALL --disable-mc --disable-root-input --disable-root-output  --vertexing-sources ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,MFT-MCH,MCH-MID,ITS,MFT,TPC,TOF,FT0,MID,EMC,FDD,HMP,FV0,TRD,MCH,CTP --vertex-track-matching-sources ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,MFT-MCH,MCH-MID,ITS,MFT,TPC,TOF,FT0,MID,EMC,FDD,HMP,FV0,TRD,MCH,CTP --pipeline primary-vertexing:1,pvertex-track-matching:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;pvertexer.maxChi2TZDebris=10;;;;;\" -b | \\
o2-secondary-vertexing-workflow $ARGSALL --disable-mc   --disable-root-input --disable-root-output  --lumi-type 1 --vertexing-sources ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,MFT-MCH,MCH-MID,ITS,MFT,TPC,TOF,FT0,MID,EMC,FDD,HMP,FV0,TRD,MCH,CTP --threads 2 --pipeline secondary-vertexing:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;;;\" -b | \\
o2-aod-producer-workflow $ARGSALL    --info-sources ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,MFT-MCH,MCH-MID,ITS,MFT,TPC,TOF,FT0,MID,EMC,FDD,HMP,FV0,TRD,MCH,CTP --disable-root-input --aod-writer-keep dangling --aod-writer-resfile \"AO2D\" --aod-writer-resmode UPDATE --disable-mc --pipeline aod-producer-workflow:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\
o2-dpl-run $ARGSALL
""",
        "digits_writer": """
o2-raw-tf-reader-workflow $ARGSALL --input-data {0} --onlyDet TPC,FT0,CTP | \\
o2-tpc-raw-to-digits-workflow $ARGSALL --input-spec "A:TPC/RAWDATA" --remove-duplicates --severity info --ignore-grp | \\
o2-tpc-reco-workflow $ARGS_ALL --input-type digitizer --output-type digits --no-ca-clusterer --disable-mc | \\
o2-dpl-run $ARGS_ALL --run
""",
        "mip_track_filter": """
o2-tpc-track-reader --input-type tracks --disable-mc --skip-clusref --hbfutils-config o2_tfidinfo.root | o2-tpc-miptrack-filter | o2-tpc-calib-dedx --file-dump 1
""",
    },
    "O2PHYSICS": {
        "pid_skimmer": """
o2-analysis-pid-tpc-skimscreation -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-dq-v0-selector -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-multiplicity-table -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-trackselection -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tof-merge -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-lf-strangenessbuilder -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-event-selection -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-track-propagation -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-ft0-corrected-table -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-base -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-timestamp -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} --aod-file {2} --aod-writer-json {3}
""",
        "D0": """
o2-analysis-derived-data-creator-d0-calibration -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-ft0-corrected-table -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-track-to-collision-associator -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-trackselection -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tof-full -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-multcenttable -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-event-selection-service -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-occ-table-producer -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-propagationservice -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-service -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tof-base -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} --aod-file {2} --aod-writer-json {3}
""",
# o2-analysis-trackqa-converter-002 -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
# o2-analysis-trackqa-converter-003 -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} --aod-file {2} --aod-writer-json {3}
        "create_bb_object": """
o2-pidparam-tpc-response --mode write --bb0 {0} --bb1 {1} --bb2 {2} --bb3 {3} --bb4 {4} --paramMIP 50 --paramChargeFactor 2.3 --paramnClNormalization 152 --recopass {5} --period {6} --save-to-file {7}
""",
        "postqa": """
o2-analysis-pid-tof-merge -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-ft0-corrected-table -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-qa -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-dq-v0-selector -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-multcenttable -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-event-selection-service -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-propagationservice -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-skimscreation -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-service -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-trackselection -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} --aod-file {2} --aod-writer-json {3} --readers 1
""",
        "K0S_reducedAP_postqa": """
o2-analysis-pid-tof-merge -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-ft0-corrected-table -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-qa -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-dq-v0-selector -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-multcenttable -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-event-selection-service -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-propagationservice -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-skimscreation -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-pid-tpc-service -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} | \\
o2-analysis-trackselection -b --session {4} --shm-segment-id {4} --shm-segment-size {0} --configuration json:/{1} --aod-file {2} --aod-writer-json {3} --readers 1
"""
    },
    "CLEANUP": {
        "remove_files": """
python3 /lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/remove_files_except.py --dir {0}
rm -rf /dev/shm/fmq*
"""
    }
}

### EXECUTION
if(args.write_scripts):

    max_session_id = 0
    if os.path.exists(".max_session.tmp"):
        with open(".max_session.tmp", 'r') as f:
            max_session_id = int(f.readlines()[0])
    max_session_id += 1
    tasks = {
        "before_eof": "",
        "after_eof": task_dict["CLEANUP"]["remove_files"]
    }

    if ("enableDistortionCorrection" not in SUBMIT["data_settings"].keys()) or ("enableDistortionCorrection" in SUBMIT["data_settings"].keys() and SUBMIT["data_settings"]["enableDistortionCorrection"]):
        task_dict["VARS"]["processing_vars"] += 'TPCCORROPT="--enable-M-shape-correction --lumi-type 2 --corrmap-lumi-mode 1 "\n'

    ### REAL DATA
    if args.real_data or SUBMIT["data_settings"]["real-data"]:

        write_dir_toplevel = os.path.join(output_dir, "tfs")

        if SUBMIT["analysis"]["run"] > 0 and ("mode" in SUBMIT["analysis"].keys()) and (SUBMIT["analysis"]["mode"] != 0):

            analysis_mode = SUBMIT["analysis"]["mode"]

            bbjob = [-1]
            adjusted_slurm_dict = copy.copy(slurm_dict)
            adjusted_slurm_dict["job_settings"]["chdir"] = output_dir
            adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 1
            adjusted_slurm_dict["job_settings"]["mem"] = "16G"
            adjusted_slurm_dict["job_settings"]["time"] = "20"
            bb_params = SUBMIT["data_settings"]["PID"]["bb-path"]

            config_analysis = None
            with open(os.path.join(output_dir, "configurations_" + analysis_mode + ".json"), 'r') as f:
                config_analysis = f.readlines()
            with open(os.path.join(output_dir, "configurations_" + analysis_mode + ".json"), 'w') as f:
                for l in config_analysis:
                    if SUBMIT["data_settings"]["PID"]["fetch-from-ccdb"]:
                        if "pidTPC.ccdbPath" in l:
                            f.write('"pidTPC.ccdbPath": "{0}",\n'.format(SUBMIT["data_settings"]["PID"]["bb-path"]))
                        elif "pidTPC.networkPathCCDB" in l:
                            f.write('"pidTPC.networkPathCCDB": "{0}",\n'.format(SUBMIT["data_settings"]["PID"]["nn-path"]))
                        elif "pidTPC.autofetchNetworks" in l:
                            f.write('"pidTPC.autofetchNetworks": "1",\n')
                        else:
                            f.write(l)
                    else:
                        if "pidTPC.param-file" in l:
                            f.write('"pidTPC.param-file": "{0}",\n'.format(os.path.join(output_dir, "bb_object.root")))
                        elif "pidTPC.networkPathLocally" in l:
                            f.write('"pidTPC.networkPathLocally": "{0}",\n'.format(os.path.join(output_dir, "nnpid.onnx")))
                        elif "pidTPC.autofetchNetworks" in l:
                            f.write('"pidTPC.autofetchNetworks": "0",\n')
                        else:
                            f.write(l)

            if not SUBMIT["data_settings"]["PID"]["fetch-from-ccdb"]:
                if ".txt" in bb_params:
                    bb_params = np.loadtxt(bb_params, dtype=str)
                    tasks["before_eof"] = task_dict["O2PHYSICS"]["create_bb_object"].format(bb_params[0], bb_params[1], bb_params[2], bb_params[3], bb_params[4], SUBMIT["data_settings"]["PID"]["recopass"], SUBMIT["data_settings"]["PID"]["period"], os.path.join(output_dir, "bb_object.root")) + "\n"
                    write_bash_script(adjusted_slurm_dict, "BB", tasks, args.submit, bbjob, output_dir, afterany=False, mode=0, requeue=False)
                elif ".root" in bb_params:
                    os.system("cp {0} {1}".format(bb_params, os.path.join(output_dir, "bb_object.root")))
                else:
                    print("ERROR: PID BB parameters not recognized, please provide a .txt or .root file")
                    sys.exit(1)

            adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 1
            adjusted_slurm_dict["job_settings"]["mem"] = "32G"
            adjusted_slurm_dict["job_settings"]["time"] = "1200"
            jobids = []
            for idx, ao2d in enumerate(glob.glob(os.path.join(output_dir, "**/AO2D.root"), recursive=True)):
                write_dir = os.path.dirname(ao2d)
                jobids_local = copy.copy(bbjob)
                analysis_dir = os.path.join(write_dir, analysis_mode)
                adjusted_slurm_dict["job_settings"]["chdir"] = analysis_dir
                tasks["after_eof"] = task_dict["CLEANUP"]["remove_files"].format(write_dir)
                os.makedirs(analysis_dir, exist_ok=True)
                tasks["before_eof"] = task_dict["O2PHYSICS"][analysis_mode].format(int(conv_slurm_mem(adjusted_slurm_dict["job_settings"]["mem"])*0.4), os.path.join(output_dir, "configurations_" + analysis_mode + ".json"), os.path.join(write_dir, "AO2D.root"), os.path.join(output_dir, "OutputDirector" + analysis_mode.upper() + ".json"), idx)
                write_bash_script(adjusted_slurm_dict, analysis_mode.upper(), tasks, args.submit, jobids_local, analysis_dir, afterany=False, mode=0, requeue=False)
                if idx == 0:
                    os.system("cp {0} {1}".format(os.path.join(analysis_dir, "*.sh"), write_dir_toplevel))
                jobids.append(jobids_local[-1])

            merger_dir = os.path.join(output_dir, "merged", analysis_mode)
            adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 1
            adjusted_slurm_dict["job_settings"]["mem"] = "256G"
            adjusted_slurm_dict["job_settings"]["time"] = "120"
            adjusted_slurm_dict["job_settings"]["chdir"] = merger_dir
            os.makedirs(merger_dir, exist_ok=True)
            tasks["before_eof"] = "hadd -f {0}/merged/{1}/merged_AnalysisResults.root {0}/tfs/*/{1}/AnalysisResults.root".format(output_dir, analysis_mode)
            if analysis_mode == "D0":
                find_aods = "python3 /lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/misc/findFiles.py --input-dir {0}/tfs --pattern=\"**/D0/AO2D_D0.root\" --dump {0}/merged/D0/AO2D_list.txt".format(output_dir)
                merger_aod = "o2-aod-merger --input {0}/merged/D0/AO2D_list.txt --output {0}/merged/merged_AO2D_D0.root --max-size 500000000".format(output_dir)
                tasks["before_eof"] += "\n" + find_aods + "\n" + merger_aod + "\n"
            final_job_dependency = ":".join([str(jid) for jid in jobids if jid > 0])
            write_bash_script(adjusted_slurm_dict, analysis_mode.upper() + "_MERGER", tasks, args.submit, [final_job_dependency], merger_dir, afterany=False, mode=0, requeue=False)

        elif SUBMIT["data_settings"]["CTF"]["runCTF"] == 1:
            skip_bad_files = ""
            if (SUBMIT["exec_settings"]["bad_runs"] != ";;") and (os.path.isfile(SUBMIT["exec_settings"]["bad_runs"])):
                skip_bad_files = "--run-time-span-file {0} --invert-irframe-selection".format(os.path.join(SUBMIT["exec_settings"]["output_dir"], "badRuns.txt"))
            os.system("cp {0} {1}".format(SUBMIT["exec_settings"]["TF_list"], output_dir))
            adjusted_slurm_dict = copy.copy(slurm_dict)
            jobids = []
            dataset = np.loadtxt(SUBMIT["exec_settings"]["TF_list"], dtype=str)
            dataset = np.array([dataset]) if len(dataset.shape) == 0 else dataset  # If only one entry, make it a 1D array
            for idx, ctf in enumerate(dataset):
                jobids_local = [-1]
                if (not ctf.startswith("alien://")) and ctf.endswith(".root"):
                    ctf = "alien://" + ctf
                file = ctf.split("/")[-1]
                write_dir = os.path.join(write_dir_toplevel, file.replace(".root", "").replace(".txt", ""))

                tasks["after_eof"] = task_dict["CLEANUP"]["remove_files"].format(write_dir)
                os.makedirs(write_dir, exist_ok=True)
                adjusted_slurm_dict["job_settings"]["chdir"] = write_dir

                adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 40
                adjusted_slurm_dict["job_settings"]["mem"] = "256G"
                adjusted_slurm_dict["job_settings"]["time"] = "120"

                if args.from_aod <= 0:
                    tasks["before_eof"] = task_dict["VARS"]["processing_vars"].format(int(idx) + 1 + max_session_id, int(conv_slurm_mem(adjusted_slurm_dict["job_settings"]["mem"])*0.4), skip_bad_files) + "\n" + task_dict["O2"]["ctf_to_aod"].format(write_dir, write_dir, ctf, int(int(adjusted_slurm_dict["job_settings"]["cpus-per-task"])*0.95))
                    write_bash_script(adjusted_slurm_dict, "RECO_CTF", tasks, args.submit, jobids_local, write_dir)
                max_session_id += 1

        else:
            reco_args = "RECOGPU=\"GPU_global.deviceType=CPU;GPU_proc.debugLevel=0;GPU_global.synchronousProcessing=1;GPU_proc.clearO2OutputFromGPU=1;GPU_proc.tpcInputWithClusterRejection=1;GPU_proc.ompThreads=-1;GPU_proc.deviceNum=-2;;;TPCCorrMap.lumiInstFactor=2.414;GPU_global.dEdxDisableResidualGainMap=1;;GPU_global.overrideNHbfPerTF=128;;".format(int(conv_slurm_mem(slurm_dict["job_settings"]["mem"])*0.4)) #;GPU_proc.forceHostMemoryPoolSize={0}
            for k1, v1 in SUBMIT["reco_task"]["input-digits"]["configKeyValues"].items():
                for k2, v2 in v1.items():
                    if "runQA" in k2:
                        continue
                    else:
                        reco_args += str(k1) + "." + str(k2) + '=' + str(v2) + ';'
            reco_args += ";$TPCTUNE\""
            task_dict["VARS"]["processing_vars"] += reco_args
            skip_bad_files = ""
            if (SUBMIT["exec_settings"]["bad_runs"] != ";;") and (os.path.isfile(SUBMIT["exec_settings"]["bad_runs"])):
                skip_bad_files = "--run-time-span-file {0} --invert-irframe-selection".format(os.path.join(SUBMIT["exec_settings"]["output_dir"], "badRuns.txt"))

            if (SUBMIT["exec_settings"]["TF_list"] == ";;"):
                adjusted_slurm_dict = copy.copy(slurm_dict)
                jobids = []
                for idx, tf in enumerate(glob.glob(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "**/*.tf"), recursive=True)):
                    file = tf.split("/")[-1]
                    write_dir = os.path.join(write_dir_toplevel, file.replace(".tf", ""))
                    tasks["after_eof"] = task_dict["CLEANUP"]["remove_files"].format(write_dir)
                    os.makedirs(write_dir, exist_ok=True)
                    adjusted_slurm_dict["job_settings"]["chdir"] = write_dir

                    jobids_local = [-1]

                    secvtx_script = "root -l -b -q '" + CONF["submission"]["secvtx_script"] + "(\"" + write_dir + "\", \"" + write_dir + "\")'"
                    tracks_script = "root -l -b -q '" + CONF["submission"]["tracks_script"] + "(\"" + write_dir + "\")'"

                    tasks["before_eof"] = task_dict["VARS"]["processing_vars"].format(int(idx) + 1 + max_session_id, int(conv_slurm_mem(adjusted_slurm_dict["job_settings"]["mem"])*0.4), skip_bad_files) + "\n" + task_dict["O2"]["tf_to_aod"].format(write_dir, write_dir, tf, int(int(adjusted_slurm_dict["job_settings"]["cpus-per-task"])*0.95))
                    adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 40
                    adjusted_slurm_dict["job_settings"]["mem"] = "256G"
                    adjusted_slurm_dict["job_settings"]["time"] = "120"
                    write_bash_script(adjusted_slurm_dict, "RECO_TF", tasks, args.submit, jobids_local, write_dir, requeue=True)

                    tasks["before_eof"] = secvtx_script + "\n" + tracks_script
                    adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 8
                    adjusted_slurm_dict["job_settings"]["mem"] = "128G"
                    adjusted_slurm_dict["job_settings"]["time"] = "30"
                    write_bash_script(adjusted_slurm_dict, "QA", tasks, args.submit, jobids_local, write_dir, afterany=False, requeue=False)

                    if SUBMIT["analysis"]["run"] > 0:
                        adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 2
                        adjusted_slurm_dict["job_settings"]["mem"] = "128G"
                        adjusted_slurm_dict["job_settings"]["time"] = "120"
                        tasks["before_eof"] = task_dict["O2PHYSICS"]["pid_skimmer"].format(int(conv_slurm_mem(adjusted_slurm_dict["job_settings"]["mem"])*0.4), os.path.join(output_dir, "configurations_pid.json"), os.path.join(write_dir, "AO2D.root"), os.path.join(output_dir, "OutputDirector.json"), idx)
                        write_bash_script(adjusted_slurm_dict, "PID", tasks, args.submit, jobids_local, write_dir, afterany=False, mode=0, requeue=False)

                    if idx == 0:
                        os.system("cp {0} {1}".format(os.path.join(write_dir, "*.sh"), write_dir_toplevel))
                    # write_bash_script(adjusted_slurm_dict, "CLEANUP", task_dict["CLEANUP"]["remove_files"].format(write_dir), args.submit, jobids, write_dir_toplevel, afterany=True, requeue=False)
                    jobids.append(jobids_local[-1])
                    max_session_id += 1
            else:
                os.system("cp {0} {1}".format(SUBMIT["exec_settings"]["TF_list"], output_dir))
                if SUBMIT["exec_settings"]["submit_per_tf"] or ("submit_per_tf" not in SUBMIT["exec_settings"]):
                    adjusted_slurm_dict = copy.copy(slurm_dict)
                    jobids = []
                    dataset = np.loadtxt(SUBMIT["exec_settings"]["TF_list"], dtype=str)
                    dataset = np.array([dataset]) if len(dataset.shape) == 0 else dataset  # If only one entry, make it a 1D array
                    for idx, tf in enumerate(dataset):
                        jobids_local = [-1]
                        if (not tf.startswith("alien://")) and tf.endswith(".tf"):
                            tf = "alien://" + tf
                        file = tf.split("/")[-1]
                        write_dir = os.path.join(write_dir_toplevel, file.replace(".tf", "").replace(".txt", ""))
                        if args.from_aod > 0:
                            write_dir = os.path.join(os.path.dirname(write_dir.replace(".root", "")), str(idx))

                        tasks["after_eof"] = task_dict["CLEANUP"]["remove_files"].format(write_dir)
                        os.makedirs(write_dir, exist_ok=True)
                        adjusted_slurm_dict["job_settings"]["chdir"] = write_dir

                        secvtx_script = "root -l -b -q '" + CONF["submission"]["secvtx_script"] + "(\"" + write_dir + "\", \"" + write_dir + "\")'"
                        tracks_script = "root -l -b -q '" + CONF["submission"]["tracks_script"] + "(\"" + write_dir + "\")'"

                        adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 40
                        adjusted_slurm_dict["job_settings"]["mem"] = "256G"
                        adjusted_slurm_dict["job_settings"]["time"] = "120"

                        if args.from_aod <= 0:
                            tasks["before_eof"] = task_dict["VARS"]["processing_vars"].format(int(idx) + 1 + max_session_id, int(conv_slurm_mem(adjusted_slurm_dict["job_settings"]["mem"])*0.4), skip_bad_files) + "\n" + task_dict["O2"]["tf_to_aod"].format(write_dir, write_dir, tf, int(int(adjusted_slurm_dict["job_settings"]["cpus-per-task"])*0.95))
                            write_bash_script(adjusted_slurm_dict, "RECO_TF", tasks, args.submit, jobids_local, write_dir)

                        if SUBMIT["analysis"]["run"] > 0:
                            aod_dir = os.path.join(write_dir, "AO2D.root")
                            if args.from_aod > 0:
                                aod_dir = tf
                            adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 2
                            adjusted_slurm_dict["job_settings"]["mem"] = "128G"
                            adjusted_slurm_dict["job_settings"]["time"] = "120"
                            tasks["before_eof"] = task_dict["O2PHYSICS"]["pid_skimmer"].format(int(conv_slurm_mem(adjusted_slurm_dict["job_settings"]["mem"])*0.4), os.path.join(output_dir, "configurations_pid.json"), aod_dir, os.path.join(output_dir, "OutputDirector.json"), idx)
                            write_bash_script(adjusted_slurm_dict, "PID", tasks, args.submit, jobids_local, write_dir, afterany=False, mode=1)

                            # tasks["before_eof"] = secvtx_script + "\n" + tracks_script
                            # adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 8
                            # adjusted_slurm_dict["job_settings"]["mem"] = "128G"
                            # adjusted_slurm_dict["job_settings"]["time"] = "30"
                            # write_bash_script(adjusted_slurm_dict, "ROOTQA", tasks, args.submit, jobids_local, write_dir, afterany=False)
                        # else:
                        #     tasks["before_eof"] = secvtx_script + "\n" + tracks_script
                        #     adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 8
                        #     adjusted_slurm_dict["job_settings"]["mem"] = "128G"
                        #     adjusted_slurm_dict["job_settings"]["time"] = "30"
                        #     write_bash_script(adjusted_slurm_dict, "QA", tasks, args.submit, jobids_local, write_dir, afterany=False)

                        if idx == 0:
                            os.system("cp {0} {1}".format(os.path.join(write_dir, "*.sh"), write_dir_toplevel))
                        jobids.append(jobids_local[-1])
                        # write_bash_script(adjusted_slurm_dict, "CLEANUP", task_dict["CLEANUP"]["remove_files"].format(write_dir), args.submit, jobids, write_dir_toplevel, afterany=True)
                else:
                    tf_list = []
                    with open(SUBMIT["exec_settings"]["TF_list"], "r") as f:
                        tf_list = [line.strip() for line in f]  # strips '\n'

                    for idx, tf in enumerate(tf_list):
                        if (not tf.startswith("alien://")) and tf.endswith(".tf"):
                            tf_list[idx] = "alien://" + tf

                    os.makedirs(output_dir, exist_ok=True)  # Move before file writing, just to be safe
                    np.savetxt(os.path.join(output_dir, "TF_list.txt"), tf_list, fmt="%s")
                    write_dir = os.path.join(output_dir, "tfs")
                    slurm_dict["job_settings"]["chdir"] = write_dir
                    tasks["after_eof"] = tasks["after_eof"].format(write_dir)
                    jobids = [-1]

                    secvtx_script = "root -l -b -q '" + CONF["submission"]["secvtx_script"] + "(\"" + write_dir + "\", \"" + write_dir + "\")'"
                    tracks_script = "root -l -b -q '" + CONF["submission"]["tracks_script"] + "(\"" + write_dir + "\")'"

                    tasks["before_eof"] = task_dict["VARS"]["processing_vars"].format(1 + max_session_id, int(conv_slurm_mem(slurm_dict["job_settings"]["mem"])*0.4), skip_bad_files) + "\n" + task_dict["O2"]["tf_to_aod"].format(write_dir, write_dir, os.path.join(output_dir, "TF_list.txt"), int(int(slurm_dict["job_settings"]["cpus-per-task"])*0.4))
                    write_bash_script(slurm_dict, "RECO_TF", tasks, args.submit, jobids, output_dir)

                    tasks["after_eof"] = secvtx_script + "\n" + tracks_script
                    write_bash_script(slurm_dict, "MIP", tasks, args.submit, jobids, output_dir, afterany=True)

                    # write_bash_script(slurm_dict, "CLEANUP", task_dict["CLEANUP"]["remove_files"].format(output_dir), args.submit, jobids, output_dir, afterany=True)
                    max_session_id += 1

            if ("run_merger" in SUBMIT["data_settings"].keys()) and SUBMIT["data_settings"]["run-merger"] > 0:
                os.makedirs("{0}/merged".format(output_dir), exist_ok=True)
                adjusted_slurm_dict = copy.copy(slurm_dict)
                adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 2
                adjusted_slurm_dict["job_settings"]["mem"] = "256G"
                adjusted_slurm_dict["job_settings"]["time"] = "480"
                adjusted_slurm_dict["job_settings"]["chdir"] = output_dir + "/merged"
                merger_analysisresults = "hadd -f {0}/merged/merged_AnalysisResults.root {0}/tfs/*/AnalysisResults.root".format(output_dir)
                merger_custom_track_qa = "hadd -f {0}/merged/merged_custom_track_qa.root {0}/tfs/*/custom_track_qa.root".format(output_dir)
                find_aods = "python3 /lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/misc/findFiles.py --input-dir {0}/tfs --pattern=\"**/AO2D_new.root\" --dump {0}/merged/AO2D_list.txt".format(output_dir)
                merger_aod = "o2-aod-merger --input {0}/merged/AO2D_list.txt --output {0}/merged/merged_AO2D_new.root --max-size 500000000".format(output_dir)
                final_job_dependency = ":".join([str(jid) for jid in jobids if jid > 0])
                write_bash_script(adjusted_slurm_dict, "FINAL_MERGE", merger_analysisresults + "\n" + merger_custom_track_qa + "\n" + find_aods + "\n" + merger_aod, args.submit, [final_job_dependency], output_dir + "/merged", afterany=True, requeue=False)

            with open(".max_session.tmp", 'w') as f:
                f.write(str(max_session_id))

    ### SIMULATION
    else:
        basic_reco_task = "o2-tpc-reco-workflow --input-type digits --output-type clusters,tracks --tpc-digit-reader \"--infile {0}/tpcdigits.root\" --tpc-native-cluster-writer \" --outfile {1}/tpc-native-clusters.root\" --shm-segment-size 100000000000".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        basic_reco_task_clusters = "o2-tpc-reco-workflow --input-type clusters --output-type tracks --shm-segment-size 100000000000"
        basic_reco_sc = "o2-tpc-reco-workflow -b --run --condition-not-after 3385078236000 --shm-segment-size 100000000000 --input-type digits --output-type clusters,tracks --tpc-digit-reader \"--infile {0}/tpcdigits.root\" --tpc-native-cluster-writer \" --outfile {1}/tpc-native-clusters.root\" --configKeyValues \";;HBFUtils.orbitFirstSampled=256;HBFUtils.nHBFPerTF=32;HBFUtils.orbitFirst=256;HBFUtils.runNumber=544116;HBFUtils.startTime=1696549278654;GPU_global.deviceType=CPU;GPU_global.dEdxDisableResidualGainMap=1;TPCGasParam.DriftV=2.58;GPU_proc.ompThreads=8;TPCCorrMap.lumiInst=2557522.0;keyval.input_dir={0};keyval.output_dir={1};\" --corrmap-lumi-mode 2 --tpc-mc-time-gain".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        calib_task_1 = "o2-global-track-cluster-reader --track-types TPC --cluster-types TPC | o2-tfidinfo-writer-workflow"
        calib_task_2 = "o2-tpc-track-reader --input-type tracks --disable-mc --skip-clusref --infile " + os.path.join(output_dir, "tpctracks.root") + " --hbfutils-config " + os.path.join(output_dir, "o2_tfidinfo.root ") + " | \\"
        calib_task_2 += "\no2-tpc-miptrack-filter | \\"
        calib_task_2 += "\no2-tpc-calib-dedx --file-dump 1"
        valid_elements_script = "root -l -b -q '" + CONF["submission"]["valid_elements_script"] + "(\"" + os.path.join(output_dir, "outITSTPCmatchingQC.root") + "\", \"" + os.path.join(output_dir, "custom_its_tpc_matching_qc.root") + "\")'"
        secvtx_script = "root -l -b -q '" + CONF["submission"]["secvtx_script"] + "(\"" + output_dir + "\", \"" + output_dir + "\")'"

        mode = SUBMIT["qa_task"]["mode"]
        if("native" in mode):
            run_afterburner_qa_macro = "root -l -b -q '/lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/qaAfterburner.C(\"native\", \"{}\")'".format(output_dir)
        if("network_" in mode):
            run_afterburner_qa_macro = "root -l -b -q '/lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/qaAfterburner.C(\"network\", \"{}\")'".format(output_dir)
        run_clusters_macro = "root -l -b /opt/alibuild/O2/Detectors/TPC/qc/macro/runClusters.C"
        run_tracks_macro = "root -l -b /opt/alibuild/O2/Detectors/TPC/qc/macro/runTracks.C"
        run_pid_macro = "root -l -b /opt/alibuild/O2/Detectors/TPC/qc/macro/runPID.C"

        copyfiles = "mkdir {1}/qed\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        copyfiles += "cp {0}/o2simdigitizerworkflow_configuration.ini {1}\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        copyfiles += "cp {0}/tpctriggers.root {1}\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        copyfiles += "cp {0}/ctpdigits.root {1}\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        copyfiles += "cp {0}/collisioncontext.root {1}\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        copyfiles += "cp {0}/o2sim_Kine.root {1}/qed\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        copyfiles += "cp {0}/o2sim_Kine.root {1}\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
        ### Never copy the production's o2match_itstpc.root into the output dir: the ITS-TPC matches
        ### have to be produced from the clusters of the clusterizer under test, otherwise every
        ### variant is compared against the same reference matches and the QC shows no difference.

        os.makedirs(os.path.join(output_dir, "networks"), exist_ok=True)

        if args.performance_test_cpu:

            if len(glob.glob(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "**/tpcreco_1.log_done"), recursive=True)) > 0:
                ### Needs the following log-files in the simulation folder:
                ### tpcreco_1.log -> For magnetic field extraction
                ### tpcreco_1.log_done
                ### itstpcMatch_1.log_done
                ### trdreco2_1.log_done
                ### toftpcmatch_1.log_done
                ### pvfinder_1.log_done
                ### svfinder_1.log_done
                ### Optional: tpcclusterpart0_1.log_done,tpcclustermerge_1.log_done

                tasks = {
                    "before_eof": "",
                    "after_eof": task_dict["CLEANUP"]["remove_files"]
                }

                toplevel_output_dir = copy.copy(output_dir)
                output_dir = os.path.join(output_dir, "reco")
                slurm_dict["job_settings"]["chdir"] = output_dir
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(toplevel_output_dir, "QA"), exist_ok=True)
                os.makedirs(os.path.join(toplevel_output_dir, "tmp"), exist_ok=True)

                reco_files_found = glob.glob(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "**/tpcreco_1.log_done"), recursive=True)
                reco_files_submit = reco_files_found

                if args.limit > 0:
                    reco_files_submit = reco_files_found[:args.limit]

                for dig_idx, digits in enumerate(reco_files_submit):

                    simulation_dir = os.path.dirname(digits)
                    print("\n-> Submitting for directory:", simulation_dir)
                    if ("reco" in args.options) or ("all" in args.options): # First, reco with SC correction for tracking efficiencies, etc.

                        final_task = ""

                        copyfiles = "cp {0}/o2simdigitizerworkflow_configuration.ini {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/tpctriggers.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/ctpdigits.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/collisioncontext.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2sim_Kine.root {1}/qed\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2sim_Kine.root {1}\n".format(simulation_dir, output_dir)
                        # copyfiles += "cp {0}/tpc_driftime_digits_lane*.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/tpc_driftime_digits*.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/sgn_Kine.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/qed_Kine.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp -r {0}/../ccdb {1}/ccdb_tmp\n".format(simulation_dir, output_dir)

                        init_task = copyfiles + "\nexport ALICEO2_CCDB_LOCALCACHE={0}/ccdb_tmp\n".format(output_dir)
                        final_task += init_task

                        # for i, clusterer in enumerate(glob.glob(os.path.join(simulation_dir, "tpcclusterpart*.log_done"))):
                        for i, clusterer in enumerate(glob.glob(os.path.join(simulation_dir, "tpcclusterpart0_1.log_done"))):

                            clusterer_task = getMCworkflow(clusterer)
                            clusterer_task = clusterer_task.replace("--run", "--run --resources-monitoring 2")

                            reco_settings_dictionary = SUBMIT["reco_task"]["input-digits"]
                            configKeyValues = ""
                            for key, val in reco_settings_dictionary.items():
                                if key=="configKeyValues":
                                    for k1, v1 in val.items():
                                        for k2, v2 in v1.items():
                                            if "nnClassificationPath" in k2:
                                                if int(v1["applyNNclusterizer"]) == 1:
                                                    new_value = ""
                                                    num_nets = len(str(v2).split(":"))
                                                    for i,net in enumerate(str(v2).split(":")):
                                                        if (dig_idx == 0):
                                                            os.system("cp {0} {1}".format(net, os.path.join(toplevel_output_dir, "networks/net_classification_" + str(i) + ".onnx")))
                                                        new_value += os.path.join(toplevel_output_dir, "networks/net_classification_" + str(i) + ".onnx")
                                                        if i != num_nets-1:
                                                            new_value += ":"
                                                    configKeyValues += str(k1) + '.' + str(k2) + '=' + str(new_value) + ';'
                                            elif "nnRegressionPath" in k2:
                                                if (int(v1["applyNNclusterizer"]) == 1) and (int(v1["nnClusterizerUseCfRegression"]) == 0):
                                                    new_value = ""
                                                    num_nets = len(str(v2).split(":"))
                                                    for i,net in enumerate(str(v2).split(":")):
                                                        if (dig_idx == 0):
                                                            os.system("cp {0} {1}".format(net, os.path.join(toplevel_output_dir, "networks/net_regression_" + str(i) + ".onnx")))
                                                        new_value += os.path.join(toplevel_output_dir, "networks/net_regression_" + str(i) + ".onnx")
                                                        if i != num_nets-1:
                                                            new_value += ":"
                                                    configKeyValues += str(k1) + '.' + str(k2) + '=' + str(new_value) + ';'
                                            elif ("runQA" in k2):
                                                continue
                                            else:
                                                configKeyValues += str(k1) + '.' + str(k2) + '=' + str(v2) + ';'
                            clusterer_task = clusterer_task.replace("--configKeyValues \"", "--configKeyValues \"" + configKeyValues)
                            final_task += "echo | " + clusterer_task + "\n" #" && \ \n"
                            final_task += "mv {0}/performanceMetrics.json {1}/performanceMetrics_{2}.json".format(output_dir, toplevel_output_dir + "/tmp", clusterer.split("/")[-1].split(".")[0]) + "\n" #" && \ \n"

                    cluster_merger = getMCworkflow(os.path.join(simulation_dir, "tpcclustermerge_1.log_done"))
                    final_task += "echo | " + cluster_merger + "\n" #" && \ \n"

                    reco_task = getMCworkflow(os.path.join(simulation_dir, "tpcreco_1.log_done"))
                    reco_task = reco_task.replace("--run", "--run --resources-monitoring 2")
                    final_task += "echo | " + reco_task + "\n" #" && \ \n"
                    final_task += "mv {0}/performanceMetrics.json {1}/performanceMetrics_tpcreco_1.json\n".format(output_dir, toplevel_output_dir + "/tmp")
                    final_task += "rm -rf {0}/ccdb_tmp\n".format(output_dir)

                    tasks["before_eof"] = final_task
                    tasks["after_eof"] = task_dict["CLEANUP"]["remove_files"].format(output_dir)

                    adjusted_slurm_dict = copy.copy(slurm_dict)
                    adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 8
                    adjusted_slurm_dict["job_settings"]["mem"] = "32G"
                    adjusted_slurm_dict["job_settings"]["time"] = "480"
                    job_name = "PERF_TEST_CPU_RECO_{}".format(dig_idx)
                    write_bash_script(adjusted_slurm_dict, job_name, tasks, args.submit, job_ids, odir=toplevel_output_dir, trapped_requeue=True)

        else:

            if len(glob.glob(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "**/reco_NOGPU.sh"), recursive=True)) > 0:
                ### Get the magnetic field from digi.log
                grep_mag_field = subprocess.check_output("grep -m1 Solenoid {}".format(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "digi.log")), shell=True)
                grep_mag_field = float(grep_mag_field.decode().strip('\n').split(" kG")[0].split(" ")[-1].strip().split("(")[-1].split("*)")[0])*5.

                qa_task = "o2-tpc-qa-clusters -b --shm-segment-id {0} --session {0}".format(args.id)
                for key, val in SUBMIT["qa_task"].items():
                    # qa_task += " --" + str(key) + ' "' + str(val) + '"'
                    if "magnetic-field" in key:
                        qa_task += " --" + str(key) + ' "' + str(grep_mag_field) + '"'
                    else:
                        qa_task += " --" + str(key) + ' "' + str(val) + '"'
                qa_task += " --simulation-path " + SUBMIT["exec_settings"]["simulation_dir"]
                qa_task += " --output-path " + output_dir
                qa_task += " --real-data " + str(SUBMIT["data_settings"]["real-data"])
                qa_task += " --infile-digits " + os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "tpcdigits.root")

                if (("reco" in args.options) or ("all" in args.options)) and ("training_data" not in mode):

                    reco_simple = ""

                    ### RECO
                    reco_settings_dictionary = SUBMIT["reco_task"]["input-digits"]

                    ### Get the distortion setting
                    # distortion_type = 0
                    # try:
                    #     distortion_type = int(SIM_JSON["simulation_settings"][SIM_JSON["simulation_settings"]["type"]]["DISTORTIONS_TYPE"])
                    # except Exception as e:
                    #     print("Problem fetching distortion type from simulation settings, using default value 0")


                    reco_task = copyfiles + "\n"
                    new_reco_line = None
                    with open(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "reco_NOGPU.sh"), "r") as f:
                        lines = f.readlines()
                        found_first_tpc_reco = False
                        for line in lines:
                            # if ("lumi-type" in line) and ("corrmap-lumi-mode" not in line):
                            #     if distortion_type == 0:
                            #         line = line.replace("--lumi-type 1", "--lumi-type 1 --corrmap-lumi-mode {}".format(distortion_type))
                            if line[:3] == "o2-" or line[:7] == "SVEXTRA":
                                if "o2-gpu-reco-workflow" in line:
                                # if "o2-tpc-reco-workflow" in line and not found_first_tpc_reco:
                                    found_first_tpc_reco = True
                                    reco_line = line
                                    configKeyValues = ""
                                    for key, val in reco_settings_dictionary.items():
                                        if key=="configKeyValues":
                                            for k1, v1 in val.items():
                                                for k2, v2 in v1.items():
                                                    if "nnClassificationPath" in k2:
                                                        if int(v1["applyNNclusterizer"]) == 1:
                                                            new_value = ""
                                                            num_nets = len(str(v2).split(":"))
                                                            for i,net in enumerate(str(v2).split(":")):
                                                                os.system("cp {0} {1}".format(net, os.path.join(output_dir, "networks/net_classification_" + str(i) + ".onnx")))
                                                                new_value += os.path.join(output_dir, "networks/net_classification_" + str(i) + ".onnx")
                                                                if i != num_nets-1:
                                                                    new_value += ":"
                                                            configKeyValues += str(k1) + '.' + str(k2) + '=' + str(new_value) + ';'
                                                    elif "nnRegressionPath" in k2:
                                                        if (int(v1["applyNNclusterizer"]) == 1) and (int(v1["nnClusterizerUseCfRegression"]) == 0):
                                                            new_value = ""
                                                            num_nets = len(str(v2).split(":"))
                                                            for i,net in enumerate(str(v2).split(":")):
                                                                os.system("cp {0} {1}".format(net, os.path.join(output_dir, "networks/net_regression_" + str(i) + ".onnx")))
                                                                new_value += os.path.join(output_dir, "networks/net_regression_" + str(i) + ".onnx")
                                                                if i != num_nets-1:
                                                                    new_value += ":"
                                                            configKeyValues += str(k1) + '.' + str(k2) + '=' + str(new_value) + ';'
                                                    elif ("runQA" in k2) or ("rundEdx" in k2):
                                                        continue
                                                    else:
                                                        configKeyValues += str(k1) + '.' + str(k2) + '=' + str(v2) + ';'

                                    # configKeyValues +=  'GPU_global.dEdxCorrFile=' + output_dir + '/calibdEdx.root;'
                                    new_reco_line = reco_line.replace("--configKeyValues \"", "--resources-monitoring 2 --configKeyValues \"" + configKeyValues)
                                    reco_simple += new_reco_line
                                    # new_reco_line = new_reco_line.replace("--shm-segment-size 100000000000", "--shm-segment-size 50000000000")
                                    reco_task += new_reco_line
                                elif "o2-dpl-run" in line:
                                    ### Adding ITS-TPC matching workflow
                                    reco_task += "o2-itstpc-matching-qc --session default --severity info --shm-segment-id 0 --shm-segment-size 100000000000  --early-forward-policy noraw --monitoring-backend no-op:// --fairmq-rate-logging 0 --timeframes-rate-limit 24 --timeframes-rate-limit-ipcid 0 --nSlicesTF 11 --disable-mc --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" | \\\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
                                    # reco_task += "o2-aod-producer-workflow --session default --severity info --shm-segment-id 0 --shm-segment-size 10000000000  --early-forward-policy noraw --monitoring-backend no-op:// --fairmq-rate-logging 0 --timeframes-rate-limit 24 --timeframes-rate-limit-ipcid 0    --info-sources ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF,TPC-TRD-TOF,ITS-TPC-TRD-TOF,ITS,TPC,TOF,FT0,TRD,CTP --disable-root-input --aod-writer-keep dangling --aod-writer-resfile \"AO2D\" --aod-writer-resmode UPDATE --disable-mc --pipeline aod-producer-workflow:1  --configKeyValues \"keyval.input_dir={0};keyval.output_dir={1};;\" -b | \\\n".format(SUBMIT["exec_settings"]["simulation_dir"], output_dir)
                                    reco_task += line
                                elif "o2-ctf-writer-workflow" in line:
                                    continue
                                else:
                                    reco_task += line

                        # reco_task = reco_nogpu

                        reco_task = reco_task.replace("/dev/null", output_dir)
                        reco_task = reco_task.replace("--timeframes-rate-limit-ipcid 0", "--timeframes-rate-limit-ipcid {}".format(args.id))
                        reco_task = reco_task.replace("--shm-segment-id 0", "--shm-segment-id {}".format(args.id))
                        reco_task = reco_task.replace("--session default", "--session {}".format(args.id))

                    ### Get the CTF file from the simulation directory. Production of AODs from raw files is currently not supported in O2 (24.05.2025)
                    def get_latest_ctf_file(path, regex_pattern="o2_ctf*.root"):
                        files = glob.glob(os.path.join(path, regex_pattern))
                        if not files:
                            return None
                        latest_file = max(files, key=os.path.getctime)
                        return latest_file

                    ctf_file = get_latest_ctf_file(SUBMIT["exec_settings"]["simulation_dir"])
                    if ctf_file:
                        aod_production = task_dict["O2"]["ctf_to_aod"].format(SUBMIT["exec_settings"]["simulation_dir"], output_dir, int(SIM_JSON["simulation_settings"][SIM_JSON["simulation_settings"]["type"]]["RUNNUMBER"])).replace("$CTFINPUT", ctf_file)
                    else:
                        print("(ERROR) No CTF file found in the simulation directory. AOD production will not be performed.")
                    aod_production = aod_production.replace("/dev/null", output_dir)
                    aod_production = aod_production.replace("--timeframes-rate-limit-ipcid 0", "--timeframes-rate-limit-ipcid {}".format(args.id))
                    aod_production = aod_production.replace("--shm-segment-id 0", "--shm-segment-id {}".format(args.id))
                    aod_production = aod_production.replace("--session default", "--session {}".format(args.id))
                    aod_production = find_and_replace(aod_production, "--shm-segment-size", int(conv_slurm_mem(slurm_dict["job_settings"]["mem"])*0.4))

                    reco_simple = reco_simple.replace("\"--severity info\"", "")
                    reco_simple = reco_simple.replace(" | \\", "")
                    reco_simple = reco_simple.replace("o2-gpu-reco-workflow", "o2-tpc-reco-workflow")
                    # if SUBMIT["data_settings"]["isSC"]:
                    #     reco_simple = reco_simple.replace("o2-tpc-reco-workflow", "o2-tpc-reco-workflow --lumi-type 2  --corrmap-lumi-mode 1")
                    reco_simple = find_and_replace(reco_simple, "--input-type", "digits --tpc-digit-reader \"--infile {0}/tpcdigits.root\"".format(SUBMIT["exec_settings"]["simulation_dir"]))
                    reco_simple = find_and_replace(reco_simple, "--output-type", "clusters,tracks,send-clusters-per-sector")
                    reco_simple = find_and_replace(reco_simple, "--shm-segment-size", int(conv_slurm_mem(slurm_dict["job_settings"]["mem"])*0.4))
                    reco_simple = find_and_replace(reco_simple, "--disable-mc", "")
                    reco_simple = find_and_replace(reco_simple, "--disable-root-output", "")
                    reco_simple = find_and_replace(reco_simple, "--gpu-reconstruction", "")
                    reco_simple = find_and_replace(reco_simple, "--early-forward-policy", "")
                    reco_simple = find_and_replace(reco_simple, "--monitoring-backend", "")
                    reco_simple = find_and_replace(reco_simple, "--fairmq-rate-logging", "")
                    reco_simple = find_and_replace(reco_simple, "--timeframes-rate-limit", "")
                    reco_simple = find_and_replace(reco_simple, "--resource-monitoring", "")
                    reco_simple = find_and_replace(reco_simple, "--pipeline", "")
                    reco_simple = find_and_replace(reco_simple, "GPU_proc.forceHostMemoryPoolSize", "")
                    reco_simple = reco_simple.replace("GPU_proc_nn.applyNNclusterizer", "GPU_proc.runQA=1;GPU_global.rundEdx=1;GPU_proc_nn.applyNNclusterizer")
                    reco_simple = reco_simple.replace("--timeframes-rate-limit-ipcid 0", "--timeframes-rate-limit-ipcid {}".format(args.id))
                    reco_simple = reco_simple.replace("--shm-segment-id 0", "--shm-segment-id {}".format(args.id))
                    reco_simple = reco_simple.replace("--session default", "--session {}".format(args.id))
                    reco_simple = reco_simple.replace("=--", "--")

                    if args.produce_aod:
                        write_bash_script(slurm_dict, "AOD_PRODUCTION", aod_production, args.submit, job_ids)

                if ("reco1" in args.options) or ("all" in args.options):
                    write_bash_script(slurm_dict, "RECO_1", reco_task, args.submit, job_ids)

                if ("reco2" in args.options) or ("all" in args.options):
                    if not ("reco1" in args.options):
                        reco_simple = copyfiles + "\n" + reco_simple
                    write_bash_script(slurm_dict, "RECO_2", reco_simple, args.submit, job_ids)

                if ("calib1" in args.options) or ("all" in args.options):
                    write_bash_script(slurm_dict, "CALIB_1", calib_task_2, args.submit, job_ids)

                if ("qa" in args.options) or ("all" in args.options):
                    write_bash_script(slurm_dict, "QA", valid_elements_script + "\n" + secvtx_script + "\n" + qa_task, args.submit, job_ids)

                if (("afterburner" in args.options) or ("all" in args.options)) and ("training_data" not in mode):
                    write_bash_script(slurm_dict, "AFTERBURNER_QA", run_afterburner_qa_macro, args.submit, job_ids)


            elif len(glob.glob(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "**/tpcreco_1.log_done"), recursive=True)) > 0:

                ### Needs the following log-files in the simulation folder:
                ### tpcreco_1.log -> For magnetic field extraction
                ### tpcreco_1.log_done
                ### itstpcMatch_1.log_done
                ### trdreco2_1.log_done
                ### toftpcmatch_1.log_done
                ### pvfinder_1.log_done
                ### svfinder_1.log_done
                ### Optional: tpcclusterpart0_1.log_done,tpcclustermerge_1.log_done

                tasks = {
                    "before_eof": "",
                    "after_eof": task_dict["CLEANUP"]["remove_files"]
                }

                toplevel_output_dir = copy.copy(output_dir)
                output_dir = os.path.join(output_dir, "reco")
                slurm_dict["job_settings"]["chdir"] = output_dir
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(toplevel_output_dir, "QA"), exist_ok=True)
                os.makedirs(os.path.join(toplevel_output_dir, "tmp"), exist_ok=True)

                digit_files_found = glob.glob(os.path.join(SUBMIT["exec_settings"]["simulation_dir"], "**/tpcreco_1.log_done"), recursive=True)
                digit_files_submit = digit_files_found
                if args.limit > 0:
                    digit_files_submit = digit_files_found[:args.limit]
                for dig_idx, digits in enumerate(digit_files_submit):

                    simulation_dir = os.path.dirname(digits)
                    print("\n-> Submitting for directory:", simulation_dir)
                    if ("reco" in args.options) or ("all" in args.options): # First, reco with SC correction for tracking efficiencies, etc.
                        copyfiles = "cp {0}/o2simdigitizerworkflow_configuration.ini {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/tpctriggers.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/ctpdigits.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/collisioncontext.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2sim_Kine.root {1}/qed\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2sim_Kine.root {1}\n".format(simulation_dir, output_dir)
                        ### Never copy the production's o2match_itstpc.root into the output dir: the ITS-TPC
                        ### matches have to be produced from the clusters of the clusterizer under test,
                        ### otherwise every variant is compared against the same reference matches.
                        # copyfiles += "cp {0}/tpc_driftime_digits_lane*.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/tpcdigits.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/sgn_Kine.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/qed_Kine.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2trac_its.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2clus_its.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2reco_ft0.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/mfttracks.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/mchtracks.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/tofclusters.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/toftracks.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/trdtracklets.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/trdcalibratedtracklets.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/globalfwdtracks.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/muontracks.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2reco_fv0.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/o2reco_fdd.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/phoscells.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/cpvclusters.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp {0}/emccells.root {1}\n".format(simulation_dir, output_dir)
                        copyfiles += "cp -r {0}/../ccdb {1}/ccdb_tmp\n".format(simulation_dir, output_dir)
                        clusterer_task = copyfiles + "\nexport ALICEO2_CCDB_LOCALCACHE={0}/ccdb_tmp\n".format(output_dir) + getMCworkflow(os.path.join(simulation_dir, "tpcreco_1.log_done")) ### Using the digits directly, not needing the chunkeddigitmerger
                        clusterer_task = clusterer_task.replace("--tpc-sectors 0-17", "--tpc-sectors 0-35")
                        reco_settings_dictionary = SUBMIT["reco_task"]["input-digits"]
                        configKeyValues = ""
                        for key, val in reco_settings_dictionary.items():
                            if key=="configKeyValues":
                                for k1, v1 in val.items():
                                    for k2, v2 in v1.items():
                                        if "nnClassificationPath" in k2:
                                            if int(v1["applyNNclusterizer"]) == 1:
                                                new_value = ""
                                                num_nets = len(str(v2).split(":"))
                                                for i,net in enumerate(str(v2).split(":")):
                                                    if (dig_idx == 0):
                                                        os.system("cp {0} {1}".format(net, os.path.join(toplevel_output_dir, "networks/net_classification_" + str(i) + ".onnx")))
                                                    new_value += os.path.join(toplevel_output_dir, "networks/net_classification_" + str(i) + ".onnx")
                                                    if i != num_nets-1:
                                                        new_value += ":"
                                                configKeyValues += str(k1) + '.' + str(k2) + '=' + str(new_value) + ';'
                                        elif "nnRegressionPath" in k2:
                                            if (int(v1["applyNNclusterizer"]) == 1) and (int(v1["nnClusterizerUseCfRegression"]) == 0):
                                                new_value = ""
                                                num_nets = len(str(v2).split(":"))
                                                for i,net in enumerate(str(v2).split(":")):
                                                    if (dig_idx == 0):
                                                        os.system("cp {0} {1}".format(net, os.path.join(toplevel_output_dir, "networks/net_regression_" + str(i) + ".onnx")))
                                                    new_value += os.path.join(toplevel_output_dir, "networks/net_regression_" + str(i) + ".onnx")
                                                    if i != num_nets-1:
                                                        new_value += ":"
                                                configKeyValues += str(k1) + '.' + str(k2) + '=' + str(new_value) + ';'
                                        elif ("runQA" in k2):
                                            continue
                                        else:
                                            configKeyValues += str(k1) + '.' + str(k2) + '=' + str(v2) + ';'
                        clusterer_task = clusterer_task.replace("--input-type clusters", "--input-type digits")
                        clusterer_task = clusterer_task.replace("--configKeyValues \"", "--resources-monitoring 2 --configKeyValues \"" + configKeyValues)
                        clusterer_task = clusterer_task.replace("--output-type tracks,", "--tpc-native-cluster-writer \" --outfile tpc-native-clusters.root\" --output-type clusters,tracks,")
                        clusterer_task = clusterer_task.replace("--configKeyValues \"", "--configKeyValues \"GPU_proc.runQA={0};GPU_QA.output=histograms.root;".format(int(reco_settings_dictionary["configKeyValues"]["GPU_proc"]["runQA"])))
                        # clusterer_task = clusterer_task.replace("GPU_proc.ompThreads=8", "GPU_proc.ompThreads=38")
                        clusterer_task = clusterer_task.replace("GPU_proc.ompThreads=8", "GPU_proc.ompThreads=1")

                        # cluster_merger = getMCworkflow(os.path.join(simulation_dir, "tpcclustermerge_1.log_done"))
                        # tasks["before_eof"] = cluster_merger + "\nexit 0"
                        # tasks["after_eof"] = "" #"rm -rf {0}/tpc-native-clusters-part*.root\n".format(output_dir)
                        # write_bash_script(slurm_dict, "CLUSTERS_2", tasks, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="CLUSTERS_2_" + str(dig_idx))

                        combined_reco_task = "\nexport ALICEO2_CCDB_LOCALCACHE={0}/ccdb_tmp\n".format(output_dir)
                        combined_reco_task += 'SVEXTRA=";svertexer.createFullV0s=true;svertexer.createFullCascades=true;"\n'
                        itstpcmatcher_1 = getMCworkflow(os.path.join(simulation_dir, "itstpcMatch_1.log_done"))
                        if ("itstpcmcstudy" in args.options):
                            itstpcmatcher_1 = itstpcmatcher_1.replace("--run", "--run --debug-tree-flags 127")
                        trdreco2_1 = getMCworkflow(os.path.join(simulation_dir, "trdreco2_1.log_done"))
                        toftpcmatch_1 = getMCworkflow(os.path.join(simulation_dir, "toftpcmatch_1.log_done"))
                        pvfinder_1 = getMCworkflow(os.path.join(simulation_dir, "pvfinder_1.log_done"))
                        svfinder_task = getMCworkflow(os.path.join(simulation_dir, "svfinder_1.log_done"))
                        svfinder_task = svfinder_task.replace("--configKeyValues \"", "--configKeyValues \"$SVEXTRA;")

                        ### These stages are ONE DPL workflow: piped together, each driver serialises its
                        ### topology to the next one and the last one runs the merged graph, so the data stays
                        ### in shared memory instead of going through ROOT files. See stripRun() and
                        ### disableRootInput() for the two rewrites the scraped production commands need.
                        ###
                        ### Verified with "--dump-workflow" on the merged pipe: 32 devices, no duplicate
                        ### outputs, and the QC reads GLO/TPCITS from itstpc-track-matcher and TPC/TRACKS from
                        ### tpc-tracker, i.e. from the clusterizer under test and not from the production.
                        hbfutils_cfg = getHBFUtilsConfig(pvfinder_1)

                        ### Supplies everything the chain reads from disk, exactly once: ITS tracks and
                        ### clusters, TOF clusters, FT0/FV0 recpoints, TRD tracklets, CTP digits and EMCAL
                        ### cells, i.e. the detectors of the vertexing-sources list that are not reconstructed
                        ### here. Only ITS may go into --track-types: any match source there would re-inject
                        ### the very readers that disableRootInput() exists to remove.
                        globalreader = r'${O2_ROOT}' + "/bin/o2-global-track-cluster-reader" \
                            + " --track-types ITS --cluster-types ITS,TOF,FT0,FV0,TRD,CTP,EMC" \
                            + " --configKeyValues \"{0}\"".format(hbfutils_cfg)
                        itstpcmatchingqc = r'${O2_ROOT}' + "/bin/o2-itstpc-matching-qc --shm-segment-size 50000000000"

                        ### clusterer_task carries the "cp" lines and the CCDB export ahead of its command;
                        ### those are plain statements and must stay outside the pipeline.
                        clusterer_prefix, _, clusterer_cmd = clusterer_task.rstrip().rpartition("\n")
                        combined_reco_task += clusterer_prefix + "\n"

                        reco_stages = [globalreader, stripRun(clusterer_cmd)]
                        reco_stages += [disableRootInput(stripRun(stage)) for stage in
                                        [itstpcmatcher_1, trdreco2_1, toftpcmatch_1, pvfinder_1, svfinder_task]]
                        reco_stages = [stage + " -b" for stage in reco_stages]

                        ### The head of the pipe still has the job-script heredoc on its stdin, and a DPL
                        ### driver whose stdin is a pipe tries to read a serialised workflow from it
                        ### (isInputConfig(), same file). That is why the first stage used to echo the tail of
                        ### the job script back as [ERROR] lines; /dev/null keeps it from reading it at all.
                        reco_stages[0] += " < /dev/null"
                        combined_reco_task += " | \\\n".join(reco_stages + [itstpcmatchingqc + " -b --run"]) + "\n"
                        adjusted_slurm_dict = copy.copy(slurm_dict)
                        reco_slurm_dict = slurm_dict.get("RECO", {
                            "cpus-per-task": 36,
                            "mem": "128G",
                            "time": "20"
                        })
                        adjusted_slurm_dict["job_settings"].update(reco_slurm_dict)
                        write_bash_script(adjusted_slurm_dict, "RECO", combined_reco_task, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="RECO_" + str(dig_idx), trapped_requeue=True)

                    if (args.submit_qa_only_once==0) or (args.submit_qa_only_once==1 and dig_idx == 0):
                        if ("qa" in args.options) or ("all" in args.options):
                            grep_mag_field = subprocess.check_output("grep -m1 Solenoid {}".format(os.path.join(simulation_dir, "tpcreco_1.log")), shell=True)
                            grep_mag_field = float(grep_mag_field.decode().strip('\n').split(" kG")[0].split(" ")[-1].strip().split("(")[-1].split("*)")[0])*5.

                            qa_task = "\nexport ALICEO2_CCDB_LOCALCACHE={0}/ccdb_tmp\n".format(output_dir) + "o2-tpc-qa-clusters -b --shm-segment-id {0} --session {0}".format(args.id)
                            for key, val in SUBMIT["qa_task"].items():
                                # qa_task += " --" + str(key) + ' "' + str(val) + '"'
                                if "magnetic-field" in key:
                                    qa_task += " --" + str(key) + ' "' + str(grep_mag_field) + '"'
                                else:
                                    qa_task += " --" + str(key) + ' "' + str(val) + '"'
                            qa_task += " --simulation-path " + simulation_dir
                            qa_task += " --output-path " + output_dir
                            qa_task += " --real-data " + str(SUBMIT["data_settings"]["real-data"])
                            # qa_task += " --read-drifttime-digits 1"
                            secvtx_script = "root -l -b -q '" + CONF["submission"]["secvtx_script"] + "(\"" + output_dir + "\", \"" + output_dir + "\")'"
                            tasks["before_eof"] = secvtx_script + "\n" + qa_task
                            tasks["after_eof"] = "rm -rf {0}/ccdb_tmp\n".format(output_dir)

                            adjusted_slurm_dict = copy.copy(slurm_dict)
                            adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 18
                            adjusted_slurm_dict["job_settings"]["mem"] = "512G"
                            adjusted_slurm_dict["job_settings"]["time"] = "120"
                            write_bash_script(adjusted_slurm_dict, "QA", tasks, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="QA_" + str(dig_idx), trapped_requeue=True)

                        if (("afterburner" in args.options) or ("all" in args.options)) and ("training_data" not in mode):
                            if "network" in mode:
                                run_afterburner_qa_macro = "root -l -b -q '/lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/qaAfterburner.C(\"network\", \"{}\")'".format(output_dir)
                            else:
                                run_afterburner_qa_macro = "root -l -b -q '/lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/src/qaAfterburner.C(\"native\", \"{}\")'".format(output_dir)
                            adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 1
                            adjusted_slurm_dict["job_settings"]["mem"] = "64G"
                            adjusted_slurm_dict["job_settings"]["time"] = "20"
                            write_bash_script(adjusted_slurm_dict, "AFTERBURNER_QA", run_afterburner_qa_macro, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="AFTERBURNER_QA_" + str(dig_idx), trapped_requeue=True)

                    if ("itstpcmcstudy" in args.options):
                        itstpcmatcher_1 = getMCworkflow(os.path.join(simulation_dir, "itstpcMatch_1.log_done"))
                        lI = ";TPCCorrMap.lumiInst=" + itstpcmatcher_1.split(";TPCCorrMap.lumiInst=")[1].split(";")[0]
                        lIF = ";TPCCorrMap.lumiInstFactor=" + itstpcmatcher_1.split(";TPCCorrMap.lumiInstFactor=")[1].split(";")[0]
                        cna = "--condition-not-after " + itstpcmatcher_1.split("--condition-not-after ")[1].split(" ")[0]
                        matchstudy_task = "\nexport ALICEO2_CCDB_LOCALCACHE={0}/ccdb_tmp\n".format(output_dir)
                        matchstudy_task += "o2-trackmc-study-workflow --shm-segment-size 8000000000 {0} --configKeyValues \"tpcitsMatch.XMatchingRef=60;trmcconf.maxTPCRefExtrap=2;trmcconf.rejectClustersResStat=0.1;{1}\"".format(cna, lI + lIF)
                        adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 8
                        adjusted_slurm_dict["job_settings"]["mem"] = "32G"
                        adjusted_slurm_dict["job_settings"]["time"] = "20"
                        write_bash_script(adjusted_slurm_dict, "ITSTPCMCSTUDY_QA", matchstudy_task, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="ITSTPCMCSTUDY_QA_" + str(dig_idx), trapped_requeue=True)

                    if (("combine" in args.options) or ("all" in args.options)):
                        ### For the first simulation the full information is saved, afterwards only the histograms
                        combine_qa = """
if [ -f %(tod)s/QA/histograms_combined.root ]; then
    cp %(od)s/dbg_TPCITSmatch.root %(tod)s/tmp/dbg_TPCITSmatch_%(id)s.root
    cp %(od)s/trackMCStudy.root %(tod)s/tmp/trackMCStudy_%(id)s.root
    cp %(od)s/histograms.root %(tod)s/tmp/histograms_%(id)s.root
    cp %(od)s/outITSTPCmatchingQC.root %(tod)s/tmp/outITSTPCmatchingQC_%(id)s.root
    cp %(od)s/job_*.out %(tod)s/tmp/.
    cp %(tod)s/QA/histograms_combined.root %(od)s/histograms_combined.root
    cp %(tod)s/QA/outITSTPCmatchingQC_combined.root %(od)s/outITSTPCmatchingQC_combined.root
    cp %(tod)s/QA/custom_secondary_vertex_qa_combined.root %(od)s/custom_secondary_vertex_qa_combined.root
    root -l -b -q '%(cs)s(\"%(od)s/histograms_combined.root;%(od)s/histograms.root\", \"%(tod)s/QA/histograms_combined.root\")'
    root -l -b -q '%(cs)s(\"%(od)s/outITSTPCmatchingQC_combined.root;%(od)s/outITSTPCmatchingQC.root\", \"%(tod)s/QA/outITSTPCmatchingQC_combined.root\")'
    hadd -f %(tod)s/QA/custom_secondary_vertex_qa_combined.root %(od)s/custom_secondary_vertex_qa_combined.root %(od)s/custom_secondary_vertex_qa.root
else
    cp %(od)s/dbg_TPCITSmatch.root %(tod)s/tmp/dbg_TPCITSmatch_%(id)s.root
    cp %(od)s/trackMCStudy.root %(tod)s/tmp/trackMCStudy_%(id)s.root
    cp %(od)s/histograms.root %(tod)s/tmp/histograms_%(id)s.root
    cp %(od)s/outITSTPCmatchingQC.root %(tod)s/tmp/outITSTPCmatchingQC_%(id)s.root
    cp %(od)s/job_*.out %(tod)s/tmp/.
    cp %(od)s/network_ideal.root %(tod)s/tmp/.
    cp %(od)s/performanceMetrics.json %(tod)s/tmp/performanceMetrics.json
    mv %(od)s/dump_cluster_error.csv %(tod)s/tmp/.
    mv %(od)s/dump_trk_index.csv %(tod)s/tmp/.
    mv %(od)s/histograms.root %(tod)s/QA/histograms_combined.root
    mv %(od)s/outITSTPCmatchingQC.root %(tod)s/QA/outITSTPCmatchingQC_combined.root
    mv %(od)s/custom_secondary_vertex_qa.root %(tod)s/QA/custom_secondary_vertex_qa_combined.root
    mv %(od)s/afterburner_qa.root %(tod)s/QA/afterburner_qa.root
    mv %(od)s/tpc_tracks_tabular_information.root %(tod)s/QA/tpc_tracks_tabular_information.root
    mv %(od)s/custom_clusters_track_clusters.root %(tod)s/QA/custom_clusters_track_clusters.root
    mv %(od)s/custom_clusters_track_paths.root %(tod)s/QA/custom_clusters_track_paths.root
    mv %(od)s/native_ideal.root %(tod)s/QA/native_ideal.root
fi

rm -rf %(od)s/*
rm -rf %(tod)s/job_RECO_*.err
rm -rf %(tod)s/job_QA_*.err
rm -rf %(tod)s/job_AFTERBURNER_QA_*
rm -rf %(tod)s/job_COMBINE_QA_*
rm -rf /dev/shm/fmq*
                        """ % {"od": output_dir, "tod": toplevel_output_dir, "cs": args.combine_script, "id": dig_idx}
                        adjusted_slurm_dict["job_settings"]["time"] = 10
                        adjusted_slurm_dict["job_settings"]["cpus-per-task"] = 1
                        adjusted_slurm_dict["job_settings"]["mem"] = "32G"
                        write_bash_script(adjusted_slurm_dict, "COMBINE_QA", combine_qa, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="COMBINE_QA_" + str(dig_idx), trapped_requeue=True)

                if (("combine" in args.options) or ("all" in args.options)):
                    tasks["before_eof"] = "root -l -b -q '" + CONF["submission"]["valid_elements_script"] + "(\"" + os.path.join(toplevel_output_dir, "QA/outITSTPCmatchingQC_combined.root") + "\", \"" + os.path.join(toplevel_output_dir, "QA/custom_its_tpc_matching_qc.root") + "\")'"
                    tasks["after_eof"] = "rm -rf %(od)s/job_ITSTPCQC*" % {"od": output_dir, "tod": toplevel_output_dir}
                    write_bash_script(adjusted_slurm_dict, "ITSTPCQC", tasks, args.submit, job_ids, odir=toplevel_output_dir, overwrite_log_name="ITSTPCQC", trapped_requeue=True)