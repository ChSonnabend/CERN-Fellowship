import sys, os, datetime, re, json, argparse

import pandas as pd
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import configurations

# sys.path.append("/lustre/alice/users/csonnab/TPC/NeuralNetworks/Neural-Networks/NeuralNetworkClasses/FullNetworks")
# from ResNetGrid_2 import *

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-locdir", "--local-training-dir", default=".", help="Local directory for training of the neural network")
args = parser.parse_args()

########### Load the configurations from config.json ###########

configs_file = open("config.json", "r")
CONF = json.load(configs_file)

### directory settings
NN_dir              = CONF["directory_settings"]["NN_dir"]
O2_TABLES           = CONF["directory_settings"]["O2_TABLES"]

### execution settings
training_dir        = CONF["exec_settings"]["training_dir"]
output_folder       = CONF["exec_settings"]["output_folder"]
enable_qa           = CONF["exec_settings"]["enable_qa"]

### network settings
save_as_pt          = CONF["network_settings"]["save_as_pt"]
save_as_onnx        = CONF["network_settings"]["save_as_onnx"]
save_loss_in_files  = CONF["network_settings"]["save_loss_in_files"]

configs_file.close()

########### Print the date, time and location for identification ###########

date = datetime.datetime.now().date()
time = datetime.datetime.now().time()

verbose = (int(os.environ.get("SLURM_PROCID", "0"))==0)

if verbose:
    print("Info:\n")
    print("SLURM job ID:", os.environ.get("SLURM_JOBID", "N/A"))
    print("Date (dd/mm/yyyy):",date.strftime('%02d/%02m/%04Y'))
    print("Time (hh/mm/ss):", time.strftime('%02H:%02M:%02S'))
    print("Output-folder:", training_dir+"/"+output_folder+"/"+args.local_training_dir)


########### Import the Neural Network class ###########

sys.path.append(NN_dir)

from GeneralPurposeClass.measure_memory_usage import print_memory_usage
from NeuralNetworkClass.NeuralNetworkClasses.utils.dataset_loading import DataLoading
from GeneralPurposeClass.extract_from_root import load_tree
from NeuralNetworkClass.NeuralNetworkClasses.NN_class import General_NN, NN
from NeuralNetworkClass.NeuralNetworkClasses.utils.functions import deep_update, print_dict

### --- Data import --- ###

if ("cluster-errors" in configurations.mode):
    data_full = pd.read_csv(configurations.data_path)
    dataIn = data_full[configurations.labels_x]
    dataOut = data_full[configurations.labels_y]
    del data_full
    
else:
    def readFromPy(pathLabels, pathData):
        return np.loadtxt(pathLabels, dtype=str), np.load(pathData)

    labelsIn, dataIn = readFromPy(configurations.data_path_X+"/input_data.txt", configurations.data_path_X+"/input_data.npy")
    labelsOut, dataOut = readFromPy(configurations.data_path_Y+"/output_data.txt", configurations.data_path_Y+"/output_data.npy")

    ### Change input size
    def extract_ints(string):
        return np.array(list(map(int, re.findall(r'\d+', string))))

    last_idx = labelsIn[np.arange(len(labelsIn))[["in_" in lbl for lbl in labelsIn]][-1]]
    max_input_size = extract_ints(last_idx) + 1
    center = (extract_ints(last_idx) / 2).astype(int)
    accepted_idcs = []

    if configurations.data_size:
        net_input_size = np.array(configurations.data_size)
    else:
        net_input_size = max_input_size + 1

    ### Simple preselection
    if np.any(np.array(net_input_size)-max_input_size > 0):
        print("Cannot satisfy input size requirements. Exiting...")
        exit()
    else:
        if verbose:
            print("Changing input size from ", max_input_size, "to", net_input_size)
        min_index = center - ((np.array(net_input_size) - 1) / 2)
        max_index = center + ((np.array(net_input_size) - 1) / 2)

        accepted_labels = []
        for ri in range(net_input_size[0]):
            for pi in range(net_input_size[1]):
                for ti in range(net_input_size[2]):
                    accepted_labels.append("in_row_"+str(int(min_index[0]+ri))+"_pad_"+str(int(min_index[1]+pi))+"_time_"+str(int(min_index[2]+ti)))
        final_in_labels = np.ones(len(labelsIn)).astype(bool)
        for i, lbl_1 in enumerate(labelsIn):
            if "in_" in lbl_1:
                final_in_labels[i] = False
                for j, lbl_2 in enumerate(accepted_labels):
                    if lbl_1 == lbl_2:
                        final_in_labels[i] = True
        accepted_idcs = final_in_labels.tolist()

        labelsIn = labelsIn[final_in_labels]
        dataIn = dataIn[:,final_in_labels]

    if ("regression3D" in configurations.mode):
        if not configurations.use_momentum:
            mask = list()
            for lbl in labelsOut:
                if ("pYpX" in lbl) or ("pZpX" in lbl):
                    mask.append(False)
                else:
                    mask.append(True)
            dataOut = dataOut[:,mask]
            labelsOut = labelsOut[mask]

        if configurations.add_flag_data:
            flagLabels, flagData = readFromPy(configurations.data_path_Y+"/output_data_flags.txt", configurations.data_path_Y+"/output_data_flags.npy")
            dataOut = np.hstack((dataOut, flagData))
            labelsOut = np.append(labelsOut, flagLabels)

    if ("flags" in configurations.mode):
        labelsOut, dataOut = readFromPy(configurations.data_path_Y+"/output_data_flags.txt", configurations.data_path_Y+"/output_data_flags.npy")

    if ("charge-overlap" in configurations.mode):
        labelsOut, dataOut = readFromPy(configurations.data_path_Y+"/output_data_charge_overlap.txt", configurations.data_path_Y+"/output_data_charge_overlap.npy")

dict_net = configurations.dict_net
dict_data = configurations.dict_data

dict_net["OTHER"]["verbose"] = True if verbose else False
dict_data["OTHER"]["verbose"] = True if verbose else False

NeuralNet = NN(configurations.network())

for step, trconf in enumerate(configurations.training_configs):

    transformer_X = getattr(configurations, "data_transformer_X", None)
    if callable(transformer_X):
        dataIn_process = transformer_X(dataIn)
    transformer_Y = getattr(configurations, "data_transformer_Y", None)
    if callable(transformer_Y):
        dataOut_process = transformer_Y(dataOut)
    transformer = getattr(configurations, "data_transformer", None)
    if callable(transformer):
        dataIn_process, dataOut_process = transformer(dataIn, dataOut, step=step)

    ### Data preparation
    X_train, X_test, y_train, y_test = train_test_split(dataIn_process[:int(configurations.num_datapoints)], dataOut_process[:int(configurations.num_datapoints)], test_size=configurations.TEST_SIZE, shuffle=configurations.SHUFFLE)
    del dataIn_process, dataOut_process

    data = DataLoading([X_train, y_train], [X_test, y_test], settings=dict_data)
    save_onnx_example = X_test[:1024]
    del X_train, X_test, y_train, y_test

    ### Evaluate training and validation loss over epochs
    deep_update(dict_net, trconf)
    NeuralNet.training(data, settings=dict_net)
    
    print("Finished training with loss function:", trconf["GLOBAL"]["loss_function"], "(step", step, ")")

### Save the network and the losses
NeuralNet.eval()
if save_as_pt == "True":
    NeuralNet.save_net(path=args.local_training_dir+'/net.pt',avoid_q=True)
    # NeuralNet.save_jit_script(path=args.local_training_dir+'/net_jit.pt')
if save_as_onnx == "True":
    NeuralNet.save_onnx(example_data=torch.tensor((save_onnx_example),requires_grad=True).float(),
                        path=args.local_training_dir+'/net.onnx')
    NeuralNet.check_onnx(path=args.local_training_dir+'/net.onnx')
if save_loss_in_files == "True":
    NeuralNet.save_losses(path=[args.local_training_dir+'/training_loss.txt',args.local_training_dir+'/validation_loss.txt'])