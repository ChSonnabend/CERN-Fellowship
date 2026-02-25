import sys
import os
import datetime
import re

import numpy as np
import pandas as pd
import json
import onnxruntime as ort
import torch
import torch.nn as nn
import argparse

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import configurations
from sys import getsizeof

# sys.path.append("/lustre/alice/users/csonnab/TPC/NeuralNetworks/Neural-Networks/NeuralNetworkClasses/FullNetworks")
# from ResNetGrid_2 import *

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-locdir", "--local-training-dir", default=".", help="Local directory for training of the neural network")
args = parser.parse_args()

########### Load the configurations from config.json ###########

configs_file = open("config.json", "r")
CONF = json.load(configs_file)
configs_file.close()

for imp in CONF["directory_settings"]["classes"]:
    sys.path.append(imp)
    
### directory settings
O2_TABLES           = CONF["directory_settings"]["O2_TABLES"]

### execution settings
training_dir        = CONF["exec_settings"]["training_dir"]
output_folder       = CONF["exec_settings"]["output_folder"]
enable_qa           = CONF["exec_settings"]["enable_qa"]

### network settings
save_as_pt          = CONF["network_settings"]["save_as_pt"]
save_as_onnx        = CONF["network_settings"]["save_as_onnx"]
save_loss_in_files  = CONF["network_settings"]["save_loss_in_files"]

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
# from custom_loss_functions import custom_mse_loss, weighted_mse_loss

###
# data_size: Determines the size of the input that the network sees (this is also the application in O2)
# input_size: Determines the size of the input that the network actually computes
# E.g.: Data = [9,11,11] -> data_size = [7,11,11] -> input_size = [-1,3,9,9]
###

dict_net = configurations.dict_net
dict_data = configurations.dict_data

dict_net["OTHER"]["verbose"] = True if verbose else False
dict_data["OTHER"]["verbose"] = True if verbose else False

data_full = pd.read_csv(configurations.data_path)
dataIn = data_full[configurations.labels_x]
dataOut = data_full[configurations.labels_y]
del data_full

dataIn, dataOut = configurations.data_transformer(dataIn, dataOut)

NeuralNet = NN(configurations.network())

### Data preparation
X_train, X_test, y_train, y_test = train_test_split(dataIn[:int(configurations.num_datapoints)], dataOut[:int(configurations.num_datapoints)], test_size=configurations.TEST_SIZE, shuffle=configurations.SHUFFLE)
del dataIn, dataOut

data = DataLoading([X_train, y_train], [X_test, y_test], settings=dict_data)
save_onnx_example = X_test[:100]
del X_train, X_test, y_train, y_test

### Evaluate training and validation loss over epochs
NeuralNet.training(data, settings=dict_net)

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