import os, sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import json
from onnxruntime.quantization import QuantFormat, QuantType


##### Loading configurations from config.json file #####

configs_file = open(os.path.join(";;", "config.json"), "r")
CONF = json.load(configs_file)

### directory settings
NN_dir = CONF["directory_settings"]["NN_dir"]
training_dir = CONF["exec_settings"]["training_dir"]

configs_file.close()


##### Loading the neural network class #####

sys.path.append(NN_dir)

from GeneralPurposeClass.extract_from_root import load_tree
from NeuralNetworkClass.NeuralNetworkClasses.NN_class import General_NN, NN
from NeuralNetworkClass.NeuralNetworkClasses.utils.custom_loss_functions import *

data_path = ";;"

labels_x = ["clusterState", "xx", "yy", "zz", "cluster.getSigmaPad()", "cluster.getSigmaTime()", "mP[0]", "mP[1]", "mP[2]", "mP[3]", "mP[4]", "mC[0]", "mC[2]", "mC[5]", "mC[9]", "mC[14]"]
labels_y = ["yy", "zz", "mP[0]", "mP[1]"]

num_datapoints = 1e7
mode = "cluster-errors"

def data_transformer(data_X, data_Y, step=0):
    X = data_X.copy()
    Y = data_Y.copy()

    # --- covariance validity ---
    mC0 = X["mC[0]"].to_numpy()
    mC2 = X["mC[2]"].to_numpy()
    ok_cov = (mC0 < 2) & (mC2 < 2) & np.isfinite(mC0) & np.isfinite(mC2)

    # --- residual pre-cut: track (Y) - cluster (X) ---
    dy0 = (Y["mP[0]"].to_numpy() - Y["yy"].to_numpy())
    dz0 = (Y["mP[1]"].to_numpy() - Y["zz"].to_numpy())
    ok_res = (np.abs(dy0) < 5) & (np.abs(dz0) < 5) & np.isfinite(dy0) & np.isfinite(dz0)

    mask = ok_cov & ok_res

    # --- slice and align ---
    X = X.loc[mask].reset_index(drop=True)
    Y = Y.loc[mask].reset_index(drop=True)

    # --- expand clusterState bitmask (0..15) into 4 binary features ---
    if "clusterState" in X.columns:
        s = X["clusterState"].to_numpy(dtype=np.uint8)
        X["cs0"] = ((s >> 0) & 1).astype(np.float32)
        X["cs1"] = ((s >> 1) & 1).astype(np.float32)
        X["cs2"] = ((s >> 2) & 1).astype(np.float32)
        X["cs3"] = ((s >> 3) & 1).astype(np.float32)
        X = X.drop(columns=["clusterState"])

    # --- build targets ---
    dy = (Y["mP[0]"].to_numpy(dtype=np.float32) - Y["yy"].to_numpy(dtype=np.float32))
    dz = (Y["mP[1]"].to_numpy(dtype=np.float32) - Y["zz"].to_numpy(dtype=np.float32))
    CY = X["mC[0]"].to_numpy(dtype=np.float32)
    CZ = X["mC[2]"].to_numpy(dtype=np.float32)
    
    if step == 0:
        target = np.stack([dy**2 - CY, dz**2 - CZ], axis=1).astype(np.float32)
    if step == 1:
        target = np.stack([dy, dz, CY, CZ], axis=1).astype(np.float32)

    # --- build inputs ---
    X_in = X.to_numpy(dtype=np.float32)

    # --- final safety: remove any non-finite values (debug-friendly) ---
    X_in = np.nan_to_num(X_in, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    return X_in, target

# data_X = data_X.to_numpy(dtype=np.float32)
# data_Y = data_Y.to_numpy(dtype=np.float32)

class network(nn.Module):
    def __init__(self, engineered=True):
        super().__init__()
        self.output_nodes = 2
        self.engineered = engineered
        self.input_dim = len(labels_x) + 3

        # how many extra features feature_expansion() appends
        self.extra_dim = self._extra_feature_dim() if engineered else 0
        dim0 = self.input_dim + self.extra_dim
        print("Input dim:", self.input_dim, "Extra engineered features:", self.extra_dim, "Total input to net:", dim0)

        self.seq_net = nn.Sequential(
            nn.Linear(dim0, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_nodes),
            nn.Softplus(beta=1.0)
        )

        # If you want to guarantee positivity of RY/RZ, do it here:
        # self.out_act = nn.Softplus(beta=1.0)
        # else:
        self.out_act = None

    def _extra_feature_dim(self):
        return -1

    def feature_expansion(self, X):
        """
        X: (N, D) float32 tensor
        returns: (N, extra_dim) engineered features
        NOTE: choose indices based on your column order in labels_x AFTER transformer (clusterState dropped, cs0..cs3 appended).
        """
        # ---- IMPORTANT ----
        # Fill these indices to match your actual X column ordering.
        # Example guesses (YOU MUST ADJUST):
        #   xx, yy, zz, cluster.getSigmaPad(), cluster.getSigmaTime(), mP[2], etc.
        # -------------------
        idx_yy = np.where(np.array(labels_x) == "yy")[0][0]
        idx_zz = np.where(np.array(labels_x) == "zz")[0][0]
        idx_ty = np.where(np.array(labels_x) == "mP[0]")[0][0]
        idx_tz = np.where(np.array(labels_x) == "mP[1]")[0][0]
        idx_sinPhi  = np.where(np.array(labels_x) == "mP[2]")[0][0]
        idx_dzds = np.where(np.array(labels_x) == "mP[3]")[0][0]
        
        remove_idx = [idx_yy, idx_zz, idx_ty, idx_tz]

        # Pairwise products (some intuitive cross terms)
        # (adjust to taste)
        # prod1 = torch.clamp((X[:, idx_yy] - X[:, idx_ty]).unsqueeze(1)**2, max=25)
        # prod2 = torch.clamp((X[:, idx_zz] - X[:, idx_tz]).unsqueeze(1)**2, max=25)
        prod3 = torch.clamp((X[:, idx_sinPhi] * X[:, idx_sinPhi]).unsqueeze(1), min=0.0, max=0.95*0.95)  # clamp to avoid extreme values in prod4
        prod4 = 1. / (1. - prod3)
        prod5 = torch.clamp((X[:, idx_dzds] * X[:, idx_dzds]).unsqueeze(1)*prod4, max=100)

        engineered = torch.cat(
            # [prod1, prod2, prod3, prod4, prod5],
            [prod3, prod4, prod5],
            dim=1
        )
        return engineered, remove_idx

    def __weight_init__(self, weight_init=None):
        if weight_init is not None:
            for layer in self.seq_net:
                if hasattr(layer, 'weight'):
                    weight_init(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.1)

    def forward(self, X):
        if self.engineered:
            E, remove_idx = self.feature_expansion(X)
            X = torch.cat([X[:, [i for i in range(X.shape[1]) if i not in remove_idx]], E], dim=1)

        out = self.seq_net(X)
        if self.out_act is not None:
            out = self.out_act(out)  # positivity for RY/RZ
        return out
    


def NLLloss(pred, target, eps=1e-6):
    # pred: (N,2) raw network outputs
    # target: (N,4) [dy, dz, CY, CZ]

    dy = target[:, 0]
    dz = target[:, 1]
    CY = target[:, 2]
    CZ = target[:, 3]

    # Enforce positive measurement variances
    RY = pred[:, 0]
    RZ = pred[:, 1]

    Sy = CY + RY
    Sz = CZ + RZ

    # Safety clamp (optional but helps early training)
    Sy = torch.clamp(Sy, min=eps)
    Sz = torch.clamp(Sz, min=eps)

    lossY = torch.log(Sy) + (dy * dy) / Sy
    lossZ = torch.log(Sz) + (dz * dz) / Sz
    
    if not torch.isfinite((lossY + lossZ).mean()):
        print("pred min/max", pred.min().item(), pred.max().item())
        print("CY min/max", CY.min().item(), CY.max().item())
        print("CZ min/max", CZ.min().item(), CZ.max().item())
        print("Sy min/max", Sy.min().item(), Sy.max().item())
        print("Sz min/max", Sz.min().item(), Sz.max().item())
        raise RuntimeError("non-finite loss")

    return (lossY + lossZ).mean()

TEST_SIZE = 0.2
SHUFFLE = True

training_configs = [
    {
        "GLOBAL" : {
            "loss_function": nn.MSELoss(),
            "epochs": 50
        }
    },
    {
        "GLOBAL" : {
            "loss_function": NLLloss,
            "epochs": 150
        },
        "SCHEDULER_OPTIONS": {
            "patience": 20,
            "factor": 0.5
        }
    }
]

dict_net = {
    "GLOBAL" : {
        "epochs" : 1000,
        "loss_function" : nn.MSELoss(),
        "optimizer" : optim.Adam,
        "scheduler" : optim.lr_scheduler.ReduceLROnPlateau,
        "weight_init" : torch.nn.init.xavier_uniform_,
        "amp": True,
        "quantization": False
    },
    "DATA_OPTIONS" : {
        "batchsize_schedule" : [0, 20, 50, 80, 120, 180],
        "batchsize_training" : [131072, 65536, 32768, 16384, 4096, 2048],
        "batchsize_validation" : 131072,
        "shuffle_every_epoch" : True,
        "num_workers" : 0,
        "pin_memory" : None,
        "copy_to_device" : True
    },
    "AMP_OPTIONS" : {
        "dtype": torch.float32,
    },
    "QUANTIZATION_OPTIONS" : {
        "ONNX": {
            "weight_type": QuantType.QInt8,
            "per_channel": False,
        },
        "PYTORCH": {
            "dtype": torch.qint8,
        },
    },
    "LOSS_OPTIONS" : {
    },
    "OPTIMIZER_OPTIONS" : {
        "lr" : 0.001,
        "weight_decay" : 0
    },
    "SCHEDULER_OPTIONS" : {
        "patience" : 10,
        "factor" : 0.3
    },
    "MACHINE_OPTIONS" : {
        "device" : None,
        "cpu_threads" : None,
        "dtype" : torch.float32
    },
    "ONNX": {
        "export_params": True,
        "opset_version": 15,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    },
    "OTHER" : {
        "verbose" : True,
        "n_samples" : np.inf,
        "save_at_epochs": None,
    }
}

dict_data = {
    "GLOBAL" : {
        "transform_data" : False,
        "copy_to_device" : False,
    },
    "SCALERS" : {
        "X_data_scalers" : [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "Y_data_scalers" : [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
    },
    "MACHINE_OPTIONS" : {
        "device" : None,
        "dtype_X" : torch.float32,
        "dtype_Y" : torch.float32,
        "amp": False,
    },
    "OTHER" : {
        "num_workers" : 0
    }
}