import os, sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import json
from onnxruntime.quantization import QuantFormat, QuantType


##### Loading configurations from config.json file #####

configs_file = open(os.path.join(";;", "config.json"), "r")
CONF = json.load(configs_file)
configs_file.close()

for imp in CONF["directory_settings"]["classes"]:
    sys.path.append(imp)

### directory settings
training_dir = CONF["exec_settings"]["training_dir"]

##### Loading the neural network class #####

from GeneralPurposeClass.extract_from_root import load_tree
from NeuralNetworkClass.NeuralNetworkClasses.NN_class import General_NN, NN
from NeuralNetworkClass.NeuralNetworkClasses.utils.custom_loss_functions import *

data_path = ";;"


# -------------------------------------------------------------------------
# NOTE:
#  - We add "internal_trkid" to labels_x so it is loaded from ROOT/tree.
#  - In data_transformer(), we:
#       (a) build an attachment label per (internal_trkid, xx) group (or cluster.num if you have it)
#       (b) drop "internal_trkid" from X and also remove it from labels_x
#  - We then train with NLL on attached clusters only.
#
# IMPORTANT:
#  Grouping key:
#   - Best: (internal_trkid, cluster.num)  # one track, one row
#   - If you do not have cluster.num, you can use xx as a proxy row index IF it is discrete padrow-like.
#     Below we default to "xx" because your feature list does not show cluster.num.
#     If you DO have "cluster.num", add it to labels_x and set GROUP_ROW_KEY="cluster.num".
# -------------------------------------------------------------------------

GROUP_TRACK_KEY = "internal_trkid"
GROUP_ROW_KEY   = "xx"          # change to "cluster.num" if available
MAX_DIST2       = 25.0          # (units^2) gate: if best candidate is farther, mark none attached

USE_SOFT_ASSIGNMENT = True      # False -> hard {0,1}, True -> soft weights
SOFT_TEMPERATURE = 5.0          # larger = softer, smaller = closer to hard argmin
SOFT_USE_NORMED = False         # True uses dy^2/CY + dz^2/CZ instead of raw dist2

# fixed, matches C++ exactly
labels_x = [
    "clusterState", "internal_trkid", "xx", "yy", "zz",
    "cluster.getSigmaPad()", "cluster.getSigmaTime()",
    "mP[0]", "mP[1]", "mP[2]", "mP[3]", "mP[4]",
    "mC[0]", "mC[2]", "mC[5]", "mC[9]", "mC[14]"
]
labels_x_NN = [
    "xx", "yy", "zz",
    "cluster.getSigmaPad()", "cluster.getSigmaTime()",
    "mP[0]", "mP[1]", "mP[2]", "mP[3]", "mP[4]",
    "mC[0]", "mC[2]", "mC[5]", "mC[9]", "mC[14]",
    "cs0", "cs1", "cs2", "cs3"
]
labels_y = ["yy", "zz", "mP[0]", "mP[1]"]

num_datapoints = 1e7
mode = "cluster-errors"


def _make_attachment_weights(X: pd.DataFrame) -> np.ndarray:
    """
    Returns float32 numpy array aligned with X rows.
    Hard: one-hot per group
    Soft: softmax per group (sum=1), optional gate by MAX_DIST2 based on min dist2
    """
    trk = X[GROUP_TRACK_KEY].to_numpy()
    row = X[GROUP_ROW_KEY].to_numpy()

    yy  = X["yy"].to_numpy(np.float32)
    zz  = X["zz"].to_numpy(np.float32)
    mp0 = X["mP[0]"].to_numpy(np.float32)
    mp1 = X["mP[1]"].to_numpy(np.float32)

    dy = mp0 - yy
    dz = mp1 - zz
    dist2 = dy*dy + dz*dz  # always available for gating

    if USE_SOFT_ASSIGNMENT and SOFT_USE_NORMED:
        CY = X["mC[0]"].to_numpy(np.float32)
        CZ = X["mC[2]"].to_numpy(np.float32)
        eps = 1e-6
        score = (dy*dy)/(CY + eps) + (dz*dz)/(CZ + eps)
    else:
        score = dist2

    # Build group ids (int codes) for (trk,row) in a vectorized way
    # factorize on tuples is fast enough; if trk is large int, you can also pack into structured array
    keys = pd.MultiIndex.from_arrays([trk, row])
    gid, ng = pd.factorize(keys, sort=False)  # gid: (N,) int

    N = len(gid)
    w = np.zeros(N, dtype=np.float32)

    if not USE_SOFT_ASSIGNMENT:
        # HARD: pick argmin score per group
        # sort by (gid, score) then pick first in each gid
        order = np.lexsort((score, gid))
        gid_s = gid[order]
        first = np.empty_like(gid_s, dtype=bool)
        first[0] = True
        first[1:] = gid_s[1:] != gid_s[:-1]
        idx_min = order[first]
        w[idx_min] = 1.0
    else:
        # SOFT: softmax over (-score/T) within each group
        T = float(SOFT_TEMPERATURE)
        if T <= 0:
            raise ValueError("SOFT_TEMPERATURE must be > 0")

        a = -score / T  # (N,)

        # sort by gid so groups are contiguous
        order = np.argsort(gid, kind="mergesort")
        gid_s = gid[order]
        a_s = a[order]

        # group boundaries
        start = np.empty_like(gid_s, dtype=bool)
        start[0] = True
        start[1:] = gid_s[1:] != gid_s[:-1]
        starts = np.flatnonzero(start)
        ends = np.r_[starts[1:], len(gid_s)]

        # stable softmax: subtract per-group max
        # compute max per segment
        max_per = np.empty(len(starts), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            max_per[i] = a_s[s:e].max()

        a_s = a_s - np.repeat(max_per, ends - starts)
        p = np.exp(a_s, dtype=np.float32)

        # normalize per segment
        sum_per = np.empty(len(starts), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            sum_per[i] = p[s:e].sum()

        p = p / np.repeat(sum_per + 1e-12, ends - starts)
        w[order] = p.astype(np.float32)

    # Optional gate using MAX_DIST2: if best dist2 in group > MAX_DIST2 -> zero out whole group
    if MAX_DIST2 is not None:
        # find per-group min dist2
        order = np.lexsort((dist2, gid))
        gid_s = gid[order]
        d_s = dist2[order]
        first = np.empty_like(gid_s, dtype=bool)
        first[0] = True
        first[1:] = gid_s[1:] != gid_s[:-1]
        idx_min = order[first]
        bad_gid = gid[idx_min][dist2[idx_min] > float(MAX_DIST2)]
        if bad_gid.size:
            bad_mask = np.isin(gid, bad_gid)
            w[bad_mask] = 0.0

    return w


def data_transformer(data_X, data_Y, step=0):
    X = data_X.copy()
    Y = data_Y.copy()

    # --- covariance validity ---
    mC0 = X["mC[0]"].to_numpy()
    mC2 = X["mC[2]"].to_numpy()
    ok_cov = (mC0 < 2) & (mC2 < 2) & np.isfinite(mC0) & np.isfinite(mC2)

    # --- residual pre-cut ---
    dy0 = (Y["mP[0]"].to_numpy() - Y["yy"].to_numpy())
    dz0 = (Y["mP[1]"].to_numpy() - Y["zz"].to_numpy())
    ok_res = (np.abs(dy0) < 5) & (np.abs(dz0) < 5) & np.isfinite(dy0) & np.isfinite(dz0)

    mask = ok_cov & ok_res
    X = X.loc[mask].reset_index(drop=True)
    Y = Y.loc[mask].reset_index(drop=True)

    # --- build weights using internal_trkid + row key ---
    att_w = _make_attachment_weights(X).astype(np.float32)

    # --- ensure cs0..cs3 exist (from clusterState) ---
    if "cs0" not in X.columns:
        if "clusterState" not in X.columns:
            raise KeyError("Need either cs0..cs3 or clusterState in the loaded data.")
        s = X["clusterState"].to_numpy(dtype=np.uint8)
        # (optional) keep as df columns if you like, but faster is to build X_in via numpy (see below)
        X["cs0"] = ((s >> 0) & 1).astype(np.float32)
        X["cs1"] = ((s >> 1) & 1).astype(np.float32)
        X["cs2"] = ((s >> 2) & 1).astype(np.float32)
        X["cs3"] = ((s >> 3) & 1).astype(np.float32)

    # --- build targets ---
    dy = (Y["mP[0]"].to_numpy(np.float32) - Y["yy"].to_numpy(np.float32))
    dz = (Y["mP[1]"].to_numpy(np.float32) - Y["zz"].to_numpy(np.float32))
    CY = X["mC[0]"].to_numpy(np.float32)
    CZ = X["mC[2]"].to_numpy(np.float32)

    if step == 0:
        target = np.stack([dy*dy - CY, dz*dz - CZ], axis=1).astype(np.float32)  # (N,2)
    if step == 1:
        target = np.stack([dy, dz, CY, CZ, att_w], axis=1).astype(np.float32)

    # --- build fixed-order inputs for the network (19 cols) ---
    X_in = X[labels_x_NN].to_numpy(dtype=np.float32)

    X_in = np.nan_to_num(X_in, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return X_in, target


class network(nn.Module):
    def __init__(self, engineered=True):
        super().__init__()
        self.output_nodes = 2

        self.engineered = engineered
        self.input_dim = len(labels_x_NN)  # must match the order and number of columns in X_in from data_transformer

        # how many extra features feature_engineering() will append
        # precompute indices once (Python, no numpy)
        self.idx_xx  = labels_x_NN.index("xx")      if "xx" in labels_x_NN else None
        self.idx_mp0 = labels_x_NN.index("mP[0]")   if "mP[0]" in labels_x_NN else None
        self.idx_mp1 = labels_x_NN.index("mP[1]")   if "mP[1]" in labels_x_NN else None

        mask_accept = [(name not in ("yy", "zz")) for name in labels_x_NN]
        self.register_buffer("mask_accept", torch.tensor(mask_accept, dtype=torch.bool))

        dim0 = int(self.mask_accept.sum().item()) if engineered else self.input_dim

        self.seq_net = nn.Sequential(
            nn.Linear(dim0, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, self.output_nodes)
        )

        self.out_act = nn.Softplus(beta=1.0)

    def __weight_init__(self, weight_init=None):
        if weight_init is not None:
            for layer in self.seq_net:
                if hasattr(layer, 'weight'):
                    weight_init(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.1)

    def feature_engineering(self, X_in: torch.Tensor) -> torch.Tensor:
        # avoid in-place edits on an input that may be viewed by exporter
        X = X_in.clone()

        if self.idx_xx is not None:
            X[:, self.idx_xx] = X[:, self.idx_xx] / 250.0
        if self.idx_mp0 is not None:
            X[:, self.idx_mp0] = X[:, self.idx_mp0] / 250.0
        if self.idx_mp1 is not None:
            X[:, self.idx_mp1] = X[:, self.idx_mp1] / 250.0

        X = X[:, self.mask_accept]  # torch boolean mask
        return X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.feature_engineering(X) if self.engineered else X
        out = self.seq_net(out)
        return out.square()


def NLLloss(pred, target, eps=1e-6):
    """
    pred:   (N,2) predicted [RY, RZ] (positive)
    target: (N,5) [dy, dz, CY, CZ, att_w]
            att_w is either:
              - hard mask {0,1}, OR
              - soft weights that sum to 1 per group (after gating may sum < 1 over whole batch)
    """
    dy = target[:, 0]
    dz = target[:, 1]
    CY = target[:, 2]
    CZ = target[:, 3]
    w  = target[:, 4]

    RY = pred[:, 0]
    RZ = pred[:, 1]

    Sy = torch.clamp(CY + RY, min=eps)
    Sz = torch.clamp(CZ + RZ, min=eps)

    lossY = torch.log(Sy) + (dy * dy) / Sy
    lossZ = torch.log(Sz) + (dz * dz) / Sz
    loss = lossY + lossZ  # (N,)

    # weighted mean; works for both hard and soft
    w = torch.clamp(w, min=0.0)
    wsum = w.sum()

    if wsum.item() == 0.0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)

    out = (w * loss).sum() / (wsum + eps)

    if not torch.isfinite(out):
        print("pred min/max", pred.min().item(), pred.max().item())
        print("CY min/max", CY.min().item(), CY.max().item())
        print("CZ min/max", CZ.min().item(), CZ.max().item())
        print("Sy min/max", Sy.min().item(), Sy.max().item())
        print("Sz min/max", Sz.min().item(), Sz.max().item())
        print("w min/max", w.min().item(), w.max().item(), "wsum", wsum.item())
        raise RuntimeError("non-finite loss")

    return out


TEST_SIZE = 0.2
SHUFFLE = True

training_configs = [
    {
        "GLOBAL": {
            "loss_function": nn.MSELoss(),
            "epochs": 0
        }
    },
    {
        "GLOBAL": {
            "loss_function": NLLloss,
            "epochs": 300
        },
        "OPTIMIZER_OPTIONS": {
            "lr": 0.003,
            "weight_decay": 0
        },
        "SCHEDULER_OPTIONS": {
            "patience": 20,
            "factor": 0.5
        }
    }
]

dict_net = {
    "GLOBAL": {
        "epochs": 1000,
        "loss_function": NLLloss,
        "optimizer": optim.Adam,
        "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "weight_init": torch.nn.init.xavier_uniform_,
        "amp": True,
        "quantization": False
    },
    "DATA_OPTIONS": {
        "batchsize_schedule": [0, 20, 50, 80, 120, 180],
        "batchsize_training": [131072, 65536, 32768, 16384, 4096, 2048],
        "batchsize_validation": 131072,
        "shuffle_every_epoch": True,
        "num_workers": 0,
        "pin_memory": None,
        "copy_to_device": True
    },
    "AMP_OPTIONS": {
        "dtype": torch.float32,
    },
    "QUANTIZATION_OPTIONS": {
        "ONNX": {
            "weight_type": QuantType.QInt8,
            "per_channel": False,
        },
        "PYTORCH": {
            "dtype": torch.qint8,
        },
    },
    "LOSS_OPTIONS": {
    },
    "OPTIMIZER_OPTIONS": {
        "lr": 0.001,
        "weight_decay": 0
    },
    "SCHEDULER_OPTIONS": {
        "patience": 10,
        "factor": 0.3
    },
    "MACHINE_OPTIONS": {
        "device": None,
        "cpu_threads": None,
        "dtype": torch.float32
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
    "OTHER": {
        "verbose": True,
        "n_samples": np.inf,
        "save_at_epochs": None,
    }
}

dict_data = {
    "GLOBAL": {
        "transform_data": False,
        "copy_to_device": False,
    },
    "SCALERS": {
        "X_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
        "Y_data_scalers": [('yeo-johnson', preprocessing.PowerTransformer(method='yeo-johnson', standardize=True))],
    },
    "MACHINE_OPTIONS": {
        "device": None,
        "dtype_X": torch.float32,
        "dtype_Y": torch.float32,
        "amp": False,
    },
    "OTHER": {
        "num_workers": 0
    }
}