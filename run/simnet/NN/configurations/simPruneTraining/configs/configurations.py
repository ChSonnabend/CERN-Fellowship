import torch
import torch.nn as nn

data_path = "/lustre/alice/users/csonnab/cern-fellowship/run/simnet/data/output/o2_kine_training_sgn_downsampled.csv"
label_y = "can_avoid_geant"

labels_x = [
    "pdg",
    "abs_pdg",
    "charge_sign",
    "mass",
    "energy",
    "ekin",
    "px",
    "py",
    "pz",
    "p",
    "pt",
    "eta",
    "phi",
    "theta",
    "rapidity",
    "vx",
    "vy",
    "vz",
    "t_ns",
    "dx_from_event",
    "dy_from_event",
    "dz_from_event",
    "r_xy",
    "r_from_event_xy",
    "r3_from_event",
    "mother_id",
    "second_mother_id",
    "process",
    "status_code",
    "weight",
    "to_be_done",
    "inhibited",
    "is_transported",
    "is_primary",
]

num_datapoints = 1_000_000
test_size = 0.2
validation_size = 0.1
random_state = 42
shuffle = True

training = {
    "epochs": 80,
    "batch_size": 16384,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "early_stopping_patience": 12,
    "num_workers": 0,
    "use_amp": True,
    "class_weighting": True,
}

bdt = {
    "scale_inputs": False,
    "model_settings": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "random_state": random_state,
        "verbose": 1,
    },
}


class network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
