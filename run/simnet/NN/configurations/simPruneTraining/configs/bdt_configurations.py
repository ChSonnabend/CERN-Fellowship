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
    "epochs": 1,
    "batch_size": 16384,
    "learning_rate": 0.05,
    "weight_decay": 0.0,
    "early_stopping_patience": 1,
    "num_workers": 0,
    "use_amp": False,
    "class_weighting": False,
}

bdt = {
    "scale_inputs": False,
    "model_settings": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "random_state": random_state,
        "verbose": 1,
    },
}
