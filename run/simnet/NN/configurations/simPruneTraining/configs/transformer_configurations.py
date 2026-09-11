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
    "epochs": 120,
    "batch_size": 4096,
    "learning_rate": 3.0e-4,
    "weight_decay": 1.0e-4,
    "early_stopping_patience": 18,
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


class FeatureSelfAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.value_scale = nn.Parameter(torch.empty(input_dim, d_model))
        self.value_bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.feature_embedding = nn.Parameter(torch.empty(input_dim, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value_scale)
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.value_scale.unsqueeze(0) + self.value_bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded[:, 0])
        return self.head(pooled).squeeze(-1)


class network(FeatureSelfAttentionClassifier):
    pass
