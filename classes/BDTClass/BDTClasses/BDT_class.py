import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BDTClassifier:
    def __init__(self, features, label, model_settings=None, scale_inputs=False):
        self.features = list(features)
        self.label = label
        self.model_settings = dict(model_settings or {})
        self.scale_inputs = bool(scale_inputs)
        self.scaler = StandardScaler() if self.scale_inputs else None
        self.model = GradientBoostingClassifier(**self.model_settings)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        x_train = self._fit_transform_x(x_train)
        self.model.fit(x_train, y_train.astype(np.int32))
        history = []
        if x_val is not None and y_val is not None:
            history.append({"split": "validation", **self.evaluate(x_val, y_val)})
        return history

    def predict_proba(self, x):
        x = self._transform_x(x)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.decision_function(x)

    def evaluate(self, x, y, threshold=0.5):
        y_true = y.astype(np.int32)
        y_score = self.predict_proba(x)
        y_pred = (y_score >= threshold).astype(np.int32)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "average_precision": float(average_precision_score(y_true, y_score)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        return metrics

    def save(self, output_dir, extra_info=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "features": self.features,
                "label": self.label,
                "model_settings": self.model_settings,
                "scale_inputs": self.scale_inputs,
            },
            output_dir / "bdt.joblib",
        )
        info = {
            "features": self.features,
            "label": self.label,
            "model_settings": self.model_settings,
            "scale_inputs": self.scale_inputs,
        }
        if extra_info:
            info.update(extra_info)
        with (output_dir / "bdt_metadata.json").open("w") as f:
            json.dump(info, f, indent=2)

    def save_onnx(self, output_path, input_dim):
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:
            raise RuntimeError(
                "BDT ONNX export needs skl2onnx. Install it in the training "
                "environment, e.g. `python3 -m pip install skl2onnx`."
            ) from exc

        estimator = self.model
        options = {id(self.model): {"zipmap": False}}
        if self.scaler is not None:
            estimator = Pipeline([("scaler", self.scaler), ("bdt", self.model)])

        initial_types = [("input", FloatTensorType([None, int(input_dim)]))]
        onnx_model = convert_sklearn(
            estimator,
            initial_types=initial_types,
            target_opset=15,
            options=options,
        )
        with Path(output_path).open("wb") as f:
            f.write(onnx_model.SerializeToString())

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        obj = cls(
            payload["features"],
            payload["label"],
            payload.get("model_settings", {}),
            payload.get("scale_inputs", False),
        )
        obj.model = payload["model"]
        obj.scaler = payload.get("scaler")
        return obj

    def _fit_transform_x(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.scaler is None:
            return x
        return self.scaler.fit_transform(x).astype(np.float32)

    def _transform_x(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.scaler is None:
            return x
        return self.scaler.transform(x).astype(np.float32)
