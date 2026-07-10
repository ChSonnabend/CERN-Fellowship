import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.optim as optim

class SurfaceMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(64, 64)):
        super().__init__()

        layers = []
        in_features = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h

        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

class HO:
    def __init__(self, settings):
        self.settings = {
            "parser": "part",

            "interpolate": {
                "dict_analyze": None,
                "hidden_sizes": (64, 64),
                "lr": 1e-2,
                "weight_decay": 1e-6,
                "n_epochs": 4000,
                "verbose": True,
                "seed": 0,
                "log_space": True,
                "use_validation_model": False,
                "validation_split": 0.2,

                "do_plot": False,
                "plot_mode": "slice",
                "plot_space": "modelspace",
                "plot_axes": None,
                "plot_center": None,

                "find_extremum": False,
                "extremum_mode": "max",
                "bounds": None,
                "n_candidates": 2048,
                "n_starts": 16,
            },

            "plot_surface": {
                "axis_u": None,
                "axis_v": None,
                "mode": "slice",
                "plot_space": "modelspace",
                "center": None,
                "grid_size": 200,
                "center_data": True,
                "percentile_span": 1.0,
                "show_contours": True,
                "show_extremum": True,
                "extremum_result": None,
                "extremum_marker": "*",
                "extremum_color": "cyan",
                "extremum_size": 260,
                "contour_levels": 12,
                "contour_color": "white",
                "contour_linewidths": 1.2,
                "contour_alpha": 0.9,
                "label_contours": True,
                "show_points": True,
                "log_space_grid": (False, False),
                "scale_axis": ("linear", "linear"),
                "limit_axis": None,
                "span": None,
                "colormap_surface": plt.cm.magma,
                "colormap_points": plt.cm.viridis,
                "title": None,
                "xlabel": "ax1",
                "ylabel": "ax2",
                "figsize": (11, 9),
            },

            "latin_hypercube": {
                "include_center": True,
                "log_space": True,
                "bounds": None,
                "random_state": None,

                "dims": (0, 1),
                "center": None,
                "center_data": False,
                "mode": "project",
                "axis_u": None,
                "axis_v": None,
                "slice_tol": 0.1,
                "labels": None,
                "xscale": "linear",
                "yscale": "linear",
                "title": None,
                "figsize": (7, 6),
            },

            "results": {}
        }
        self.settings.update(settings)
        if isinstance(self.settings["parser"], str):
            if self.settings["parser"] == "part":
                self.parser_func = self._parse_part
            elif self.settings["parser"] == "part_2":
                self.parser_func = self._parse_part_2
            else:
                raise ValueError(f"Unknown parser string: {self.settings['parser']}")

    def _parse_part(self, x):
        return float(f"{x[0]}.{x[1:]}")

    def _parse_part_2(self, x):
        return float(x)

    def _parse_points_from_dict(self, dict_analyze, sep="_"):
        """
        Convert a dict of the form
            {"x1_x2_..._xN": value, ...}
        into:
            points: shape (n_samples, n_dim)
            vals:   shape (n_samples,)
        """
        keys = list(dict_analyze.keys())
        pts = []

        for k in keys:
            parts = k.split(sep)
            pts.append([self.parser_func(p) for p in parts])

        points = np.asarray(pts, dtype=float)
        vals = np.asarray([dict_analyze[k] for k in keys], dtype=float)

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Input must contain points with dimension N >= 2.")

        return points, vals

    def _orthonormalize_two_vectors(self, v1, v2, atol=1e-12):
        """
        Gram-Schmidt orthonormalization of two vectors.
        """
        v1 = np.asarray(v1, dtype=float).copy()
        v2 = np.asarray(v2, dtype=float).copy()

        n1 = np.linalg.norm(v1)
        if n1 < atol:
            raise ValueError("First axis vector has near-zero norm.")
        e1 = v1 / n1

        v2 = v2 - np.dot(v2, e1) * e1
        n2 = np.linalg.norm(v2)
        if n2 < atol:
            raise ValueError("Second axis vector is linearly dependent on the first.")
        e2 = v2 / n2

        return e1, e2

    def _model_space_to_points(self, fit_info, Xn):
        """
        Normalized model space -> raw points.
        """
        Xn = np.asarray(Xn, dtype=np.float32)

        if Xn.ndim == 1:
            Xn = Xn[None, :]

        X_transformed = Xn * fit_info["x_std"] + fit_info["x_mean"]
        pts = X_transformed.copy()

        log_mask = np.asarray(fit_info["log_space"], dtype=bool)
        for i in range(X_transformed.shape[1]):
            if log_mask[i]:
                pts[:, i] = 10.0 ** X_transformed[:, i]

        return pts

    def _points_to_model_space(self, fit_info, points):
        """
        Raw points -> normalized model space.
        """
        points = np.asarray(points, dtype=np.float32)

        if points.ndim == 1:
            points = points[None, :]

        if points.shape[1] != fit_info["input_dim"]:
            raise ValueError("Point dimensionality does not match the fitted model.")

        log_mask = np.asarray(fit_info["log_space"], dtype=bool)

        X_transformed = points.copy()
        for i in range(points.shape[1]):
            if log_mask[i]:
                if np.any(points[:, i] <= 0):
                    raise ValueError(f"All coordinates in dimension {i} must be > 0 because that dimension uses log10-space.")
                X_transformed[:, i] = np.log10(points[:, i])

        Xn = (X_transformed - fit_info["x_mean"]) / fit_info["x_std"]
        return Xn

    def _make_raw_plot_basis(self, fit_info, center_model, axis_u_model, axis_v_model, eps=1e-4):
        """
        Build a local 2D basis in X_raw induced by the model-space directions.

        The plane is defined in model space, but for plotting in X_raw we need
        corresponding raw-space tangent directions around the chosen center.
        """
        center_model = np.asarray(center_model, dtype=float)
        axis_u_model = np.asarray(axis_u_model, dtype=float)
        axis_v_model = np.asarray(axis_v_model, dtype=float)

        u_hat_model, v_hat_model = self._orthonormalize_two_vectors(axis_u_model, axis_v_model)

        p0 = self._model_space_to_points(fit_info, center_model)[0]
        pu = self._model_space_to_points(fit_info, center_model + eps * u_hat_model)[0]
        pv = self._model_space_to_points(fit_info, center_model + eps * v_hat_model)[0]

        u_raw = (pu - p0) / eps
        v_raw = (pv - p0) / eps
        u_hat_raw, v_hat_raw = self._orthonormalize_two_vectors(u_raw, v_raw)

        return {
            "center_raw": p0,
            "u_hat_raw": u_hat_raw,
            "v_hat_raw": v_hat_raw,
            "u_hat_model": u_hat_model,
            "v_hat_model": v_hat_model,
        }

    # ============================================================
    # PCA
    # ============================================================

    def compute_pca_directions(self, fit_info, n_components=None):
        """
        PCA of the dataset in normalized model space (Xn).

        Returns both:
        - principal directions in model space
        - local raw-space directions at the PCA center
        """
        X = np.asarray(fit_info["Xn"], dtype=float)
        center_model = X.mean(axis=0)
        Xc = X - center_model

        U, S, VT = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S ** 2) / max(X.shape[0] - 1, 1)
        ratio = eigvals / np.sum(eigvals)

        if n_components is None:
            n_components = X.shape[1]

        components_model = VT[:n_components].copy()
        center_raw = self._model_space_to_points(fit_info, center_model)[0]

        components_raw_local = []
        for k in range(components_model.shape[0]):
            info = self._make_raw_plot_basis(
                fit_info=fit_info,
                center_model=center_model,
                axis_u_model=components_model[k],
                axis_v_model=np.roll(components_model[k], 1),  # temporary helper vector
                eps=1e-4
            )
            components_raw_local.append(info["u_hat_raw"])

        components_raw_local = np.asarray(components_raw_local)

        return {
            "center": center_model.copy(),              # backward compatibility
            "center_model": center_model.copy(),
            "center_raw": center_raw.copy(),
            "components": components_model.copy(),      # backward compatibility
            "components_model": components_model.copy(),
            "components_raw_local": components_raw_local.copy(),
            "explained_variance": eigvals[:n_components].copy(),
            "explained_variance_ratio": ratio[:n_components].copy(),
        }

    # ============================================================
    # main high-level function
    # ============================================================

    def fit_nn_surface_nd(self, points=None, vals=None):
        """
        Fit NN surrogate in N dimensions.
        Public API uses self.settings["interpolate"].
        """

        cfg = self.settings.get("interpolate", {})

        if points is None:
            points = cfg["points"]
        if vals is None:
            vals = cfg["vals"]

        hidden_sizes = cfg.get("hidden_sizes", (64, 64))
        lr = cfg.get("lr", 1e-2)
        weight_decay = cfg.get("weight_decay", 1e-6)
        n_epochs = cfg.get("n_epochs", 4000)
        verbose = cfg.get("verbose", True)
        seed = cfg.get("seed", 0)
        log_space = cfg.get("log_space", True)
        use_validation_model = bool(cfg.get("use_validation_model", False))
        validation_split = float(cfg.get("validation_split", 0.2))

        points = np.asarray(points, dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("points must have shape (n_samples, n_dim) with n_dim >= 2")
        if vals.ndim != 1 or vals.shape[0] != points.shape[0]:
            raise ValueError("vals must have shape (n_samples,)")

        np.random.seed(seed)
        torch.manual_seed(seed)

        d = points.shape[1]
        X_raw = points.copy()

        if isinstance(log_space, bool):
            log_mask = np.full(d, log_space, dtype=bool)
        else:
            if len(log_space) != d:
                raise ValueError("If log_space is a list, it must have length equal to input dimension")
            log_mask = np.asarray(log_space, dtype=bool)

        X_transformed = X_raw.copy()
        for i in range(d):
            if log_mask[i]:
                if np.any(X_raw[:, i] <= 0):
                    raise ValueError(f"All coordinates in dimension {i} must be > 0 when log_space[{i}] is True")
                X_transformed[:, i] = np.log10(X_raw[:, i])

        t = vals.copy()

        x_mean = X_transformed.mean(axis=0)
        x_std = X_transformed.std(axis=0)
        x_std[x_std == 0.0] = 1.0

        t_mean = float(np.mean(t))
        t_std = float(np.std(t))
        if t_std == 0.0:
            t_std = 1.0

        Xn = (X_transformed - x_mean) / x_std
        tn = (t - t_mean) / t_std

        n_samples = Xn.shape[0]
        do_validation = use_validation_model and (validation_split > 0.0) and (n_samples >= 4)

        train_idx = np.arange(n_samples)
        val_idx = np.array([], dtype=int)
        if do_validation:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_samples)
            n_val = int(round(n_samples * validation_split))
            n_val = max(1, min(n_samples - 1, n_val))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            if train_idx.size == 0:
                do_validation = False
                train_idx = np.arange(n_samples)
                val_idx = np.array([], dtype=int)

        x_tensor = torch.tensor(Xn, dtype=torch.float32)
        t_tensor = torch.tensor(tn, dtype=torch.float32)
        x_train_tensor = torch.tensor(Xn[train_idx], dtype=torch.float32)
        t_train_tensor = torch.tensor(tn[train_idx], dtype=torch.float32)
        x_val_tensor = None
        t_val_tensor = None
        if do_validation:
            x_val_tensor = torch.tensor(Xn[val_idx], dtype=torch.float32)
            t_val_tensor = torch.tensor(tn[val_idx], dtype=torch.float32)

        model = SurfaceMLP(input_dim=d, hidden_sizes=hidden_sizes)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        loss_history = []
        val_loss_history = []
        best_val_loss = np.inf
        best_epoch = -1
        best_state = None

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = model(x_train_tensor)
            loss = criterion(pred, t_train_tensor)
            loss.backward()
            optimizer.step()

            loss_history.append(float(loss.item()))

            if do_validation:
                with torch.no_grad():
                    pred_val = model(x_val_tensor)
                    val_loss = criterion(pred_val, t_val_tensor)
                    val_loss_value = float(val_loss.item())
                val_loss_history.append(val_loss_value)
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())

            if verbose and ((epoch + 1) % 500 == 0 or epoch == 0):
                if do_validation:
                    print(f"epoch {epoch + 1:5d}  train_loss = {loss.item():.6g}  val_loss = {val_loss_value:.6g}")
                else:
                    print(f"epoch {epoch + 1:5d}  loss = {loss.item():.6g}")

        if do_validation and best_state is not None:
            model.load_state_dict(best_state)

        with torch.no_grad():
            pred_train_n = model(x_tensor).cpu().numpy()
            pred_train = pred_train_n * t_std + t_mean

        residuals = t - pred_train
        rmse = float(np.sqrt(np.mean((t - pred_train) ** 2)))
        mape = float(np.mean(np.abs(t - pred_train) / np.maximum(np.abs(t), 1.0)))

        train_residuals = t[train_idx] - pred_train[train_idx]
        train_rmse = float(np.sqrt(np.mean((t[train_idx] - pred_train[train_idx]) ** 2)))
        train_mape = float(np.mean(np.abs(t[train_idx] - pred_train[train_idx]) / np.maximum(np.abs(t[train_idx]), 1.0)))

        val_rmse = None
        val_mape = None
        if do_validation and val_idx.size > 0:
            val_rmse = float(np.sqrt(np.mean((t[val_idx] - pred_train[val_idx]) ** 2)))
            val_mape = float(np.mean(np.abs(t[val_idx] - pred_train[val_idx]) / np.maximum(np.abs(t[val_idx]), 1.0)))

        fit_info = {
            "model": model,
            "input_dim": d,
            "log_space": log_mask.copy(),
            "points": points.astype(float),
            "vals": t.astype(float),
            "X_raw": X_raw.astype(float),
            "X_transformed": X_transformed.astype(float),
            "Xn": Xn.astype(float),
            "x_mean": x_mean.astype(float),
            "x_std": x_std.astype(float),
            "t_mean": float(t_mean),
            "t_std": float(t_std),
            "pred_train": pred_train.astype(float),
            "residuals": residuals.astype(float),
            "rmse": rmse,
            "mape": mape,
            "train_indices": train_idx.astype(int),
            "val_indices": val_idx.astype(int),
            "train_rmse": train_rmse,
            "train_mape": train_mape,
            "val_rmse": val_rmse,
            "val_mape": val_mape,
            "used_validation_model": bool(do_validation),
            "best_val_loss": None if not do_validation else float(best_val_loss),
            "best_epoch": None if not do_validation else int(best_epoch),
            "loss_history": np.asarray(loss_history, dtype=float),
            "val_loss_history": np.asarray(val_loss_history, dtype=float),
        }

        self.settings["fit_info"] = fit_info
        self.settings.setdefault("results", {})["fit"] = fit_info
        return fit_info

    def grid_interpolate_nn_nd(self):
        """
        Main high-level workflow driven by self.settings["interpolate"].
        """
        cfg = self.settings.get("interpolate", {})
        dict_analyze = cfg["dict_analyze"]

        points, vals = self._parse_points_from_dict(dict_analyze)

        cfg["points"] = points
        cfg["vals"] = vals
        self.settings["interpolate"] = cfg

        fit_info = self.fit_nn_surface_nd(points=points, vals=vals)

        pca_info = self.compute_pca_directions(fit_info)

        extremum_result = None
        if cfg.get("find_extremum", False):
            extremum_result = self.find_surface_extremum_nn_nd(fit_info=fit_info)

            label = "Maximum" if extremum_result["mode"] == "max" else "Minimum"
            print(f"{label} in requested region:")
            print("  x_opt     =", np.array2string(extremum_result["x_opt"], precision=6))
            print(f"  value_opt = {extremum_result['value_opt']:.6g}")

        plot_result = None
        if cfg.get("do_plot", False):
            d = fit_info["input_dim"]
            plot_axes = cfg.get("plot_axes", None)
            plot_center = cfg.get("plot_center", None)

            if plot_axes is None:
                if pca_info["components_model"].shape[0] < 2:
                    raise ValueError("Need at least two PCA components to build a 2D plot")
                axis_u = pca_info["components_model"][0]
                axis_v = pca_info["components_model"][1]
            else:
                if len(plot_axes) != 2:
                    raise ValueError("plot_axes must be a tuple/list like (axis_u, axis_v)")
                axis_u, axis_v = plot_axes
                axis_u = np.asarray(axis_u, dtype=float)
                axis_v = np.asarray(axis_v, dtype=float)
                if axis_u.shape != (d,) or axis_v.shape != (d,):
                    raise ValueError(f"plot_axes vectors must each have shape ({d},)")

            if plot_center is None:
                plot_center_model = pca_info["center_model"]
            else:
                plot_center_model = np.asarray(plot_center, dtype=float)
                if plot_center_model.shape != (d,):
                    raise ValueError(f"plot_center must have shape ({d},)")

            self.settings.setdefault("plot_surface", {})
            self.settings["plot_surface"]["axis_u"] = axis_u
            self.settings["plot_surface"]["axis_v"] = axis_v
            self.settings["plot_surface"]["mode"] = cfg.get("plot_mode", "slice")
            self.settings["plot_surface"]["center"] = plot_center_model
            self.settings["plot_surface"]["plot_space"] = cfg.get("plot_space", "modelspace")
            self.settings["plot_surface"]["extremum_result"] = extremum_result

            plot_result = self.plot_surface_2d()

        print(f"Train RMSE  = {fit_info['rmse']:.6g}")
        print(f"Train MAPE  = {fit_info['mape']:.6g}")

        result = {
            "points": points,
            "vals": vals,
            "fit_info": fit_info,
            "pca_info": pca_info,
            "extremum_result": extremum_result,
            "x_opt": None if extremum_result is None else extremum_result["x_opt"],
            "value_opt": None if extremum_result is None else extremum_result["value_opt"],
            "plot_result": plot_result,
        }

        self.settings.setdefault("results", {})["interpolate"] = result
        self.settings["result"] = result
        return result

    def evaluate_nn_surface_nd(self, points, fit_info=None):
        """
        Evaluate fitted NN surface at one point or many points.
        """
        fit_info = self.settings.get("fit_info", None)
        if fit_info is None:
            raise ValueError("No fit_info available. Run fit_nn_surface_nd first.")

        points = np.asarray(points, dtype=np.float32)
        scalar_input = (points.ndim == 1)

        Xn = self._points_to_model_space(fit_info, points)

        with torch.no_grad():
            x_tensor = torch.tensor(Xn, dtype=torch.float32)
            pred_n = fit_info["model"](x_tensor).cpu().numpy()

        pred = pred_n * fit_info["t_std"] + fit_info["t_mean"]

        if scalar_input:
            return float(pred[0])
        return pred

    # ============================================================
    # optimization on fitted surface
    # ============================================================

    def find_surface_extremum_nn_nd(self, fit_info=None):
        """
        Find min or max of the fitted NN surface in a bounded hyper-rectangle.
        Uses self.settings["interpolate"].
        """
        if fit_info is None:
            fit_info = self.settings.get("fit_info", None)
            if fit_info is None:
                raise ValueError("No fit_info available. Run fit_nn_surface_nd first.")

        cfg = self.settings.get("interpolate", {})

        bounds = cfg.get("bounds", None)
        mode = cfg.get("extremum_mode", cfg.get("mode", "max"))
        n_candidates = cfg.get("n_candidates", 2048)
        n_starts = cfg.get("n_starts", 16)
        random_state = cfg.get("seed", cfg.get("random_state", 0))
        optimizer = str(cfg.get("optimizer", "lbfgsb")).lower()
        de_maxiter = int(cfg.get("de_maxiter", 200))
        de_popsize = int(cfg.get("de_popsize", 15))
        de_polish = bool(cfg.get("de_polish", False))

        if bounds is None:
            points = np.asarray(fit_info["X_raw"], dtype=float)
            bounds = [(points[:, j].min(), points[:, j].max()) for j in range(points.shape[1])]

        bounds = np.asarray(bounds, dtype=float)
        d = fit_info["input_dim"]
        log_mask = np.asarray(fit_info["log_space"], dtype=bool)

        if bounds.shape != (d, 2):
            raise ValueError(f"bounds must have shape ({d}, 2)")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each bound must satisfy low < high")
        if np.any(log_mask) and np.any(bounds[log_mask, :] <= 0):
            raise ValueError("All bounds must be > 0 in dimensions where log_space=True")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if n_starts < 1:
            raise ValueError("n_starts must be >= 1")

        rng = np.random.RandomState(random_state)

        lo = bounds[:, 0]
        hi = bounds[:, 1]

        lo_t = lo.copy()
        hi_t = hi.copy()
        lo_t[log_mask] = np.log10(lo[log_mask])
        hi_t[log_mask] = np.log10(hi[log_mask])

        U = np.zeros((n_candidates, d), dtype=float)
        for j in range(d):
            u = (np.arange(n_candidates) + rng.uniform(size=n_candidates)) / float(n_candidates)
            rng.shuffle(u)
            U[:, j] = u

        Xcand_t = lo_t[None, :] + U * (hi_t - lo_t)[None, :]
        Pcand = Xcand_t.copy()
        Pcand[:, log_mask] = 10.0 ** Pcand[:, log_mask]

        Vcand = self.evaluate_nn_surface_nd(Pcand, fit_info=fit_info)

        order = np.argsort(Vcand)
        if mode == "max":
            order = order[::-1]

        n_use = min(n_starts, len(order))
        starts_t = Xcand_t[order[:n_use]]

        def objective(x_t):
            p = np.asarray(x_t, dtype=float).copy()
            p[log_mask] = 10.0 ** p[log_mask]
            val = float(self.evaluate_nn_surface_nd(p, fit_info=fit_info))
            return -val if mode == "max" else val

        opt_bounds = list(zip(lo_t, hi_t))
        best_res = None

        if optimizer in ("lbfgsb", "multi_start_lbfgsb"):
            for x0 in starts_t:
                res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=opt_bounds)
                if (best_res is None) or (res.fun < best_res.fun):
                    best_res = res

        elif optimizer in ("de", "differential_evolution"):
            best_res = differential_evolution(
                objective,
                bounds=opt_bounds,
                seed=random_state,
                maxiter=de_maxiter,
                popsize=de_popsize,
                polish=de_polish,
            )

        elif optimizer in ("hybrid", "de_lbfgsb", "hybrid_de_lbfgsb"):
            de_res = differential_evolution(
                objective,
                bounds=opt_bounds,
                seed=random_state,
                maxiter=de_maxiter,
                popsize=de_popsize,
                polish=False,
            )
            best_res = minimize(objective, x0=de_res.x, method="L-BFGS-B", bounds=opt_bounds)

        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                "Valid options: lbfgsb, de, hybrid_de_lbfgsb"
            )

        x_best_t = np.asarray(best_res.x, dtype=float)
        p_best = x_best_t.copy()
        p_best[log_mask] = 10.0 ** p_best[log_mask]

        value_best = float(self.evaluate_nn_surface_nd(p_best, fit_info=fit_info))

        result = {
            "mode": mode,
            "optimizer": optimizer,
            "x_opt": p_best.astype(float),
            "x_opt_transformed": x_best_t.astype(float),
            "value_opt": value_best,
            "bounds": bounds.astype(float),
            "bounds_transformed": np.column_stack([lo_t, hi_t]).astype(float),
            "optimizer_result": best_res,
            "success": bool(best_res.success),
            "message": str(best_res.message),
            "fun": float(best_res.fun),
            "nfev": int(getattr(best_res, "nfev", -1)),
            "nit": int(getattr(best_res, "nit", -1)),
            "candidate_values": np.asarray(Vcand, dtype=float),
            "candidate_points": np.asarray(Pcand, dtype=float),
        }

        self.settings.setdefault("results", {})["extremum"] = result
        self.settings["extremum_result"] = result
        return result

    # ============================================================
    # generate new configs
    # ============================================================

    def latin_hypercube_nd(self, center, half_widths, n_samples):
        """
        Latin hypercube in N dimensions.
        Uses self.settings["latin_hypercube"] for optional config.
        """
        cfg = self.settings.get("latin_hypercube", {})

        include_center = cfg.get("include_center", True)
        log_space = cfg.get("log_space", True)
        bounds = cfg.get("bounds", None)
        random_state = cfg.get("random_state", None)

        center = np.asarray(center, dtype=float)
        half_widths = np.asarray(half_widths, dtype=float)

        if center.ndim != 1:
            raise ValueError("center must be 1D")
        if half_widths.shape != center.shape:
            raise ValueError("half_widths must match center shape")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        d = center.size
        rng = np.random.RandomState(random_state)
        n_lhs = n_samples - 1 if include_center else n_samples

        samples = np.zeros((0, d), dtype=float)

        if n_lhs > 0:
            U = np.zeros((n_lhs, d), dtype=float)
            for j in range(d):
                u = (np.arange(n_lhs) + rng.uniform(size=n_lhs)) / float(n_lhs)
                rng.shuffle(u)
                U[:, j] = u

            if log_space:
                if np.any(center <= 0):
                    raise ValueError("All center coordinates must be > 0 when log_space=True")
                lc = np.log10(center)
                lo = lc - half_widths
                hi = lc + half_widths
                X = lo[None, :] + U * (hi - lo)[None, :]
                samples = 10.0 ** X
            else:
                lo = center - half_widths
                hi = center + half_widths
                samples = lo[None, :] + U * (hi - lo)[None, :]

            if bounds is not None:
                bounds = np.asarray(bounds, dtype=float)
                if bounds.shape != (d, 2):
                    raise ValueError("bounds must have shape (D, 2)")
                samples = np.clip(samples, bounds[:, 0], bounds[:, 1])

        if include_center:
            center_row = center[None, :]
            samples = np.vstack([center_row, samples]) if len(samples) else center_row

        result = {
            "samples": samples,
            "center": center.copy(),
            "half_widths": half_widths.copy(),
            "n_samples": int(n_samples),
            "include_center": bool(include_center),
            "log_space": log_space,
            "bounds": None if bounds is None else np.asarray(bounds, dtype=float),
            "random_state": random_state,
        }

        self.settings.setdefault("results", {})["latin_hypercube"] = result
        return result

    # ============================================================
    # 2D plotting
    # ============================================================

    def plot_surface_2d(self):
        """
        Plot a 2D surface using two axes defined in the selected space.

        Semantics:
        - plot_space == "modelspace":
            axis_u, axis_v, center, span are interpreted in model space.
            The grid is constructed in model space.

        - plot_space == "raw":
            axis_u, axis_v, center, span are interpreted in raw space.
            The grid is constructed in raw space, evaluated via
            evaluate_nn_surface_nd(raw_points), and plotted in raw-plane coordinates.
        """
        fit_info = self.settings["fit_info"]
        cfg = self.settings.get("plot_surface", {})

        axis_u = cfg["axis_u"]
        axis_v = cfg["axis_v"]

        mode = cfg.get("mode", "slice")
        plot_space = cfg.get("plot_space", "modelspace")
        center = cfg.get("center", None)
        grid_size = cfg.get("grid_size", 200)
        center_data = cfg.get("center_data", True)
        percentile_span = cfg.get("percentile_span", 1.0)

        show_contours = cfg.get("show_contours", True)
        show_extremum = cfg.get("show_extremum", True)
        extremum_result = cfg.get("extremum_result", None)

        extremum_marker = cfg.get("extremum_marker", "*")
        extremum_color = cfg.get("extremum_color", "cyan")
        extremum_size = cfg.get("extremum_size", 260)

        contour_levels = cfg.get("contour_levels", 12)
        contour_color = cfg.get("contour_color", "white")
        contour_linewidths = cfg.get("contour_linewidths", 1.2)
        contour_alpha = cfg.get("contour_alpha", 0.9)
        label_contours = cfg.get("label_contours", True)

        show_points = cfg.get("show_points", True)
        show_best_sample = cfg.get("show_best_sample", True)
        best_sample_color = cfg.get("best_sample_color", "red")
        best_sample_linewidth = cfg.get("best_sample_linewidth", 2.2)
        best_sample_size = cfg.get("best_sample_size", 280)
        print_plot_diagnostics = cfg.get("print_plot_diagnostics", True)

        log_space_grid = cfg.get("log_space_grid", (False, False))
        scale_axis = cfg.get("scale_axis", ("linear", "linear"))
        limit_axis = cfg.get("limit_axis", None)
        span = cfg.get("span", None)

        colormap_surface = cfg.get("colormap_surface", plt.cm.magma)
        colormap_points = cfg.get("colormap_points", plt.cm.viridis)

        title = cfg.get("title", None)
        xlabel = cfg.get("xlabel", "ax1")
        ylabel = cfg.get("ylabel", "ax2")
        figsize = cfg.get("figsize", (11, 9))
        savefig = cfg.get("savefig", 0)
        showfig = cfg.get("showfig", False)

        if mode not in ("project", "slice"):
            raise ValueError("mode must be 'project' or 'slice'")
        if plot_space not in ("modelspace", "raw"):
            raise ValueError("plot_space must be 'modelspace' or 'raw'")

        d = fit_info["input_dim"]
        axis_u = np.asarray(axis_u, dtype=float)
        axis_v = np.asarray(axis_v, dtype=float)

        if axis_u.shape != (d,) or axis_v.shape != (d,):
            raise ValueError(f"axis_u and axis_v must have shape ({d},)")

        Xn = np.asarray(fit_info["Xn"], dtype=float)
        Xraw = np.asarray(fit_info["X_raw"], dtype=float)
        vals = np.asarray(fit_info["vals"], dtype=float)

        # ------------------------------------------------------------
        # Interpret center and axes in the selected space
        # ------------------------------------------------------------
        if plot_space == "modelspace":
            if center is None:
                center_model = Xn.mean(axis=0)
            else:
                center_model = np.asarray(center, dtype=float)
                if center_model.shape != (d,):
                    raise ValueError(f"center must have shape ({d},)")

            center_raw = self._model_space_to_points(
                fit_info, center_model.reshape(1, -1)
            )[0]

            u_hat_model, v_hat_model = self._orthonormalize_two_vectors(axis_u, axis_v)

            raw_basis = self._make_raw_plot_basis(
                fit_info=fit_info,
                center_model=center_model,
                axis_u_model=u_hat_model,
                axis_v_model=v_hat_model,
            )
            u_hat_raw = raw_basis["u_hat_raw"]
            v_hat_raw = raw_basis["v_hat_raw"]

        else:  # raw
            if center is None:
                center_raw = Xraw.mean(axis=0)
            else:
                center_raw = np.asarray(center, dtype=float)
                if center_raw.shape != (d,):
                    raise ValueError(f"center must have shape ({d},)")

            center_model = self._points_to_model_space(
                fit_info, center_raw.reshape(1, -1)
            )[0]

            u_hat_raw, v_hat_raw = self._orthonormalize_two_vectors(axis_u, axis_v)

            # Local corresponding model directions, only for projections/metadata
            eps = 1e-6
            p_u_raw = center_raw + eps * u_hat_raw
            p_v_raw = center_raw + eps * v_hat_raw

            p_u_model = self._points_to_model_space(fit_info, p_u_raw.reshape(1, -1))[0]
            p_v_model = self._points_to_model_space(fit_info, p_v_raw.reshape(1, -1))[0]

            du_model = p_u_model - center_model
            dv_model = p_v_model - center_model
            u_hat_model, v_hat_model = self._orthonormalize_two_vectors(du_model, dv_model)

        # ------------------------------------------------------------
        # Project data samples into the chosen plane coordinates
        # ------------------------------------------------------------
        if plot_space == "modelspace":
            if center_data:
                Xc = Xn - center_model[None, :]
            else:
                Xc = Xn.copy()

            u_data = Xc @ u_hat_model
            v_data = Xc @ v_hat_model

        else:
            if center_data:
                Xc = Xraw - center_raw[None, :]
            else:
                Xc = Xraw.copy()

            u_data = Xc @ u_hat_raw
            v_data = Xc @ v_hat_raw

        # ------------------------------------------------------------
        # Span in chosen space
        # ------------------------------------------------------------
        if span is None:
            qlo = 0.5 * (1.0 - percentile_span)
            qhi = 1.0 - qlo

            u_min, u_max = np.quantile(u_data, [qlo, qhi])
            v_min, v_max = np.quantile(v_data, [qlo, qhi])

            if np.isclose(u_min, u_max):
                u_min -= 1.0
                u_max += 1.0
            if np.isclose(v_min, v_max):
                v_min -= 1.0
                v_max += 1.0
        else:
            (u_min, u_max), (v_min, v_max) = span

        if log_space_grid[0]:
            if u_min <= 0 or u_max <= 0:
                raise ValueError("u_min and u_max must be > 0 for log-space u-grid")
            u_grid = np.logspace(np.log10(u_min), np.log10(u_max), grid_size)
        else:
            u_grid = np.linspace(u_min, u_max, grid_size)

        if log_space_grid[1]:
            if v_min <= 0 or v_max <= 0:
                raise ValueError("v_min and v_max must be > 0 for log-space v-grid")
            v_grid = np.logspace(np.log10(v_min), np.log10(v_max), grid_size)
        else:
            v_grid = np.linspace(v_min, v_max, grid_size)

        U_plot, V_plot = np.meshgrid(u_grid, v_grid, indexing="ij")

        if plot_space == "modelspace":
            x0_plot = np.dot(center_model, u_hat_model)
            y0_plot = np.dot(center_model, v_hat_model)
        else:
            x0_plot = np.dot(center_raw, u_hat_raw)
            y0_plot = np.dot(center_raw, v_hat_raw)

        if center_data:
            U_local = U_plot
            V_local = V_plot
        else:
            U_local = U_plot - x0_plot
            V_local = V_plot - y0_plot

        # ------------------------------------------------------------
        # Construct plane in chosen space
        # ------------------------------------------------------------
        if plot_space == "modelspace":
            Xplane_model = (
                center_model[None, None, :]
                + U_local[:, :, None] * u_hat_model[None, None, :]
                + V_local[:, :, None] * v_hat_model[None, None, :]
            )
            plane_points_model = Xplane_model.reshape(-1, d)

            plane_points_raw = self._model_space_to_points(fit_info, plane_points_model)
            plane_points_raw_3d = plane_points_raw.reshape(grid_size, grid_size, d)

        else:
            Xplane_raw = (
                center_raw[None, None, :]
                + U_local[:, :, None] * u_hat_raw[None, None, :]
                + V_local[:, :, None] * v_hat_raw[None, None, :]
            )
            plane_points_raw_3d = Xplane_raw
            plane_points_raw = Xplane_raw.reshape(-1, d)

            plane_points_model = self._points_to_model_space(fit_info, plane_points_raw)
            Xplane_model = plane_points_model.reshape(grid_size, grid_size, d)

        # ------------------------------------------------------------
        # Coordinates used for plotting
        # ------------------------------------------------------------
        X_plot_grid = U_plot
        Y_plot_grid = V_plot

        if plot_space == "modelspace":
            if show_points:
                if center_data:
                    Xc = Xn - center_model[None, :]
                else:
                    Xc = Xn
                x_points = Xc @ u_hat_model
                y_points = Xc @ v_hat_model

        else:  # raw
            if show_points:
                if center_data:
                    Xc = Xraw - center_raw[None, :]
                else:
                    Xc = Xraw
                x_points = Xc @ u_hat_raw
                y_points = Xc @ v_hat_raw

        # ------------------------------------------------------------
        # Surface construction
        # ------------------------------------------------------------
        if mode == "project":
            surface = griddata(
                points=np.column_stack([u_data, v_data]),
                values=vals,
                xi=(U_plot, V_plot),
                method="linear",
            )

            surface_nearest = griddata(
                points=np.column_stack([u_data, v_data]),
                values=vals,
                xi=(U_plot, V_plot),
                method="nearest",
            )

            mask_nan = np.isnan(surface)
            surface[mask_nan] = surface_nearest[mask_nan]

        else:
            # evaluate_nn_surface_nd expects raw points
            surface = self.evaluate_nn_surface_nd(
                plane_points_raw, fit_info=fit_info
            ).reshape(grid_size, grid_size)

        fig, ax = plt.subplots(figsize=figsize)

        pcm = ax.pcolormesh(
            X_plot_grid,
            Y_plot_grid,
            surface,
            shading="auto",
            cmap=colormap_surface,
        )
        plt.colorbar(pcm, ax=ax, label="NN fitted value")

        if show_contours:
            cs = ax.contour(
                X_plot_grid,
                Y_plot_grid,
                surface,
                levels=contour_levels,
                colors=contour_color,
                linewidths=contour_linewidths,
                alpha=contour_alpha,
            )
            if label_contours:
                ax.clabel(cs, inline=True, fontsize=10, fmt="%.3g")

        if show_points:
            ax.scatter(
                x_points, y_points,
                c=vals,
                cmap=colormap_points,
                edgecolors="k",
                s=35,
                zorder=3,
                label="samples",
            )

        i_max_data = int(np.argmax(vals))
        x_best_data_raw = Xraw[i_max_data]
        best_data_val = float(vals[i_max_data])

        if plot_space == "modelspace":
            x_best_data_model = self._points_to_model_space(
                fit_info,
                x_best_data_raw.reshape(1, -1),
            )[0]
            delta_best = x_best_data_model - center_model if center_data else x_best_data_model
            x_best_plot = float(np.dot(delta_best, u_hat_model))
            y_best_plot = float(np.dot(delta_best, v_hat_model))
        else:
            delta_best = x_best_data_raw - center_raw if center_data else x_best_data_raw
            x_best_plot = float(np.dot(delta_best, u_hat_raw))
            y_best_plot = float(np.dot(delta_best, v_hat_raw))

        if show_best_sample:
            ax.scatter(
                [x_best_plot], [y_best_plot],
                marker="o",
                facecolors="none",
                edgecolors=best_sample_color,
                linewidths=best_sample_linewidth,
                s=best_sample_size,
                zorder=6,
                label=f"max sample: {best_data_val:.4g}",
            )

        extremum_plot_xy = None
        if show_extremum and extremum_result is not None:
            x_opt_raw = np.asarray(extremum_result["x_opt"], dtype=float)
            if x_opt_raw.shape != (d,):
                raise ValueError(f"extremum_result['x_opt'] must have shape ({d},)")

            x_opt_model = self._points_to_model_space(
                fit_info, x_opt_raw.reshape(1, -1)
            )[0]

            if plot_space == "modelspace":
                delta = x_opt_model - center_model if center_data else x_opt_model
                x_ext = np.dot(delta, u_hat_model)
                y_ext = np.dot(delta, v_hat_model)
            else:
                delta = x_opt_raw - center_raw if center_data else x_opt_raw
                x_ext = np.dot(delta, u_hat_raw)
                y_ext = np.dot(delta, v_hat_raw)

            extremum_plot_xy = np.array([x_ext, y_ext], dtype=float)

            label = extremum_result.get("mode", "extremum")
            val = extremum_result.get("value_opt", None)
            extremum_label = label if val is None else f"{label}: {val:.4g}"

            ax.scatter(
                [x_ext], [y_ext],
                marker=extremum_marker,
                c=extremum_color,
                s=extremum_size,
                edgecolors="k",
                linewidths=1.2,
                zorder=5,
                label=extremum_label,
            )

        if show_points or show_best_sample or (show_extremum and extremum_result is not None):
            ax.legend()

        if print_plot_diagnostics:
            idx_flat = int(np.nanargmax(surface))
            iu, iv = np.unravel_index(idx_flat, surface.shape)
            max_surface_val = float(surface[iu, iv])
            u_at_max = float(U_plot[iu, iv])
            v_at_max = float(V_plot[iu, iv])

            print(
                "[PLOT] panel diagnostics: "
                f"panel='{xlabel} vs {ylabel}'; "
                f"max_sample_value={best_data_val:.8g}; "
                f"max_sample_point={np.array2string(x_best_data_raw, precision=6)}; "
                f"max_surface_on_panel={max_surface_val:.8g} at (u,v)=({u_at_max:.6g},{v_at_max:.6g})"
            )

            if show_extremum and extremum_result is not None:
                ext_val = float(extremum_result.get("value_opt", np.nan))
                ext_point = np.asarray(extremum_result.get("x_opt", np.full(d, np.nan)), dtype=float)
                print(
                    "[PLOT] extremum diagnostics: "
                    f"value_opt={ext_val:.8g}; "
                    f"x_opt={np.array2string(ext_point, precision=6)}"
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if limit_axis is not None:
            if limit_axis[0] is not None:
                ax.set_xlim(limit_axis[0])
            if limit_axis[1] is not None:
                ax.set_ylim(limit_axis[1])

        ax.set_xscale(scale_axis[0])
        ax.set_yscale(scale_axis[1])

        if title is None:
            title = f"2D {mode} in {plot_space}"
        ax.set_title(title)

        plt.tight_layout()

        if savefig:
            plt.savefig(savefig)

        if showfig:
            plt.show()

        plt.close(fig)

        result = {
            "U_grid": U_plot,
            "V_grid": V_plot,
            "U_plot": U_plot,
            "V_plot": V_plot,
            "U_local": U_local,
            "V_local": V_local,
            "X_plot_grid": X_plot_grid,
            "Y_plot_grid": Y_plot_grid,
            "surface": surface,
            "center_model": center_model,
            "center_raw": center_raw,
            "axis_u_model": u_hat_model,
            "axis_v_model": v_hat_model,
            "axis_u_raw": u_hat_raw,
            "axis_v_raw": v_hat_raw,
            "plane_points_model": plane_points_model,
            "plane_points_raw": plane_points_raw,
            "extremum_plot_xy": extremum_plot_xy,
            "plot_space": plot_space,
            "mode": mode,
        }

        self.settings.setdefault("results", {})["plot_surface"] = result
        return result

    def plot_latin_hypercube_nd(self, samples):
        """
        Plot a 2D view of N-dimensional samples.
        """
        cfg = self.settings.get("latin_hypercube", {})

        dims = cfg.get("dims", (0, 1))
        center = cfg.get("center", None)
        center_data = cfg.get("center_data", False)
        mode = cfg.get("mode", "project")
        axis_u = cfg.get("axis_u", None)
        axis_v = cfg.get("axis_v", None)
        slice_tol = cfg.get("slice_tol", 0.1)
        labels = cfg.get("labels", None)
        xscale = cfg.get("xscale", "linear")
        yscale = cfg.get("yscale", "linear")
        title = cfg.get("title", None)
        figsize = cfg.get("figsize", (7, 6))
        savefig = cfg.get("savefig", 0)
        showfig = cfg.get("showfig", False)

        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] < 2:
            raise ValueError("samples must have shape (n_samples, D) with D >= 2")

        n_samples, d = samples.shape

        if mode not in ("project", "slice"):
            raise ValueError("mode must be 'project' or 'slice'")

        if center is None:
            center = samples.mean(axis=0)
        else:
            center = np.asarray(center, dtype=float)
            if center.shape != (d,):
                raise ValueError(f"center must have shape ({d},)")

        using_dims = (axis_u is None or axis_v is None)

        if using_dims:
            i, j = dims
            if i == j:
                raise ValueError("dims must contain two different dimensions")
            if not (0 <= i < d and 0 <= j < d):
                raise ValueError("dims out of range")

            axis_u = np.zeros(d, dtype=float)
            axis_v = np.zeros(d, dtype=float)
            axis_u[i] = 1.0
            axis_v[j] = 1.0

            default_xlabel = labels[i] if labels is not None else f"dim {i}"
            default_ylabel = labels[j] if labels is not None else f"dim {j}"
        else:
            axis_u = np.asarray(axis_u, dtype=float)
            axis_v = np.asarray(axis_v, dtype=float)
            if axis_u.shape != (d,) or axis_v.shape != (d,):
                raise ValueError(f"axis_u and axis_v must have shape ({d},)")
            default_xlabel = "coordinate along axis_u"
            default_ylabel = "coordinate along axis_v"

        u_hat, v_hat = self._orthonormalize_two_vectors(axis_u, axis_v)

        Xc = samples.copy()
        if center_data:
            Xc -= center[None, :]

        u_coords = Xc @ u_hat
        v_coords = Xc @ v_hat

        X_plane_part = (
            u_coords[:, None] * u_hat[None, :]
            + v_coords[:, None] * v_hat[None, :]
        )
        residual = Xc - X_plane_part
        dist_to_plane = np.linalg.norm(residual, axis=1)

        if mode == "project":
            mask = np.ones(n_samples, dtype=bool)
            mode_label = "projection"
        else:
            mask = dist_to_plane <= slice_tol
            mode_label = f"slice (tol={slice_tol:.3g})"

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            u_coords[mask],
            v_coords[mask],
            s=40,
            label=f"samples: {mask.sum()} shown / {n_samples} total"
        )

        if center_data:
            ax.scatter([0.0], [0.0], s=140, marker="*", label="center")
            center_plot = np.array([0.0, 0.0], dtype=float)
        else:
            center_plot = np.array([np.dot(center, u_hat), np.dot(center, v_hat)], dtype=float)
            ax.scatter(center_plot[0], center_plot[1], s=140, marker="*", label="center")

        ax.set_xlabel(default_xlabel)
        ax.set_ylabel(default_ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if title is None:
            title = f"Latin hypercube {mode_label}"
        ax.set_title(title)

        ax.legend()
        plt.tight_layout()

        if savefig:
            plt.savefig(savefig)

        if showfig:
            plt.show()

        plt.close(fig)

        result = {
            "u_coords": u_coords,
            "v_coords": v_coords,
            "dist_to_plane": dist_to_plane,
            "mask": mask,
            "center": center,
            "center_plot": center_plot,
            "axis_u": u_hat,
            "axis_v": v_hat,
            "mode": mode,
        }

        self.settings.setdefault("results", {})["latin_hypercube_plot"] = result
        return result

    # ============================================================
    # Helper: NSigma around function
    # ============================================================

    def _select_around_function(self, points, fmean, fsigma_low, fsigma_high, fpoints=(lambda x: x[-1]), nsigma=1):
        # points[-1] is the output value, points[:-1] are the input parameters
        return (
            (fpoints(points) > (fmean(points) - nsigma * fsigma_low(points)))
            * (fpoints(points) < (fmean(points) + nsigma * fsigma_high(points)))
        )