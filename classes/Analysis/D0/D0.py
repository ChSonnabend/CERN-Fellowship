import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import json
import uuid
import datetime
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, cKDTree


class D0Analysis:
    def __init__(self, pdg_mass=1.86484, bkg_function="exp"):
        """
        Parameters
        ----------
        pdg_mass : float
            Nominal D0 mass in GeV/c².
        bkg_function : str
            Background model to use: 'exp' (default) or 'pol3'.
        """
        if bkg_function not in ("exp", "pol3"):
            raise ValueError("bkg_function must be either 'exp' or 'pol3'.")

        self.pdg_mass = pdg_mass
        self.bkg_function = bkg_function
        self.study = None
        self.best_trial = None
        self.df = None

    # -----------------------------------------------------------------
    # Utility: Gaussian + background models
    # -----------------------------------------------------------------
    @staticmethod
    def signal_plus_bkg_exp(x, mean, sigma, a, b, c):
        """Gaussian signal + exponential background"""
        x = np.asarray(x, dtype=np.float64)
        return a * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + b * np.exp(c * x)

    @staticmethod
    def signal_plus_bkg_pol3(x, mean, sigma, a, p0, p1, p2, p3):
        """Gaussian signal + 3rd order polynomial background"""
        x = np.asarray(x, dtype=np.float64)
        gauss = a * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
        return gauss + poly

    def _merge_plotargs(self, plotargs=None):
        """Merge user plotargs with default plotting style."""
        default_plotargs = {
            "fontsize": 20,
            "scatter_size": 50,
            "figsize": (16, 10),
            "colorbar": {"aspect": 40, "pad": 0.01},
            "legend_fontsize": 16,
            "legend_title": None
        }
        if plotargs is None:
            return default_plotargs
        merged = default_plotargs.copy()
        merged.update(plotargs)
        merged["colorbar"] = {
            **default_plotargs["colorbar"],
            **plotargs.get("colorbar", {}),
        }
        return merged

    # -----------------------------------------------------------------
    # Fit invariant mass spectrum
    # -----------------------------------------------------------------
    def fit_mass_spectrum(
        self, masses, bins=80, mass_window=None,
        plot=False, ax=None, label=None, savefig=False,
        plotargs=None, xlabel=None, bkg_function=None
    ):
        """
        Fit the D0 mass spectrum with a Gaussian + exponential or polynomial background.

        Parameters
        ----------
        masses : array-like
            Invariant mass values.
        bins : int
            Number of bins.
        mass_window : tuple
            Fit window (min, max).
        bkg_function : str
            'exp' (default) or 'pol3' for polynomial background.
        """
        if mass_window is None:
            mass_window = (0.9 * self.pdg_mass, 1.1 * self.pdg_mass)
        plotargs = self._merge_plotargs(plotargs)
        fontsize = plotargs["fontsize"]
        figsize = plotargs["figsize"]
        legend_fontsize = plotargs["legend_fontsize"]
        legend_title = plotargs["legend_title"]

        m = np.asarray(masses, dtype=np.float64)
        m = m[np.isfinite(m)]
        lo, hi = mass_window
        m = m[(m >= lo) & (m <= hi)]
        if m.size < 100:
            return {"S": 0.0, "B": 0.0, "sigma": 999.0, "params": None}

        counts, edges = np.histogram(m, bins=bins, range=mass_window)
        centers = 0.5 * (edges[1:] + edges[:-1])
        bin_width = edges[1] - edges[0]
        self.bin_width = bin_width
        counts_density = counts
        yerr = np.sqrt(np.maximum(counts, 1.0))

        # --- Initial guesses ---
        mean_guess = self.pdg_mass
        sigma_guess = 0.02
        a_guess = max(1.0, counts_density.max())

        bkg_function = bkg_function or self.bkg_function
        if bkg_function == "exp":
            # --- Exponential background ---
            b_guess = np.median(counts_density)
            c_guess = -1.0
            lower = [mean_guess * 0.99, 0.002, 0.0, 0.0, -50]
            upper = [mean_guess * 1.01, 0.1, np.inf, np.inf, 0]
            fit_func = self.signal_plus_bkg_exp
            p0 = [mean_guess, sigma_guess, a_guess, b_guess, c_guess]

        elif bkg_function == "pol3":
            # --- Polynomial background ---
            p0_guess = np.median(counts_density)
            p1_guess, p2_guess, p3_guess = 0.0, 0.0, 0.0
            lower = [mean_guess * 0.99, 0.002, 0.0, -np.inf, -np.inf, -np.inf, -np.inf]
            upper = [mean_guess * 1.01, 0.1, np.inf, np.inf, np.inf, np.inf, np.inf]
            fit_func = self.signal_plus_bkg_pol3
            p0 = [mean_guess, sigma_guess, a_guess, p0_guess, p1_guess, p2_guess, p3_guess]

        else:
            raise ValueError("Unsupported background function. Use 'exp' or 'pol3'.")

        try:
            popt, pcov = curve_fit(
                fit_func,
                centers, counts_density,
                sigma=yerr,
                p0=p0,
                bounds=(lower, upper),
                absolute_sigma=True,
                maxfev=20000,
            )

            mean, sigma = popt[0], popt[1]
            a = popt[2]
            # Integral (signal) remains in "counts" units, not density
            S = a * sigma * np.sqrt(2 * np.pi) * 0.9973002 / bin_width # 0.9973002 within ±3σ
            
            # --- Signal uncertainty propagation ---
            try:
                var_a = pcov[2, 2]
                var_sigma = pcov[1, 1]
                cov_a_sigma = pcov[2, 1]

                dS_da = sigma * np.sqrt(2 * np.pi)
                dS_dsigma = a * np.sqrt(2 * np.pi)
                S_err = np.sqrt(
                    (dS_da**2) * var_a +
                    (dS_dsigma**2) * var_sigma +
                    2 * dS_da * dS_dsigma * cov_a_sigma
                )
            except Exception:
                S_err = np.nan

            # --- Background integral estimation ---
            m_min, m_max = mean - 3 * sigma, mean + 3 * sigma
            if bkg_function == "exp":
                b, c = popt[3], popt[4]
                if abs(c) > 1e-12:
                    B = (b / c) * (np.exp(c * m_max) - np.exp(c * m_min)) / bin_width
                else:
                    B = b * (m_max - m_min) / bin_width
            else:
                p0b, p1b, p2b, p3b = popt[3:7]
                poly_integral = (
                    p0b * (m_max - m_min)
                    + 0.5 * p1b * (m_max**2 - m_min**2)
                    + (1 / 3) * p2b * (m_max**3 - m_min**3)
                    + 0.25 * p3b * (m_max**4 - m_min**4)
                )
                B = poly_integral / bin_width

            # --- Plotting ---
            if plot:
                if ax is None:
                    fig, ax = plt.subplots(figsize=figsize)
                ax.errorbar(
                    centers, counts_density,
                    xerr=bin_width / 2, yerr=yerr,
                    fmt="o", color="black", markersize=4,
                    label="Data" if label is None else label
                )

                x_fit = np.linspace(lo, hi, 500)
                ax.plot(
                    x_fit, fit_func(x_fit, *popt),
                    "r-", lw=2,
                    label=("Fit: ${0}\pm{1}$ GeV/$c^2$".format(round(mean, 4), round(sigma, 4)))
                )
                plt.scatter(self.pdg_mass,np.max(counts_density), alpha=0, color="white", label = "$S={0}$".format(round(S, 1))) #"$S={0}\pm{1}$".format(round(S, 1), round(S_err, 1)))
                plt.scatter(self.pdg_mass,np.max(counts_density), alpha=0, color="white",label = "$S/\sqrt{S+B}=$" + f"{S/np.sqrt(S+B):.2f}")

                ax.axvline(self.pdg_mass, color='blue', linestyle='--', linewidth=1,
                        label=f"PDG: {self.pdg_mass:.5f} GeV/c²")
                if xlabel is not None:
                    ax.set_xlabel(xlabel, fontsize=fontsize)
                else:
                    ax.set_xlabel(r"$m(D^0)$ [GeV/$c^2$]", fontsize=fontsize)
                ax.set_ylabel("Counts [#]", fontsize=fontsize)
                leg = ax.legend(title=legend_title, title_fontsize=legend_fontsize, fontsize=legend_fontsize)
                leg._legend_title_box._text.set_multialignment('center')
                ax.grid(alpha=0.6)
                plt.tight_layout()
                if savefig:
                    os.makedirs(os.path.dirname(savefig), exist_ok=True)
                    plt.savefig(savefig, bbox_inches="tight")
                plt.show()

            return {
                "S": float(S),
                "B": float(max(B, 0.0)),
                "significance": float(S / np.sqrt(S + B)) if B > 0 else 0,
                "params": popt,
            }

        except Exception as e:
            print(f"Fit failed ({bkg_function}): {e}")
            return {"S": 0.0, "B": 0.0, "sigma": 999.0, "params": None}

    # -----------------------------------------------------------------
    # Build combined DataFrame
    # -----------------------------------------------------------------
    @staticmethod
    def build_df_from_arrays(arrays):
        mapping = [
            ("fBdtScorePromptD0", "prompt", "D0"),
            ("fBdtScoreNonpromptD0", "nonprompt", "D0"),
            ("fBdtScoreBkgD0", "background", "D0"),
            ("fBdtScorePromptD0bar", "prompt", "D0bar"),
            ("fBdtScoreNonpromptD0bar", "nonprompt", "D0bar"),
            ("fBdtScoreBkgD0bar", "background", "D0bar"),
        ]

        frames = []
        for name, label, species in mapping:
            n = len(arrays[name])
            mass_key = "fInvMassD0" if species == "D0" else "fInvMassD0bar"
            df_part = pd.DataFrame({
                "bdt_score": arrays[name],
                "label": label,
                "species": species,
                "mass": arrays[mass_key],
                "cospa": arrays["fCosPa"][:n],
            })
            frames.append(df_part)
        return pd.concat(frames, ignore_index=True)

    # -----------------------------------------------------------------
    # Pareto compromise selection
    # -----------------------------------------------------------------
    def select_pareto_compromise(self, study, weights=None, verbose=True):
        """
        Selects the best compromise trial from a Pareto front using weighted optimization.

        Parameters
        ----------
        study : optuna.study.Study
            The Optuna multi-objective study.
        weights : dict
            Dictionary of weights that should contain keys "weight_sig" and "weight_yield".
        verbose : bool
            Whether to print summary info.

        Returns
        -------
        best_trial : optuna.trial.FrozenTrial
            The selected compromise trial.
        """
        pareto_trials = [t for t in study.best_trials if t.values is not None]
        if not pareto_trials:
            raise RuntimeError("No Pareto-optimal trials found in study.")

        best = max(
            pareto_trials,
            key=lambda t: (weights.get('weight_sig', 0.7) * t.values[0] * np.sqrt(1./(2.*self.bin_width)) + weights.get('weight_yield', 0.3) * np.log1p(t.values[1]) / self.bin_width)
        )

        if verbose:
            print(f"\n--- Selected Pareto Compromise ---")
            print(f"Weighted score = {weights.get('weight_sig', 0.7):.2f}*S/√(S+B) + {weights.get('weight_yield', 0.3):.2f}*log(1+S)")
            print(f"Significance: {best.values[0]:.3f}, Signal: {best.values[1]:.0f}")
            for k, v in best.params.items():
                print(f"{k:22s}: {v:.5f}")
        return best

    # -----------------------------------------------------------------
    # Multi-objective optimization (adds stable study name)
    # -----------------------------------------------------------------
    def optimize_moo(self, df, mass_window=(1.80, 1.92),
                    n_trials=250, bins=80,
                    min_selected=200,
                    cospa_quantiles=(0.50, 0.995),
                    bdt_quantiles=(0.05, 0.95),
                    weights={
                        "weight_sig": 0.3,
                        "weight_yield": 0.7
                    },
                    plot_best=False):
        """
        Run multi-objective Optuna optimization and automatically select
        the best compromise trial using Pareto weighting.
        """
        self.df = df
        self.mass_window = mass_window
        self.weights = weights

        # Give the study a stable unique name
        study_name = f"D0Study_{uuid.uuid4().hex[:8]}"
        print(f"[INFO] Creating new Optuna study: {study_name}")

        ql, qh = bdt_quantiles
        ranges = {}
        for species in ["D0", "D0bar"]:
            sub = df.loc[df["species"] == species, "bdt_score"].astype(float)
            lo, hi = float(sub.quantile(ql)), float(sub.quantile(qh))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(sub.min()), float(sub.max())
            ranges[species] = (lo, hi)

        cl, ch = cospa_quantiles
        cos_lo, cos_hi = float(df["cospa"].quantile(cl)), float(df["cospa"].quantile(ch))
        if not np.isfinite(cos_lo) or not np.isfinite(cos_hi) or cos_lo >= cos_hi:
            cos_lo, cos_hi = float(df["cospa"].min()), float(df["cospa"].max())

        def objective(trial):
            cuts = {}
            for label in ["prompt", "nonprompt", "background"]:
                for species in ["D0", "D0bar"]:
                    lo, hi = ranges[species]
                    name = f"cut_{label}_{species}"
                    cuts[name] = trial.suggest_float(name, lo, hi)
            cospa_cut = trial.suggest_float("cospa_cut", cos_lo, cos_hi)

            selected = []
            for label in ["prompt", "nonprompt", "background"]:
                for species in ["D0", "D0bar"]:
                    key = f"cut_{label}_{species}"
                    sel = df[
                        (df["species"] == species)
                        & (df["label"] == label)
                        & (df["bdt_score"] > cuts[key])
                        & (df["cospa"] > cospa_cut)
                        & (df["mass"].between(*mass_window))
                    ]
                    selected.append(sel)
            sel_all = pd.concat(selected, ignore_index=True)

            if len(sel_all) < min_selected:
                raise optuna.TrialPruned()

            res = self.fit_mass_spectrum(sel_all["mass"].values,
                                        bins=bins, mass_window=mass_window,
                                        bkg_function=self.bkg_function)
            S, B = res["S"], res["B"]

            if S + B <= 0 or not np.isfinite(S) or not np.isfinite(B):
                raise optuna.TrialPruned()

            significance = S / np.sqrt(S + B)
            return significance, S

        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize", "maximize"]
        )
        study.optimize(objective, n_trials=n_trials)
        self.study = study

        # Select best trial using Pareto weighting
        self.best_trial = self.select_pareto_compromise(study, weights=self.weights)

        print(f"\n✅ Optimization completed: {len(study.best_trials)} Pareto-optimal points found.")
        if plot_best:
            self.plot_best_fit(df, self.best_trial.params, mass_window, bins)
            self.plot_pareto_front_many(study, self.best_trial)

        return study, self.best_trial

    # -----------------------------------------------------------------
    # Plot best fit from chosen trial
    # -----------------------------------------------------------------
    def plot_best_fit(self, df, params, mass_window, bins, savefig=False):
        parts = []
        for label in ["prompt", "nonprompt", "background"]:
            for species in ["D0", "D0bar"]:
                key = f"cut_{label}_{species}"
                mask = (
                    (df["species"] == species)
                    & (df["label"] == label)
                    & (df["bdt_score"] > params[key])
                    & (df["cospa"] > params["cospa_cut"])
                    & (df["mass"].between(*mass_window))
                )
                parts.append(df[mask])
        sel_all = pd.concat(parts, ignore_index=True)
        self.fit_mass_spectrum(sel_all["mass"].values,
                               bins=bins, mass_window=mass_window,
                               bkg_function=self.bkg_function,
                               plot=True, savefig=savefig)

    def plot_pareto_front(self, study=None, plotargs=None, savefig=False):
        plotargs = self._merge_plotargs(plotargs)
        fontsize = plotargs["fontsize"]
        figsize = plotargs["figsize"]
        legend_fontsize = plotargs["legend_fontsize"]

        if self.study is None and study is None:
            raise RuntimeError("No study found. Run optimize_moo() first or pass a study.")

        trials = (study or self.study).trials
        values = np.array([t.values for t in trials if t.values is not None])
        pareto_trials = [t for t in trials if t in (study or self.study).best_trials]
        pareto_values = np.array([t.values for t in pareto_trials])

        plt.figure(figsize=figsize)
        plt.scatter(values[:, 0], values[:, 1],
                    c='blue', alpha=0.5, s=plotargs["scatter_size"], label='All Trials')
        plt.scatter(pareto_values[:, 0], pareto_values[:, 1],
                    c='red', s=plotargs["scatter_size"] * 1.6, label='Pareto Front')
        plt.xlabel('Significance (S/√(S+B))', fontsize=fontsize)
        plt.ylabel('Signal Yield (S)', fontsize=fontsize)
        # plt.title('Pareto Front of Multi-Objective Optimization', fontsize=fontsize)
        leg = plt.legend(fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        leg._legend_title_box._text.set_multialignment('center')
        plt.grid(alpha=0.6)
        plt.tight_layout()
        if savefig:
            os.makedirs(os.path.dirname(savefig), exist_ok=True)
            plt.savefig(savefig, bbox_inches="tight")
        plt.show()

    def plot_pareto_front_many(self, study, best, cmap=plt.cm.plasma, colormode='distance', scale_SB = 1.0, plotargs=None, savefig=False):
        """
        Plot all trials in objective space (S, significance) with a color scale
        showing distance to the Pareto front.

        Parameters
        ----------
        study : optuna.study.Study
            The Optuna multi-objective study.
        best : optuna.trial.FrozenTrial
            The selected best (compromise) trial.
        cmap : matplotlib colormap, optional
            Colormap used for distances (default: plasma).
        plotargs : dict, optional
            Dictionary with figure and style parameters. Defaults to:
            {
                "fontsize": 20,
                "scatter_size": 50,
                "figsize": (16, 10),
                "colorbar": {"aspect": 40, "pad": 0.01}
            }
        """

        plotargs = self._merge_plotargs(plotargs)
        fontsize = plotargs["fontsize"]
        ssize = plotargs["scatter_size"]
        figsize = plotargs["figsize"]
        legend_fontsize = plotargs["legend_fontsize"]

        # --- Gather all trials with valid results ---
        trials = [t for t in study.trials if t.values is not None]
        sigs = np.array([t.values[0] for t in trials]) * np.sqrt(1./(2.*scale_SB))
        yields = np.array([t.values[1] for t in trials]) / scale_SB

        # --- Pareto front (true non-dominated) ---
        pareto = [t for t in study.best_trials]
        pareto_sigs = np.array([t.values[0] for t in pareto]) * np.sqrt(1./(2.*scale_SB))
        pareto_yields = np.array([t.values[1] for t in pareto]) / scale_SB

        # --- Compute distance to Pareto front (normalized Euclidean distance) ---
        sigs_norm = (sigs - sigs.min()) / (sigs.max() - sigs.min() + 1e-9)
        yields_norm = (yields - yields.min()) / (yields.max() - yields.min() + 1e-9)
        pareto_sigs_norm = (pareto_sigs - sigs.min()) / (sigs.max() - sigs.min() + 1e-9)
        pareto_yields_norm = (pareto_yields - yields.min()) / (yields.max() - yields.min() + 1e-9)

        pareto_points = np.column_stack((pareto_yields_norm, pareto_sigs_norm))
        trial_points = np.column_stack((yields_norm, sigs_norm))

        tree = cKDTree(pareto_points)
        distances, _ = tree.query(trial_points)
        normed_dist = distances / distances.max() if distances.max() > 0 else distances

        # --- Convex hull (optional envelope for aesthetics) ---
        try:
            hull = ConvexHull(np.column_stack((yields, sigs)))
            hull_points = np.column_stack((yields[hull.vertices], sigs[hull.vertices]))
        except Exception:
            hull_points = np.column_stack((pareto_yields, pareto_sigs))
            
        color_vals = normed_dist
        legend_label = "Normalized distance to Pareto front"
        if colormode == "iterations":
            color_vals = np.array([t.number for t in trials])
            legend_label = "Trial number"

        # --- Pareto front & chosen compromise ---
        
        # --- Plot setup ---
        plt.figure(figsize=plotargs["figsize"])
        # plt.scatter(pareto_yields, pareto_sigs, c="black", s=ssize * 1.6, label="Pareto front", zorder=3)
        plt.scatter(
            best.values[1] / scale_SB,
            best.values[0] * np.sqrt(1./(2.*scale_SB)),
            facecolors='none',      # transparent fill
            edgecolors='red',       # red border
            s=ssize * 3,
            linewidth=2.0,
            label="Selected compromise",
            zorder=5,
        )
        sc = plt.scatter(
            yields,
            sigs,
            c=color_vals,
            cmap=cmap,
            s=ssize,
            edgecolors="none",
            alpha=0.85,
            label="All trials",
        )

        # --- Optional hull outline ---
        # plt.plot(hull_points[:, 0], hull_points[:, 1], "--", color="gray", lw=1, alpha=0.6)

        # --- Colorbar ---
        cbar = plt.colorbar(sc, **plotargs["colorbar"])
        cbar.set_label(legend_label, fontsize=fontsize)

        # --- Axes labels & style ---
        plt.xlabel("Signal yield S", fontsize=fontsize)
        plt.ylabel("Significance S/$\sqrt{(S+B)}$", fontsize=fontsize)
        # plt.title("Pareto front colored by proximity to optimal region", fontsize=fontsize)
        plt.grid(alpha=0.6)
        leg = plt.legend(title_fontsize=legend_fontsize, fontsize=legend_fontsize)
        leg._legend_title_box._text.set_multialignment('center')
        plt.tight_layout()
        if savefig:
            os.makedirs(os.path.dirname(savefig), exist_ok=True)
            plt.savefig(savefig, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------
    # Apply best Optuna cuts to the DataFrame
    # -----------------------------------------------------------------
    def apply_cuts(self, df=None):
        if self.study is None:
            raise RuntimeError("No study found. Run optimize_moo() first.")
        if df is None:
            df = self.df
        params = self.best_trial.params
        mask_total = np.ones(len(df), dtype=bool)
        mask_total &= df["cospa"] > params["cospa_cut"]

        for label in ["prompt", "nonprompt", "background"]:
            for species in ["D0", "D0bar"]:
                key = f"cut_{label}_{species}"
                mask = (
                    (df["species"] == species)
                    & (df["label"] == label)
                    & (df["bdt_score"] > params[key])
                )
                mask_total &= ~((df["species"] == species) & (df["label"] == label)) | mask
        return pd.Series(mask_total, index=df.index)

    # -----------------------------------------------------------------
    # Save study with extended metadata (append mode)
    # -----------------------------------------------------------------
    def save_study(
        self,
        filepath="optuna_study.db",
        storage_backend="sqlite",
        objective_name="S/sqrt(S+B)",
    ):
        """
        Save Optuna study to disk (SQLite or JSON) with full metadata.
        Appends new metadata entries if .meta.json already exists.
        """
        if self.study is None:
            raise RuntimeError("No study to save. Run optimize_moo() first.")

        study_name = self.study.study_name or "UnnamedStudy"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Determine final weights ---
        weights = self.weights

        # === SQLite backend ======================================================
        if storage_backend == "sqlite":
            storage_url = f"sqlite:///{filepath}" if not filepath.startswith("sqlite:///") else filepath
            storage = optuna.storages.RDBStorage(storage_url)
            existing = [s.study_name for s in optuna.get_all_study_summaries(storage)]

            if study_name in existing:
                print(f"⚠️ Study name '{study_name}' already exists in {filepath}. Creating a new version.")
                study_name += "_v" + datetime.now().strftime("%Y%m%d_%H%M%S")

            optuna.copy_study(
                from_study_name=self.study.study_name,
                from_storage=self.study._storage,
                to_study_name=study_name,
                to_storage=storage_url,
            )

            # --- Metadata entry ---
            meta_entry = {
                "timestamp": timestamp,
                "study_name": study_name,
                "mass_window": self.mass_window,
                "objective": objective_name,
                "weights": self.weights,
                "bkg_function": self.bkg_function,
                "best_trial": {
                    "values": self.best_trial.values if self.best_trial else None,
                    "params": self.best_trial.params if self.best_trial else None,
                },
                "n_trials": len(self.study.trials),
                "backend": "sqlite",
            }

            # --- Append metadata log ---
            meta_path = filepath + ".meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):  # convert old single-entry file
                        data = [data]
                except Exception:
                    data = []
            else:
                data = []

            data.append(meta_entry)
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"✅ Study '{study_name}' saved to SQLite file: {filepath}")
            print(f"🗂  Metadata appended to: {meta_path}")

        # === JSON (journal) backend =============================================
        elif storage_backend == "journal":
            data = {
                "timestamp": timestamp,
                "study_name": study_name,
                "mass_window": mass_window,
                "objective": objective_name,
                "weights": weights,
                "best_trial": {
                    "values": self.best_trial.values if self.best_trial else None,
                    "params": self.best_trial.params if self.best_trial else None,
                },
                "all_trials": [
                    {
                        "number": t.number,
                        "values": t.values,
                        "params": t.params,
                        "state": str(t.state),
                    }
                    for t in self.study.trials
                ],
                "n_trials": len(self.study.trials),
                "backend": "journal",
            }

            # --- Append to JSON array ---
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    try:
                        prev_data = json.load(f)
                        if isinstance(prev_data, dict):
                            prev_data = [prev_data]
                    except Exception:
                        prev_data = []
            else:
                prev_data = []

            prev_data.append(data)
            with open(filepath, "w") as f:
                json.dump(prev_data, f, indent=2)

            print(f"✅ Study appended to JSON log: {filepath}")

        else:
            raise ValueError(f"Unsupported backend: {storage_backend}")


    # -----------------------------------------------------------------
    # Load study (auto-select latest or specified study)
    # -----------------------------------------------------------------
    def load_study(
        self,
        filepath="optuna_study.db",
        storage_backend="sqlite",
        study_name=None,
        weights=None
    ):
        """
        Load an existing Optuna study from disk and restore metadata.

        If available, automatically reuses stored weighting factors.
        """
        # === SQLite BACKEND =====================================================
        if storage_backend == "sqlite":
            meta_path = filepath + ".meta.json"

            # --- Step 1: select study name ---
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"No metadata file found at {meta_path}. Cannot auto-detect study."
                )

            with open(meta_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

            if study_name is None:
                latest_entry = max(
                    data,
                    key=lambda d: datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S"),
                )
                study_name = latest_entry["study_name"]
                print(f"🕒 Loaded latest study entry from metadata: '{study_name}'")
            else:
                latest_entry = next((e for e in data if e["study_name"] == study_name), None)
                if latest_entry is None:
                    raise ValueError(f"Study '{study_name}' not found in metadata.")

            # --- Restore stored weights if available ---
            if "weights" in latest_entry and isinstance(latest_entry["weights"], dict):
                self.weights = latest_entry["weights"]
                print(f"Loaded weights from metadata:", self.weights)

            # --- Step 2: load the study ---
            storage_url = f"sqlite:///{filepath}" if not filepath.startswith("sqlite:///") else filepath
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_url)
            except Exception as e:
                raise RuntimeError(f"Failed to load study '{study_name}': {e}")

            self.study = study

            # --- Step 3: re-identify best Pareto compromise ---
            self.best_trial = self.select_pareto_compromise(self.study, weights=self.weights)
            self.bkg_function = latest_entry.get("bkg_function", "exp")

            print(f"✅ Best trial re-evaluated using Pareto weighting: ", self.weights)
            print(f"✅ Loaded study '{study_name}' from SQLite file: {filepath}")
            return study

        # === JOURNAL BACKEND ====================================================
        elif storage_backend == "journal":
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Journal file not found: {filepath}")

            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

            # --- Step 1: pick correct study entry ---
            if study_name is None:
                entry = max(
                    data,
                    key=lambda d: datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S"),
                )
                print(f"🕒 Loaded latest JSON study entry: '{entry['study_name']}'")
            else:
                entry = next((e for e in data if e["study_name"] == study_name), None)
                if entry is None:
                    raise ValueError(f"Study '{study_name}' not found in {filepath}.")

            # --- Step 2: restore weights ---
            weights = entry.get("weights", {})
            print(f"⚖️  Loaded weights from metadata:", weights)

            # --- Step 3: reconstruct study ---
            study = optuna.create_study(directions=["maximize", "maximize"])
            for t in entry.get("all_trials", []):
                trial = optuna.trial.create_trial(params=t["params"], values=t["values"])
                study.add_trial(trial)

            self.study = study
            self.best_trial = self.select_pareto_compromise(
                self.study, weights=self.weights
            )

            print(f"✅ Loaded study '{entry['study_name']}' from JSON file: {filepath}")
            return study

        else:
            raise ValueError(f"Unsupported backend: {storage_backend}")