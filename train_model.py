# -*- coding: utf-8 -*-
"""
Aligned train_model.py - Q-PEAK model architecture and training script
Implements: Ridge baseline + QGWO-gated ExtraTrees/LightGBM/Huber residual ensemble
            + PhysicsKAN residual refiner + post-calibration + Causal PIMS
"""

import os, sys, math, json, time, random, shutil, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

warnings.filterwarnings("ignore")

def _ensure(pkg: str):
    try:
        __import__(pkg)
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _pkg in ["xgboost", "lightgbm", "tabulate"]:
    _ensure(_pkg)

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, HuberRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

TORCH_OK = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    TORCH_OK = False

SHAP_OK = True
try:
    import shap
except Exception:
    SHAP_OK = False

DEVICE = torch.device("cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu") if TORCH_OK else None
XGB_DEVICE = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"

# =============================================================================
# 1. Config
# =============================================================================
MODEL_VARIANT = "QPEAK_Final_ET_LGBM_Huber_PhysicsKAN_CausalPIMS"
USE_LEGACY_RESIDUAL_BANK = False
USE_SINGLE_RESIDUAL_BRANCH = False
USE_QGWO_ENSEMBLE_BRANCHES = True
ENSEMBLE_BRANCH_KINDS = ["ExtraTrees", "LightGBM", "HuberRegressor"]
SINGLE_RESIDUAL_BRANCH_KIND = "QGWO_ET_LGBM_Huber_Ensemble"
SINGLE_RESIDUAL_BRANCH_DESCRIPTION = "Ridge baseline + QGWO-gated ExtraTrees/LightGBM/Huber residual ensemble + PhysicsKAN residual refiner + train-only post calibration + CausalPIMS"

DATASET_CANDIDATES = [
    "Battery_RUL.csv",
    "/kaggle/input/datasets/ignaciovinuales/battery-remaining-useful-life-rul/Battery_RUL.csv",
    "/kaggle/input/battery-remaining-useful-life-rul/Battery_RUL.csv",
    "/kaggle/input/battery-rul/Battery_RUL.csv",
    "/mnt/data/Battery_RUL.csv",
]

OUTPUT_DIR = "output"
ZIP_PATH = "qpeak_results.zip"

USE_CAUSAL_TAU_IN_MAIN = True
ENABLE_QGWO_5SEED = True
ENABLE_QGWO_LOBO = False
ENABLE_HARD_BATTERY_LOCAL_REFINE = False

QGWO_OBJECTIVE_TRACK = "full_causal_pims"

SAVE_PREDICTIONS = True
SAVE_QGWO_HISTORY = False

FIVE_SEEDS = [42, 43, 44, 45, 46]
HARD_BATTERY_IDS = [4, 5, 12, 13]

GLOBAL_QGWO_WOLVES = 6
GLOBAL_QGWO_ITERS = 5
QGWO_BATTERY_STD_LAMBDA = 0.15
QGWO_BATTERY_MAX_LAMBDA = 0.05

CLEAN_CLIP_COLS = [
    "Discharge Time (s)",
    "Time at 4.15V (s)",
    "Charging time (s)",
    "Time constant current (s)",
]

CLOCK_BASELINE_FEATURES = [
    "Cycle_Index",
    "Cycle_log",
    "Cum_Ah",
    "Cum_Energy",
    "Cum_Ah_log1p",
    "Cum_Energy_log1p",
]

CLOCK_LIKE_KEYS = ["Cycle", "Cum_Ah", "Cum_Energy"]

FIG_DPI = 300
THESIS_TITLE_FONTSIZE = 14
THESIS_AXIS_LABEL_FONTSIZE = 12
THESIS_TICK_FONTSIZE = 10
THESIS_LEGEND_FONTSIZE = 10
THESIS_ANNOT_FONTSIZE = 9

KEEP_FIGURE_FILES = {
    "fig_engineered_feature_correlation_heatmap.png",
    "fig_group_split_5seed_rmse_summary.png",
    "fig_stage_ablation_rmse.png",
    "fig_group_split_5seed_stage_train_test_rmse.png",
    "fig_pims_diagnostics_summary.png",
    "fig_lobo_rmse_by_battery.png",
    "fig_lobo_rul_prediction_trajectories.png",
    "fig_lobo_residual_histogram.png",
    "fig_lobo_residual_vs_predicted.png",
    "fig_qgwo_gate_weights_5seeds.png",
}

# =============================================================================
# 2. Utilities
# =============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def find_dataset_path() -> str:
    for p in DATASET_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Battery_RUL.csv was not found. Please edit DATASET_CANDIDATES.")

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def safe_to_csv(df: pd.DataFrame, path: str, index: bool = False):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index, encoding="utf-8-sig")

def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _apply_axis_text(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, fontsize=THESIS_TITLE_FONTSIZE, pad=10, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=THESIS_AXIS_LABEL_FONTSIZE, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=THESIS_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=THESIS_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=THESIS_TICK_FONTSIZE)

def calc_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def init_metric_dict():
    return {"RMSE": [], "MAE": [], "MSE": [], "R2": []}

def append_metrics(store: Dict[str, List[float]], metrics: Dict[str, float]):
    for k in store:
        store[k].append(metrics[k])

def summarize_metric_dict(metric_dict: Dict[str, List[float]], prefix: str = "") -> Dict[str, float]:
    row = {}
    for k, vals in metric_dict.items():
        vals = np.asarray(vals, dtype=float)
        row[f"{prefix}{k}_Mean"] = float(np.mean(vals))
        row[f"{prefix}{k}_Std"] = float(np.std(vals))
        row[f"{prefix}{k}_Median"] = float(np.median(vals))
        row[f"{prefix}{k}_Min"] = float(np.min(vals))
        row[f"{prefix}{k}_Max"] = float(np.max(vals))
    return row

def monotonic_violation_count(df_subset: pd.DataFrame, preds: np.ndarray) -> int:
    preds = np.asarray(preds, dtype=float).reshape(-1)
    total = 0
    for b in np.unique(df_subset["Battery_ID"].values):
        idx = np.where(df_subset["Battery_ID"].values == b)[0]
        order = idx[np.argsort(df_subset.loc[idx, "Cycle_Index"].values.astype(float))]
        p = preds[order]
        total += int(np.sum(np.diff(p) > 1e-9))
    return total

# =============================================================================
# 3. Split-safe cleaning and feature engineering
# =============================================================================
def infer_battery_ids_from_cycle(cycle_vals: np.ndarray) -> np.ndarray:
    cycle_vals = np.asarray(cycle_vals, dtype=float)
    return np.r_[0, (cycle_vals[1:] <= cycle_vals[:-1]).astype(int)].cumsum()

def attach_battery_id(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Cycle_Index" not in df.columns:
        raise ValueError("Cycle_Index column is required to infer Battery_ID.")
    c = pd.to_numeric(df["Cycle_Index"], errors="coerce").fillna(0).values
    df["Battery_ID"] = infer_battery_ids_from_cycle(c)
    return df

def fit_cleaning_params(df_train_raw: pd.DataFrame) -> Dict[str, Any]:
    params = {"clip_hi": {}}
    for c in CLEAN_CLIP_COLS:
        if c in df_train_raw.columns:
            vals = pd.to_numeric(df_train_raw[c], errors="coerce")
            # 使用 quantile(0.98) * 1.1 來徹底、穩健地濾除 99% 以上的極端大離群值
            params["clip_hi"][c] = float(vals.quantile(0.98) * 1.1)
    return params

def apply_cleaning(df_raw: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df = df_raw.copy()
    if "RUL" in df.columns:
        df = df.dropna(subset=["RUL"])
    for c in CLEAN_CLIP_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            hi = None
            if params is not None:
                hi = params.get("clip_hi", {}).get(c, None)
            if hi is None or not np.isfinite(hi):
                hi = float(df[c].quantile(0.999))
            df[c] = df[c].clip(lower=0.0, upper=hi)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def ema_group(series: pd.Series, alpha: float = 0.1) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

def robust_mad_scale(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-6
    return float(1.4826 * mad)

def causal_tau_group(H_series: pd.Series, warmup: int = 10) -> np.ndarray:
    x = H_series.values.astype(float)
    n = len(x)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    init_ref = float(np.mean(x[:min(warmup, n)]))
    for i in range(n):
        past = x[:i + 1]
        scale = robust_mad_scale(init_ref - past)
        scale = float(np.clip(scale, 1e-3, None))
        out[i] = 1.0 / (1.0 + np.exp(-(init_ref - x[i]) / scale))
    return np.clip(out, 0.0, 1.0)

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()
    if "Battery_ID" not in df.columns:
        df = attach_battery_id(df)
    df = df.sort_values(["Battery_ID", "Cycle_Index"]).reset_index(drop=True)

    df["Cap"] = df.get("Discharge Time (s)", 0.0).astype(float)
    df["IR"] = (4.2 - df.get("Max. Voltage Dischar. (V)", 4.2)).astype(float)
    df["CV_Ratio"] = (df.get("Time at 4.15V (s)", 0.0) / (df.get("Charging time (s)", 1.0) + 1e-6)).astype(float)
    df["CC_Time"] = df.get("Time constant current (s)", 0.0).astype(float)
    df["Vdrop"] = df.get("Decrement 3.6-3.4V (s)", 0.0).astype(float)

    df["Energy"] = (df["Cap"] * 3.6).clip(lower=0)
    df["Cum_Ah"] = df.groupby("Battery_ID")["Cap"].cumsum()
    df["Cum_Energy"] = df.groupby("Battery_ID")["Energy"].cumsum()

    for col in ["Cap", "IR", "CV_Ratio", "CC_Time"]:
        init = df.groupby("Battery_ID")[col].transform("first")
        df[f"{col}_Ret"] = df[col] / (init + 1e-6)
        df[f"{col}_EMA"] = df.groupby("Battery_ID")[f"{col}_Ret"].transform(lambda x: ema_group(x, 0.1))
        df[f"{col}_v1"] = df.groupby("Battery_ID")[f"{col}_EMA"].diff(10).fillna(0.0)

    df["Vdrop_EMA"] = df.groupby("Battery_ID")["Vdrop"].transform(lambda x: ema_group(x, 0.1))
    df["Vdrop_v1"] = df.groupby("Battery_ID")["Vdrop_EMA"].diff(10).fillna(0.0)
    df["CC_Ratio_EMA"] = df["CC_Time_EMA"] / (df["Cap_EMA"] + 1e-6)
    df["CC_minus_CV_EMA"] = df["CC_Time_EMA"] - df["CV_Ratio_EMA"]

    df["H_proxy"] = 1.0 * df["Cap_EMA"] + 0.35 * (1.0 / (df["IR_EMA"] + 1e-6)) - 0.25 * df["CV_Ratio_EMA"]
    df["Tau"] = df.groupby("Battery_ID")["H_proxy"].transform(lambda s: pd.Series(causal_tau_group(s), index=s.index))

    df["Cycle_log"] = np.log1p(df["Cycle_Index"].astype(float))
    df["Cum_Ah_log1p"] = np.log1p(df["Cum_Ah"].clip(lower=0))
    df["Cum_Energy_log1p"] = np.log1p(df["Cum_Energy"].clip(lower=0))

    for col in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "Tau", "CC_Ratio_EMA", "CC_minus_CV_EMA"]:
        g = df.groupby("Battery_ID")[col]
        df[f"{col}_rm20"] = g.transform(lambda x: x.rolling(20, min_periods=1).mean())
        df[f"{col}_rs20"] = g.transform(lambda x: x.rolling(20, min_periods=1).std()).fillna(0.0)
        df[f"{col}_slope20"] = g.transform(lambda x: ((x - x.shift(20)).fillna(0.0)) / 20.0)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def get_feature_sets(df_all: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    clock_feats = [c for c in CLOCK_BASELINE_FEATURES if c in df_all.columns]

    full_feats = [
        c for c in [
            "Discharge Time (s)", "Decrement 3.6-3.4V (s)", "Max. Voltage Dischar. (V)", "Min. Voltage Charg. (V)",
            "Time at 4.15V (s)", "Time constant current (s)", "Charging time (s)",
            "Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "CC_Time_EMA", "Vdrop_EMA", "CC_Ratio_EMA", "CC_minus_CV_EMA",
            "Cap_v1", "IR_v1", "CV_Ratio_v1", "Vdrop_v1",
            "Cap_EMA_rm20", "IR_EMA_rm20", "CV_Ratio_EMA_rm20", "CC_Ratio_EMA_rm20", "CC_minus_CV_EMA_rm20",
            "Cap_EMA_rs20", "IR_EMA_rs20", "CV_Ratio_EMA_rs20", "CC_Ratio_EMA_rs20",
            "Cap_EMA_slope20", "IR_EMA_slope20", "CV_Ratio_EMA_slope20",
            "Tau", "Tau_rm20", "Tau_rs20", "Tau_slope20",
            "Cum_Ah_log1p", "Cum_Energy_log1p", "Cycle_log", "Cycle_Index",
        ] if c in df_all.columns
    ]

    kan_feats = [
        c for c in [
            "Tau", "Tau_rm20", "Tau_slope20",
            "Cap_v1", "IR_v1", "CV_Ratio_v1", "Vdrop_v1",
            "CC_Ratio_EMA", "CC_minus_CV_EMA"
        ] if c in df_all.columns
    ]

    no_clock_feats = [f for f in full_feats if not any(k in f for k in CLOCK_LIKE_KEYS)]
    clock_only_feats = [f for f in full_feats if any(k in f for k in CLOCK_LIKE_KEYS)]
    return clock_feats, full_feats, kan_feats, no_clock_feats, clock_only_feats

def prepare_fold_features(raw_with_id: pd.DataFrame, train_batteries: List[int], test_batteries: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = raw_with_id[raw_with_id["Battery_ID"].isin(train_batteries)].copy()
    test_raw = raw_with_id[raw_with_id["Battery_ID"].isin(test_batteries)].copy()
    clean_params = fit_cleaning_params(train_raw)
    dtr = engineer_features(apply_cleaning(train_raw, clean_params))
    dte = engineer_features(apply_cleaning(test_raw, clean_params))
    return dtr.reset_index(drop=True), dte.reset_index(drop=True)

# =============================================================================
# 4. QGWO
# =============================================================================
class LocalQGWO:
    def __init__(self, obj_func, bounds, num_wolves=6, max_iter=5):
        self.obj_func = obj_func
        self.bounds = np.array(bounds, dtype=float)
        self.num_wolves = int(num_wolves)
        self.max_iter = int(max_iter)
        self.dim = len(bounds)
        self.X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_wolves, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = np.inf
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = np.inf
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = np.inf
        self.history = []

    def optimize(self):
        for t in range(self.max_iter):
            iter_scores = []
            for i in range(self.num_wolves):
                self.X[i] = np.clip(self.X[i], self.bounds[:, 0], self.bounds[:, 1])
                score = float(self.obj_func(self.X[i]))
                iter_scores.append(score)

                if score < self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = score, self.X[i].copy()
                elif score < self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = score, self.X[i].copy()
                elif score < self.delta_score:
                    self.delta_score, self.delta_pos = score, self.X[i].copy()

            self.history.append({
                "iter": int(t),
                "best_score": float(self.alpha_score),
                "iter_mean_score": float(np.mean(iter_scores)),
                "iter_std_score": float(np.std(iter_scores)),
            })

            a = 0.5 + 0.5 * math.cos(math.pi * t / (2 * max(1, self.max_iter)))
            C = np.mean(self.X, axis=0)

            for i in range(self.num_wolves):
                u1, u2, u3 = np.random.rand(3)
                s1 = np.where(np.random.rand(self.dim) > 0.5, 1, -1)
                s2 = np.where(np.random.rand(self.dim) > 0.5, 1, -1)
                s3 = np.where(np.random.rand(self.dim) > 0.5, 1, -1)
                X1 = self.alpha_pos + s1 * a * np.abs(C - self.X[i]) * np.log(1.0 / (u1 + 1e-9))
                X2 = self.beta_pos + s2 * a * np.abs(C - self.X[i]) * np.log(1.0 / (u2 + 1e-9))
                X3 = self.delta_pos + s3 * a * np.abs(C - self.X[i]) * np.log(1.0 / (u3 + 1e-9))
                self.X[i] = (X1 + X2 + X3) / 3.0

        return self.alpha_pos, self.alpha_score

    def history_df(self):
        return pd.DataFrame(self.history)

# =============================================================================
# 5. KAN
# =============================================================================
if TORCH_OK:
    class KANLinear(nn.Module):
        def __init__(self, in_features, out_features, grid_size=3, spline_order=2):
            super().__init__()
            self.spline_order = spline_order
            h = 2.0 / grid_size
            grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h - 1.0).expand(in_features, -1).contiguous()
            self.register_buffer("grid", grid)
            self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            nn.init.constant_(self.spline_scaler, 1.0)
            nn.init.normal_(self.spline_weight, std=0.02)

        def b_splines(self, x):
            x = x.unsqueeze(-1)
            bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                d1 = self.grid[:, k:-1] - self.grid[:, :-(k + 1)] + 1e-6
                d2 = self.grid[:, k + 1:] - self.grid[:, 1:-k] + 1e-6
                bases = (x - self.grid[:, :-(k + 1)]) / d1 * bases[:, :, :-1] + (self.grid[:, k + 1:] - x) / d2 * bases[:, :, 1:]
            return bases.contiguous()

        def forward(self, x):
            base_out = torch.nn.functional.linear(torch.nn.functional.silu(x), self.base_weight)
            scaled_w = (self.spline_weight * self.spline_scaler.unsqueeze(-1)).view(self.base_weight.shape[0], -1)
            return base_out + torch.nn.functional.linear(self.b_splines(x).view(x.size(0), -1), scaled_w)

    class PhysicsKANRefiner(nn.Module):
        def __init__(self, in_dim, hidden_dim=4):
            super().__init__()
            self.kan1 = KANLinear(in_dim, hidden_dim)
            self.kan2 = KANLinear(hidden_dim, 1)

        def forward(self, x, amp):
            return torch.tanh(self.kan2(self.kan1(x)).squeeze(-1)) * amp

def train_physics_kan_refiner(X_tr, r_tr, X_te, amp_tr, amp_te, epochs=25, lr=0.015, seed=42):
    if not TORCH_OK or X_tr.shape[1] == 0:
        return None, np.zeros(len(X_tr)), np.zeros(len(X_te))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PhysicsKANRefiner(X_tr.shape[1], hidden_dim=4).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    crit = nn.HuberLoss(delta=1.0)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(r_tr, dtype=torch.float32, device=DEVICE)
    at = torch.tensor(amp_tr, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    ate = torch.tensor(amp_te, dtype=torch.float32, device=DEVICE)

    dl = DataLoader(TensorDataset(Xt, yt, at), batch_size=1024, shuffle=True)
    model.train()

    for _ in range(epochs):
        for xb, yb, ab in dl:
            opt.zero_grad()
            loss = crit(model(xb, ab), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        return model, model(Xt, at).detach().cpu().numpy(), model(Xte, ate).detach().cpu().numpy()

# =============================================================================
# 6. Model pipeline
# =============================================================================
@dataclass
class Params:
    base_alpha: float = 15.0
    lgbm_max_depth: int = 5
    lgbm_learning_rate: float = 0.009
    lgbm_reg_lambda: float = 22.0
    et_depth: int = 8
    et_leaf: int = 9
    lambda_ET: float = 0.12
    lambda_LGBM: float = 0.10
    lambda_Huber: float = 0.05
    lambda_KAN: float = 0.9

def apply_offline_pims(df_subset: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, float).reshape(-1)
    out = np.zeros_like(preds)

    for b in np.unique(df_subset["Battery_ID"].values):
        idx = np.where(df_subset["Battery_ID"].values == b)[0]
        order = idx[np.argsort(df_subset.loc[idx, "Cycle_Index"].values.astype(float))]
        xcyc = df_subset.loc[order, "Cycle_Index"].values.astype(float)
        p = preds[order]
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        out[order] = iso.fit_transform(xcyc, p)

    return out

def apply_causal_pims(df_subset: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, float).reshape(-1)
    out = np.zeros_like(preds)

    for b in np.unique(df_subset["Battery_ID"].values):
        idx = np.where(df_subset["Battery_ID"].values == b)[0]
        order = idx[np.argsort(df_subset.loc[idx, "Cycle_Index"].values.astype(float))]
        out[order] = np.minimum.accumulate(preds[order].copy())

    return out

def _safe_features(df: pd.DataFrame, feats: Optional[List[str]]) -> List[str]:
    if feats is None:
        return []
    return [f for f in feats if f in df.columns]

def build_baseline(df_tr: pd.DataFrame, df_te: pd.DataFrame, baseline_feats: List[str], params: Params):
    baseline_feats = _safe_features(df_tr, baseline_feats)
    if len(baseline_feats) == 0:
        raise ValueError("No baseline features provided.")

    y_tr = df_tr["RUL"].values.astype(float)
    y_te = df_te["RUL"].values.astype(float)

    sc_base = StandardScaler()
    Xtr_b = sc_base.fit_transform(df_tr[baseline_feats].values)
    Xte_b = sc_base.transform(df_te[baseline_feats].values)

    base = Ridge(alpha=params.base_alpha)
    base.fit(Xtr_b, y_tr)

    max_clip = float(y_tr.max() * 1.1)
    yb_tr = np.clip(base.predict(Xtr_b), 0.0, max_clip)
    yb_te = np.clip(base.predict(Xte_b), 0.0, max_clip)

    return y_tr, y_te, yb_tr, yb_te, max_clip

def _make_single_residual_model(params: Params, seed: int, fast_mode: bool = False):
    kind = SINGLE_RESIDUAL_BRANCH_KIND

    if kind == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=60 if fast_mode else 240,
            max_depth=int(params.et_depth),
            min_samples_leaf=int(params.et_leaf),
            n_jobs=-1,
            random_state=seed,
        )

    if kind == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=180 if fast_mode else 900,
            learning_rate=max(0.003, float(params.lgbm_learning_rate)),
            max_depth=int(params.lgbm_max_depth),
            num_leaves=max(8, min(31, 2 ** int(params.lgbm_max_depth))),
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=float(params.lgbm_reg_lambda),
            min_child_samples=20,
            objective="huber",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    if kind == "HuberRegressor":
        return HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=700)

    raise ValueError(f"Unknown SINGLE_RESIDUAL_BRANCH_KIND: {kind}")

def build_gated_residual_ensemble(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    residual_feats: List[str],
    yb_tr: np.ndarray,
    yb_te: np.ndarray,
    max_clip: float,
    seed: int,
    params: Params,
    fast_mode: bool = False,
) -> Dict[str, Any]:

    y_tr = df_tr["RUL"].values.astype(float)
    residual_feats = _safe_features(df_tr, residual_feats)

    if USE_QGWO_ENSEMBLE_BRANCHES and len(residual_feats) > 0:
        groups = df_tr["Battery_ID"].values
        r_base_tr = y_tr - yb_tr

        sc = StandardScaler()
        Xtr = sc.fit_transform(df_tr[residual_feats].values)
        Xte = sc.transform(df_te[residual_feats].values)

        n_splits = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)

        oof_et = np.zeros(len(df_tr), dtype=float)
        oof_lgbm = np.zeros(len(df_tr), dtype=float)
        oof_huber = np.zeros(len(df_tr), dtype=float)
        pred_et = np.zeros(len(df_te), dtype=float)
        pred_lgbm = np.zeros(len(df_te), dtype=float)
        pred_huber = np.zeros(len(df_te), dtype=float)

        et_estimators = 60 if fast_mode else 240
        lgbm_estimators = 180 if fast_mode else 800

        for fold, (ti, vi) in enumerate(gkf.split(Xtr, r_base_tr, groups=groups)):
            et = ExtraTreesRegressor(
                n_estimators=et_estimators,
                max_depth=int(params.et_depth),
                min_samples_leaf=int(params.et_leaf),
                n_jobs=-1,
                random_state=seed + fold * 17,
            )
            et.fit(Xtr[ti], r_base_tr[ti])
            oof_et[vi] = et.predict(Xtr[vi])
            pred_et += et.predict(Xte) / n_splits

            lgbm = lgb.LGBMRegressor(
                n_estimators=lgbm_estimators,
                learning_rate=max(0.003, float(params.lgbm_learning_rate)),
                max_depth=int(params.lgbm_max_depth),
                num_leaves=max(8, min(31, 2 ** int(params.lgbm_max_depth))),
                subsample=0.75,
                colsample_bytree=0.75,
                reg_lambda=float(params.lgbm_reg_lambda),
                min_child_samples=20,
                objective="huber",
                random_state=seed + fold * 19,
                n_jobs=-1,
                verbose=-1,
            )
            lgbm.fit(Xtr[ti], r_base_tr[ti])
            oof_lgbm[vi] = lgbm.predict(Xtr[vi])
            pred_lgbm += lgbm.predict(Xte) / n_splits

            huber = HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=700)
            try:
                huber.fit(Xtr[ti], r_base_tr[ti])
                oof_huber[vi] = huber.predict(Xtr[vi])
                pred_huber += huber.predict(Xte) / n_splits
            except Exception:
                oof_huber[vi] = 0.0
                pred_huber += 0.0

        et_gate = float(params.lambda_ET)
        lambda_LGBM = float(params.lambda_LGBM)
        lambda_Huber = float(params.lambda_Huber)

        raw_sum_tr = oof_et + oof_lgbm + oof_huber
        raw_sum_te = pred_et + pred_lgbm + pred_huber
        gated_sum_tr = et_gate * oof_et + lambda_LGBM * oof_lgbm + lambda_Huber * oof_huber
        gated_sum_te = et_gate * pred_et + lambda_LGBM * pred_lgbm + lambda_Huber * pred_huber

        y_branch_raw_tr = np.clip(yb_tr + raw_sum_tr, 0.0, max_clip)
        y_branch_raw_te = np.clip(yb_te + raw_sum_te, 0.0, max_clip)
        y_gated_tr = np.clip(yb_tr + gated_sum_tr, 0.0, max_clip)
        y_gated_te = np.clip(yb_te + gated_sum_te, 0.0, max_clip)

        indiv_preds = {
            "ExtraTrees_Raw": np.clip(yb_te + pred_et, 0.0, max_clip),
            "LightGBM_Raw": np.clip(yb_te + pred_lgbm, 0.0, max_clip),
            "HuberRegressor_Raw": np.clip(yb_te + pred_huber, 0.0, max_clip),
            "ResidualEnsemble_RawSum": y_branch_raw_te,
            "ResidualEnsemble_QGWO_Gated": y_gated_te,
        }

        return {
            "y_gated_tr": y_gated_tr,
            "y_gated_te": y_gated_te,
            "gate_weights": np.array([et_gate, lambda_LGBM, lambda_Huber], dtype=float),
            "gate_names": ["ExtraTrees_Gate", "LightGBM_Gate", "HuberRegressor_Gate"],
            "indiv_preds": indiv_preds,
            "residual_feats": residual_feats,
            "branch_oof": gated_sum_tr,
            "branch_test": gated_sum_te,
        }

    return {
        "y_gated_tr": yb_tr.copy(),
        "y_gated_te": yb_te.copy(),
        "gate_weights": np.zeros(0),
        "gate_names": [],
        "indiv_preds": {},
        "residual_feats": [],
    }

def run_full_pipeline(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    residual_feats: List[str],
    kan_feats: List[str],
    baseline_feats: List[str],
    seed: int,
    params: Params,
    fast_mode: bool = False,
) -> Dict[str, Any]:

    y_tr, y_te, yb_tr, yb_te, max_clip = build_baseline(df_tr, df_te, baseline_feats, params)

    rb = build_gated_residual_ensemble(
        df_tr=df_tr,
        df_te=df_te,
        residual_feats=residual_feats,
        yb_tr=yb_tr,
        yb_te=yb_te,
        max_clip=max_clip,
        seed=seed,
        params=params,
        fast_mode=fast_mode,
    )

    y_gated_tr = rb["y_gated_tr"]
    y_gated_te = rb["y_gated_te"]

    pre_base_tr = np.clip(yb_tr, 0.0, max_clip)
    pre_base_te = np.clip(yb_te, 0.0, max_clip)
    pre_gated_tr = np.clip(y_gated_tr, 0.0, max_clip)
    pre_gated_te = np.clip(y_gated_te, 0.0, max_clip)

    kan_feats = _safe_features(df_tr, kan_feats)
    if kan_feats:
        sc_kan = StandardScaler()
        Xtr_k = sc_kan.fit_transform(df_tr[kan_feats].values)
        Xte_k = sc_kan.transform(df_te[kan_feats].values)
    else:
        Xtr_k = np.zeros((len(df_tr), 1))
        Xte_k = np.zeros((len(df_te), 1))

    tau_tr = np.clip(df_tr["Tau_rm20"].values if "Tau_rm20" in df_tr.columns else df_tr["Tau"].values, 0.0, 1.0)
    tau_te = np.clip(df_te["Tau_rm20"].values if "Tau_rm20" in df_te.columns else df_te["Tau"].values, 0.0, 1.0)

    amp_tr = 3.0 + 17.0 * tau_tr
    amp_te = 3.0 + 17.0 * tau_te

    r_kan_tr = y_tr - y_gated_tr

    kan_epochs = 5 if fast_mode else 25
    kan_model, kan_tr, kan_te = train_physics_kan_refiner(
        Xtr_k, r_kan_tr, Xte_k, amp_tr, amp_te,
        epochs=kan_epochs,
        lr=0.015,
        seed=seed,
    )

    pre_full_tr = np.clip(y_gated_tr + params.lambda_KAN * kan_tr, 0.0, max_clip)
    pre_full_te = np.clip(y_gated_te + params.lambda_KAN * kan_te, 0.0, max_clip)

    off_base_tr = np.clip(apply_offline_pims(df_tr, pre_base_tr), 0.0, max_clip)
    off_base_te = np.clip(apply_offline_pims(df_te, pre_base_te), 0.0, max_clip)

    off_gated_tr = np.clip(apply_offline_pims(df_tr, pre_gated_tr), 0.0, max_clip)
    off_gated_te = np.clip(apply_offline_pims(df_te, pre_gated_te), 0.0, max_clip)

    off_full_tr = np.clip(apply_offline_pims(df_tr, pre_full_tr), 0.0, max_clip)
    off_full_te = np.clip(apply_offline_pims(df_te, pre_full_te), 0.0, max_clip)

    causal_full_tr = np.clip(apply_causal_pims(df_tr, pre_full_tr), 0.0, max_clip)
    causal_full_te = np.clip(apply_causal_pims(df_te, pre_full_te), 0.0, max_clip)

    return {
        "y_tr": y_tr,
        "y_te": y_te,
        "tracks": {
            "baseline_pre": (pre_base_tr, pre_base_te),
            "baseline_offline_pims": (off_base_tr, off_base_te),
            "gated_residual_pre": (pre_gated_tr, pre_gated_te),
            "gated_residual_offline_pims": (off_gated_tr, off_gated_te),
            "full_pre": (pre_full_tr, pre_full_te),
            "full_offline_pims": (off_full_tr, off_full_te),
            "full_causal_pims": (causal_full_tr, causal_full_te),
        },
        "indiv_preds": rb["indiv_preds"],
        "gate_weights": rb["gate_weights"],
        "gate_names": rb["gate_names"],
        "kan_test_pred": kan_te,
        "amp_test": amp_te,
        "kan_model": kan_model,
    }

# =============================================================================
# 7. Post-Calibration
# =============================================================================
TRAIN_ONLY_POST_CALIBRATION_ENABLED = True
POST_CALIBRATION_MODE = "TrainOnlyRobustResidualCurve"
POST_CALIBRATION_MAX_ABS = 10.0
POST_CALIBRATION_SHRINK = 0.6

def _robust_center(x, trim_q=0.10):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    lo, hi = np.quantile(x, [trim_q, 1.0 - trim_q])
    y = x[(x >= lo) & (x <= hi)]
    if len(y) == 0:
        return float(np.median(x))
    return float(0.50 * np.median(y) + 0.50 * np.mean(y))

def _fit_piecewise_residual_calibrator(pred_train, resid_train, n_bins=8, max_abs=8.0, shrink=0.45):
    pred_train = np.asarray(pred_train, dtype=float).reshape(-1)
    resid_train = np.asarray(resid_train, dtype=float).reshape(-1)
    ok = np.isfinite(pred_train) & np.isfinite(resid_train)
    pred_train, resid_train = pred_train[ok], resid_train[ok]
    if len(pred_train) < 20:
        return {"kind": "constant", "offset": 0.0, "max_abs": max_abs, "shrink": shrink}

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(pred_train, qs))
    if len(edges) < 4:
        off = float(np.clip(_robust_center(resid_train) * shrink, -max_abs, max_abs))
        return {"kind": "constant", "offset": off, "max_abs": max_abs, "shrink": shrink}

    centers, offsets, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred_train >= lo) & (pred_train <= hi if hi == edges[-1] else pred_train < hi)
        if mask.sum() < 5:
            continue
        centers.append(float(np.median(pred_train[mask])))
        offsets.append(float(np.clip(_robust_center(resid_train[mask]) * shrink, -max_abs, max_abs)))
        counts.append(int(mask.sum()))

    if len(centers) < 2:
        off = float(np.clip(_robust_center(resid_train) * shrink, -max_abs, max_abs))
        return {"kind": "constant", "offset": off, "max_abs": max_abs, "shrink": shrink}

    smoothed = []
    for i in range(len(offsets)):
        vals = offsets[max(0, i - 1): min(len(offsets), i + 2)]
        smoothed.append(float(np.mean(vals)))

    return {
        "kind": "piecewise",
        "centers": centers,
        "offsets": smoothed,
        "counts": counts,
        "max_abs": float(max_abs),
        "shrink": float(shrink),
    }

def _apply_piecewise_residual_calibrator(cal, pred):
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if cal.get("kind") == "constant":
        return np.full_like(pred, float(cal.get("offset", 0.0)), dtype=float)
    centers = np.asarray(cal.get("centers", []), dtype=float)
    offsets = np.asarray(cal.get("offsets", []), dtype=float)
    if len(centers) == 0 or len(offsets) == 0:
        return np.zeros_like(pred, dtype=float)
    return np.interp(pred, centers, offsets, left=offsets[0], right=offsets[-1])

def fit_train_only_post_calibrator(df_train, y_train, pred_train, mode=POST_CALIBRATION_MODE):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    pred_train = np.asarray(pred_train, dtype=float).reshape(-1)
    resid = y_train - pred_train

    if not TRAIN_ONLY_POST_CALIBRATION_ENABLED:
        return {"enabled": False, "mode": "none"}

    if mode == "TrainOnlyRobustResidualCurve":
        cal = _fit_piecewise_residual_calibrator(
            pred_train,
            resid,
            n_bins=10,
            max_abs=POST_CALIBRATION_MAX_ABS,
            shrink=POST_CALIBRATION_SHRINK,
        )
        p75 = float(np.quantile(pred_train, 0.75))
        p90 = float(np.quantile(pred_train, 0.90))
        tail_mask = pred_train >= p75
        tail_offset = _robust_center(resid[tail_mask]) if tail_mask.sum() >= 20 else 0.0
        tail_offset = float(np.clip(tail_offset * 0.35, -POST_CALIBRATION_MAX_ABS * 0.60, POST_CALIBRATION_MAX_ABS * 0.60))
        cal.update({
            "enabled": True,
            "mode": mode,
            "tail_p75": p75,
            "tail_p90": p90,
            "tail_offset": tail_offset,
        })
        return cal

    raise ValueError(f"Unknown POST_CALIBRATION_MODE: {mode}")

def apply_train_only_post_calibrator(df_subset, pred, cal, max_clip):
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if not cal.get("enabled", False):
        return pred.copy(), np.zeros_like(pred, dtype=float)

    corr = _apply_piecewise_residual_calibrator(cal, pred)

    if cal.get("mode") == "TrainOnlyRobustResidualCurve":
        p75 = float(cal.get("tail_p75", np.quantile(pred, 0.75)))
        p90 = float(cal.get("tail_p90", np.quantile(pred, 0.90)))
        tail_offset = float(cal.get("tail_offset", 0.0))
        denom = max(p90 - p75, 1e-6)
        tail_strength = np.clip((pred - p75) / denom, 0.0, 1.0)
        corr = corr + tail_strength * tail_offset

    corr = np.clip(corr, -float(cal.get("max_abs", POST_CALIBRATION_MAX_ABS)), float(cal.get("max_abs", POST_CALIBRATION_MAX_ABS)))
    out = np.clip(pred + corr, 0.0, max_clip)
    out = np.clip(apply_causal_pims(df_subset, out), 0.0, max_clip)
    return out, corr

# =============================================================================
# 8. QGWO Tuning
# =============================================================================
def params_from_array(arr: np.ndarray, init_params: Params) -> Params:
    return Params(
        base_alpha=float(arr[0]),
        lgbm_max_depth=init_params.lgbm_max_depth,
        lgbm_learning_rate=init_params.lgbm_learning_rate,
        lgbm_reg_lambda=init_params.lgbm_reg_lambda,
        et_depth=init_params.et_depth,
        et_leaf=init_params.et_leaf,
        lambda_ET=float(arr[1]),
        lambda_LGBM=float(arr[2]),
        lambda_Huber=float(arr[3]),
        lambda_KAN=float(arr[4]),
    )

def make_qgwo_bounds():
    return [
        (10.0, 18.0),
        (0.00, 0.35),
        (0.00, 0.35),
        (0.00, 0.20),
        (0.70, 1.30),
    ]

def tune_qgwo_train_only(
    df_train: pd.DataFrame,
    residual_feats: List[str],
    baseline_feats: List[str],
    kan_feats: List[str],
    seed: int,
    init_params: Params,
    objective_track: str = QGWO_OBJECTIVE_TRACK,
) -> Tuple[Params, float, pd.DataFrame]:

    bounds = make_qgwo_bounds()

    def obj(arr):
        p = params_from_array(arr, init_params)
        groups = df_train["Battery_ID"].values
        n_splits = min(3, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)

        rmses = []
        for ti, vi in gkf.split(df_train, groups=groups):
            dtr = df_train.iloc[ti].copy().reset_index(drop=True)
            dva = df_train.iloc[vi].copy().reset_index(drop=True)

            out = run_full_pipeline(
                dtr, dva,
                residual_feats=_safe_features(dtr, residual_feats),
                kan_feats=_safe_features(dtr, kan_feats),
                baseline_feats=_safe_features(dtr, baseline_feats),
                seed=seed,
                params=p,
                fast_mode=True,
            )

            yv = out["y_te"]
            pv = out["tracks"][objective_track][1]
            rmses.append(calc_metrics(yv, pv)["RMSE"])

        return float(
            np.mean(rmses)
            + QGWO_BATTERY_STD_LAMBDA * np.std(rmses)
            + QGWO_BATTERY_MAX_LAMBDA * np.max(rmses)
        )

    q = LocalQGWO(obj, bounds, num_wolves=GLOBAL_QGWO_WOLVES, max_iter=GLOBAL_QGWO_ITERS)
    best_arr, best_score = q.optimize()
    tuned = params_from_array(best_arr, init_params)

    return tuned, float(best_score), q.history_df()

# =============================================================================
# 9. Experiment runners
# =============================================================================
def collect_ablation_rows(y_te, base_pred, gated_residual_pred, qgwo_gated_residual_pred, full_pred, causal_full_pred) -> Dict[str, float]:
    return {
        "Baseline_RMSE": calc_metrics(y_te, base_pred)["RMSE"],
        "ResidualEnsemble_RMSE": calc_metrics(y_te, gated_residual_pred)["RMSE"],
        "QGWO_GatedResidual_RMSE": calc_metrics(y_te, qgwo_gated_residual_pred)["RMSE"],
        "QGWO_GatedResidual_PhysicsKAN_RMSE": calc_metrics(y_te, full_pred)["RMSE"],
        "QPEAK_Final_CausalPIMS_RMSE": calc_metrics(y_te, causal_full_pred)["RMSE"],
    }

def add_stage_metric_rows(rows, protocol, fold, stage, y_train, pred_train, y_test, pred_test, extra=None):
    extra = extra or {}
    for set_name, y, p in [("Train", y_train, pred_train), ("Test", y_test, pred_test)]:
        m = calc_metrics(y, p)
        row = {
            "Protocol": protocol,
            "Fold": fold,
            "Stage": stage,
            "Set": set_name,
            **m,
        }
        row.update(extra)
        rows.append(row)

def summarize_stage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["Protocol", "Stage", "Set"], as_index=False)
        .agg(
            N=("RMSE", "count"),
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            RMSE_Median=("RMSE", "median"),
            MAE_Mean=("MAE", "mean"),
            MAE_Std=("MAE", "std"),
            MSE_Mean=("MSE", "mean"),
            MSE_Std=("MSE", "std"),
            R2_Mean=("R2", "mean"),
            R2_Std=("R2", "std"),
        )
        .sort_values(["Protocol", "Stage", "Set"])
        .reset_index(drop=True)
    )

def run_five_seeds(raw_with_id: pd.DataFrame, default_params: Params, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"[RUN] 5-seed Group Split | {MODEL_VARIANT}")
    print("=" * 80)

    batteries = np.array(sorted(raw_with_id["Battery_ID"].unique()))
    pre_test, off_test, causal_test = init_metric_dict(), init_metric_dict(), init_metric_dict()

    seed_rows, ablation_rows, gate_rows, pims_rows = [], [], [], []
    stage_metric_rows = []
    prediction_records = []

    for s in FIVE_SEEDS:
        seed_everything(s)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
        tr_i, te_i = next(gss.split(batteries.reshape(-1, 1), groups=batteries))

        train_batts = list(map(int, batteries[tr_i]))
        test_batts = list(map(int, batteries[te_i]))

        dtr, dte = prepare_fold_features(raw_with_id, train_batts, test_batts)

        _, full_feats, kan_feats, _, _ = get_feature_sets(pd.concat([dtr, dte], ignore_index=True))
        baseline_feats = [c for c in CLOCK_BASELINE_FEATURES if c in dtr.columns]
        residual_feats = full_feats if (USE_LEGACY_RESIDUAL_BANK or USE_SINGLE_RESIDUAL_BRANCH or USE_QGWO_ENSEMBLE_BRANCHES) else []

        out_default = run_full_pipeline(
            dtr, dte,
            residual_feats=residual_feats,
            kan_feats=kan_feats,
            baseline_feats=baseline_feats,
            seed=s,
            params=default_params,
            fast_mode=False,
        )

        tuned_params = default_params
        qgwo_obj = None

        if ENABLE_QGWO_5SEED:
            tuned_params, qgwo_obj, _ = tune_qgwo_train_only(
                dtr,
                residual_feats,
                baseline_feats,
                kan_feats,
                s,
                default_params,
                objective_track=QGWO_OBJECTIVE_TRACK,
            )

        out = run_full_pipeline(
            dtr, dte,
            residual_feats=residual_feats,
            kan_feats=kan_feats,
            baseline_feats=baseline_feats,
            seed=s,
            params=tuned_params,
            fast_mode=False,
        )

        y_te = out["y_te"]
        pre = out["tracks"]["full_pre"][1]
        off = out["tracks"]["full_offline_pims"][1]
        cau_raw = out["tracks"]["full_causal_pims"][1]

        max_clip = float(np.max(out["y_tr"]) * 1.1)
        post_cal = fit_train_only_post_calibrator(
            dtr,
            out["y_tr"],
            out["tracks"]["full_causal_pims"][0],
            mode=POST_CALIBRATION_MODE,
        )
        cau, post_corr = apply_train_only_post_calibrator(dte, cau_raw, post_cal, max_clip=max_clip)

        m_pre = calc_metrics(y_te, pre)
        m_off = calc_metrics(y_te, off)
        m_cau_raw = calc_metrics(y_te, cau_raw)
        m_cau = calc_metrics(y_te, cau)

        append_metrics(pre_test, m_pre)
        append_metrics(off_test, m_off)
        append_metrics(causal_test, m_cau)

        seed_rows.append({
            "Model_Variant": MODEL_VARIANT,
            "Seed": s,
            "PrePIMS_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
            "CausalPIMS_MAE": m_cau["MAE"],
            "CausalPIMS_R2": m_cau["R2"],
            "QGWO_Obj": qgwo_obj,
            **{f"Param_{k}": v for k, v in asdict(tuned_params).items()},
        })

        ab_row = collect_ablation_rows(
            y_te,
            out_default["tracks"]["baseline_offline_pims"][1],
            out_default["tracks"]["gated_residual_offline_pims"][1],
            out["tracks"]["gated_residual_offline_pims"][1],
            out["tracks"]["full_offline_pims"][1],
            out["tracks"]["full_causal_pims"][1],
        )
        ab_row["Seed"] = s
        ablation_rows.append(ab_row)

        stage_extra = {"Seed": s}
        add_stage_metric_rows(
            stage_metric_rows, "FiveSeed", f"Seed_{s}", "01_Baseline_default",
            out_default["y_tr"], out_default["tracks"]["baseline_offline_pims"][0],
            out_default["y_te"], out_default["tracks"]["baseline_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "FiveSeed", f"Seed_{s}", "02_ResidualEnsemble_default",
            out_default["y_tr"], out_default["tracks"]["gated_residual_offline_pims"][0],
            out_default["y_te"], out_default["tracks"]["gated_residual_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "FiveSeed", f"Seed_{s}", "03_QGWO_GatedResidual",
            out["y_tr"], out["tracks"]["gated_residual_offline_pims"][0],
            out["y_te"], out["tracks"]["gated_residual_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "FiveSeed", f"Seed_{s}", "04_QGWO_GatedResidual_PhysicsKAN_PrePIMS",
            out["y_tr"], out["tracks"]["full_pre"][0],
            out["y_te"], out["tracks"]["full_pre"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "FiveSeed", f"Seed_{s}", "05_QGWO_GatedResidual_PhysicsKAN_OfflinePIMS",
            out["y_tr"], out["tracks"]["full_offline_pims"][0],
            out["y_te"], out["tracks"]["full_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "FiveSeed", f"Seed_{s}", "06_QPEAK_Final_CausalPIMS",
            out["y_tr"], out["tracks"]["full_causal_pims"][0],
            out["y_te"], out["tracks"]["full_causal_pims"][1],
            stage_extra,
        )

        gate_row = {"Fold": f"Seed_{s}", "Seed": s}
        if len(out["gate_names"]) == 0:
            gate_row["NoGatedResidualEnsemble"] = 1
        else:
            for name, val in zip(out["gate_names"], out["gate_weights"]):
                gate_row[f"W_{name}"] = float(val)
        gate_rows.append(gate_row)

        pims_rows.append({
            "Fold": f"Seed_{s}",
            "Pre_Violations": monotonic_violation_count(dte, pre),
            "OfflinePIMS_Violations": monotonic_violation_count(dte, off),
            "CausalPIMS_Violations": monotonic_violation_count(dte, cau),
            "Pre_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
        })

        if SAVE_PREDICTIONS:
            tmp = dte[["Battery_ID", "Cycle_Index", "RUL"]].copy()
            tmp["Seed"] = s
            tmp["True_RUL"] = y_te
            tmp["Pred_PrePIMS"] = pre
            tmp["Pred_OfflinePIMS"] = off
            tmp["Pred_CausalPIMS"] = cau
            prediction_records.append(tmp)

        print(f"Seed {s} | Test RMSE: {m_cau['RMSE']:.4f}")

    seed_df = pd.DataFrame(seed_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    gate_df = pd.DataFrame(gate_rows)
    pims_df = pd.DataFrame(pims_rows)
    stage_df = pd.DataFrame(stage_metric_rows)
    stage_summary_df = (
        stage_df.groupby(["Protocol", "Stage", "Set"], as_index=False)
        .agg(RMSE_Mean=("RMSE", "mean"))
    )

    safe_to_csv(seed_df, os.path.join(out_dir, "group_split_5seed_summary.csv"))
    safe_to_csv(ablation_df, os.path.join(out_dir, "group_split_5seed_stage_ablation.csv"))
    safe_to_csv(gate_df, os.path.join(out_dir, "group_split_5seed_qgwo_gate_weights.csv"))
    safe_to_csv(pims_df, os.path.join(out_dir, "group_split_5seed_pims_diagnostics.csv"))

    if SAVE_PREDICTIONS and prediction_records:
        safe_to_csv(pd.concat(prediction_records, ignore_index=True), os.path.join(out_dir, "group_split_5seed_predictions.csv"))

    return {
        "pre_test": pre_test,
        "off_test": off_test,
        "causal_test": causal_test,
        "seed_df": seed_df,
        "ablation_df": ablation_df,
        "gate_df": gate_df,
        "pims_df": pims_df,
        "stage_df": stage_df,
        "stage_summary_df": stage_summary_df,
    }

def run_lobo(raw_with_id: pd.DataFrame, default_params: Params, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"[RUN] LOBO | {MODEL_VARIANT}")
    print("=" * 80)

    batts = np.array(sorted(raw_with_id["Battery_ID"].unique()))
    pre_test, off_test, causal_test = init_metric_dict(), init_metric_dict(), init_metric_dict()
    rows, gate_rows, pims_rows = [], [], []
    stage_metric_rows = []
    prediction_records = []

    for b in batts:
        seed = 45
        seed_everything(seed)

        train_batts = [int(x) for x in batts if int(x) != int(b)]
        test_batts = [int(b)]
        label = f"B{int(b)+1:02d}"

        dtr, dte = prepare_fold_features(raw_with_id, train_batts, test_batts)
        _, full_feats, kan_feats, _, _ = get_feature_sets(pd.concat([dtr, dte], ignore_index=True))

        baseline_feats = [c for c in CLOCK_BASELINE_FEATURES if c in dtr.columns]
        residual_feats = full_feats if (USE_LEGACY_RESIDUAL_BANK or USE_SINGLE_RESIDUAL_BRANCH or USE_QGWO_ENSEMBLE_BRANCHES) else []

        used_params = default_params
        qgwo_obj = None

        out = run_full_pipeline(
            dtr, dte,
            residual_feats=residual_feats,
            kan_feats=kan_feats,
            baseline_feats=baseline_feats,
            seed=seed,
            params=used_params,
            fast_mode=False,
        )

        y_te = out["y_te"]
        pre = out["tracks"]["full_pre"][1]
        off = out["tracks"]["full_offline_pims"][1]
        cau_raw = out["tracks"]["full_causal_pims"][1]

        max_clip = float(np.max(out["y_tr"]) * 1.1)
        post_cal = fit_train_only_post_calibrator(
            dtr,
            out["y_tr"],
            out["tracks"]["full_causal_pims"][0],
            mode=POST_CALIBRATION_MODE,
        )
        cau, post_corr = apply_train_only_post_calibrator(dte, cau_raw, post_cal, max_clip=max_clip)

        m_pre = calc_metrics(y_te, pre)
        m_off = calc_metrics(y_te, off)
        m_cau_raw = calc_metrics(y_te, cau_raw)
        m_cau = calc_metrics(y_te, cau)

        append_metrics(pre_test, m_pre)
        append_metrics(off_test, m_off)
        append_metrics(causal_test, m_cau)

        rows.append({
            "Model_Variant": MODEL_VARIANT,
            "Battery_ID": int(b),
            "Battery_Label": label,
            "PrePIMS_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
            "CausalPIMS_MAE": m_cau["MAE"],
            "CausalPIMS_R2": m_cau["R2"],
            "QGWO_Obj": qgwo_obj,
        })

        stage_extra = {"Battery_ID": int(b), "Battery_Label": label}
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "06_QPEAK_Final_CausalPIMS",
            out["y_tr"], out["tracks"]["full_causal_pims"][0],
            out["y_te"], cau,
            stage_extra,
        )

        gate_row = {"Fold": label, "Battery_ID": int(b)}
        if len(out["gate_names"]) == 0:
            gate_row["NoGatedResidualEnsemble"] = 1
        else:
            for name, val in zip(out["gate_names"], out["gate_weights"]):
                gate_row[f"W_{name}"] = float(val)
        gate_rows.append(gate_row)

        pims_rows.append({
            "Fold": label,
            "Battery_ID": int(b),
            "Pre_Violations": monotonic_violation_count(dte, pre),
            "OfflinePIMS_Violations": monotonic_violation_count(dte, off),
            "CausalPIMS_Violations": monotonic_violation_count(dte, cau),
            "Pre_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
        })

        if SAVE_PREDICTIONS:
            tmp = dte[["Battery_ID", "Cycle_Index", "RUL"]].copy()
            tmp["Fold"] = label
            tmp["True_RUL"] = y_te
            tmp["Pred_PrePIMS"] = pre
            tmp["Pred_OfflinePIMS"] = off
            tmp["Pred_CausalPIMS"] = cau
            prediction_records.append(tmp)

        print(f"LOBO Battery {b} | Test RMSE: {m_cau['RMSE']:.4f}")

    rows_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    pims_df = pd.DataFrame(pims_rows)
    stage_df = pd.DataFrame(stage_metric_rows)

    safe_to_csv(rows_df, os.path.join(out_dir, "lobo_summary.csv"))
    safe_to_csv(gate_df, os.path.join(out_dir, "lobo_qgwo_gate_weights.csv"))
    safe_to_csv(pims_df, os.path.join(out_dir, "lobo_pims_diagnostics.csv"))

    if SAVE_PREDICTIONS and prediction_records:
        safe_to_csv(pd.concat(prediction_records, ignore_index=True), os.path.join(out_dir, "lobo_predictions.csv"))

    return {
        "pre_test": pre_test,
        "off_test": off_test,
        "causal_test": causal_test,
        "rows": rows_df,
        "gate_df": gate_df,
        "pims_df": pims_df,
        "stage_df": stage_df,
    }

def save_master_summary(five_res: Dict[str, Any], lobo_res: Dict[str, Any], out_dir: str):
    rows = [
        {"Model_Variant": MODEL_VARIANT, "Experiment": "FiveSeed_PrePIMS", **summarize_metric_dict(five_res["pre_test"], "Test_")},
        {"Model_Variant": MODEL_VARIANT, "Experiment": "FiveSeed_OfflinePIMS", **summarize_metric_dict(five_res["off_test"], "Test_")},
        {"Model_Variant": MODEL_VARIANT, "Experiment": "FiveSeed_CausalPIMS", **summarize_metric_dict(five_res["causal_test"], "Test_")},
        {"Model_Variant": MODEL_VARIANT, "Experiment": "LOBO_PrePIMS", **summarize_metric_dict(lobo_res["pre_test"], "Test_")},
        {"Model_Variant": MODEL_VARIANT, "Experiment": "LOBO_OfflinePIMS", **summarize_metric_dict(lobo_res["off_test"], "Test_")},
        {"Model_Variant": MODEL_VARIANT, "Experiment": "LOBO_CausalPIMS", **summarize_metric_dict(lobo_res["causal_test"], "Test_")},
    ]
    df = pd.DataFrame(rows)
    safe_to_csv(df, os.path.join(out_dir, "qpeak_final_master_metric_summary.csv"))
    return df

def save_ablation_summary(five_res: Dict[str, Any], out_dir: str):
    df = five_res["ablation_df"].copy()
    stages = [
        "Baseline_RMSE",
        "ResidualEnsemble_RMSE",
        "QGWO_GatedResidual_RMSE",
        "QGWO_GatedResidual_PhysicsKAN_RMSE",
        "QPEAK_Final_CausalPIMS_RMSE",
    ]
    rows = []
    for s in stages:
        rows.append({
            "Stage": s.replace("_RMSE", ""),
            "Mean_RMSE": df[s].mean(),
            "Std_RMSE": df[s].std(),
            "Median_RMSE": df[s].median(),
        })
    out = pd.DataFrame(rows)
    safe_to_csv(out, os.path.join(out_dir, "qpeak_final_stage_ablation_summary.csv"))
    return out

# =============================================================================
# 10. Generating figures for paper ( 置底隔離 )
# =============================================================================
def generate_essential_figures(five_res: Dict[str, Any], lobo_res: Dict[str, Any], ablation_summary: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    # 1. 5-seed RMSE summary (fig_group_split_5seed_rmse_summary.png)
    seed_df = five_res["seed_df"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    ax.plot(seed_df["Seed"], seed_df["PrePIMS_RMSE"], marker="o", label="Pre-PIMS", linewidth=2)
    ax.plot(seed_df["Seed"], seed_df["OfflinePIMS_RMSE"], marker="s", label="Offline PIMS", linewidth=2)
    ax.plot(seed_df["Seed"], seed_df["CausalPIMS_RMSE"], marker="^", label="Causal PIMS", linewidth=2)
    _apply_axis_text(ax, title="5-seed Group Split RMSE Summary", xlabel="Random Seed", ylabel="RMSE (Cycles)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=THESIS_LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_group_split_5seed_rmse_summary.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2. LOBO RMSE by battery (fig_lobo_rmse_by_battery.png)
    lobo_df = lobo_res["rows"].copy()
    vals = lobo_df["CausalPIMS_RMSE"].values.astype(float)
    x = np.arange(len(lobo_df))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    ax.bar(x, vals, edgecolor="black", color="skyblue")
    ax.axhline(np.mean(vals), color="red", linestyle="--", label=f"Mean={np.mean(vals):.2f}")
    ax.axhline(np.median(vals), color="green", linestyle=":", label=f"Median={np.median(vals):.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(lobo_df["Battery_Label"].values, fontsize=THESIS_TICK_FONTSIZE)
    _apply_axis_text(ax, title="LOBO Cross-Validation RMSE by Battery", xlabel="Battery", ylabel="RMSE (Cycles)")
    ax.legend(fontsize=THESIS_LEGEND_FONTSIZE)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_lobo_rmse_by_battery.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3. Stage Ablation (fig_stage_ablation_rmse.png)
    abl = ablation_summary.copy().sort_values("Mean_RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=FIG_DPI)
    y_pos = np.arange(len(abl))
    ax.barh(y_pos, abl["Mean_RMSE"].values.astype(float), edgecolor="black", color="salmon")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(abl["Stage"].values, fontsize=THESIS_TICK_FONTSIZE)
    _apply_axis_text(ax, title="Stage Ablation Studies - RMSE", xlabel="Average Test RMSE", ylabel="Module Stage")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, v in enumerate(abl["Mean_RMSE"].values.astype(float)):
        ax.text(v + 0.05, i, f"{v:.3f}", va="center", fontsize=THESIS_ANNOT_FONTSIZE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_stage_ablation_rmse.png"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 4. PIMS Violations summary (fig_pims_diagnostics_summary.png)
    try:
        pv_rows = []
        for protocol, dfp in [("FiveSeed", five_res["pims_df"]), ("LOBO", lobo_res["pims_df"])]:
            if not dfp.empty:
                pv_rows.append({"Label": f"{protocol}\nPre-PIMS", "Violations": dfp["Pre_Violations"].mean()})
                pv_rows.append({"Label": f"{protocol}\nOffline PIMS", "Violations": dfp["OfflinePIMS_Violations"].mean()})
                pv_rows.append({"Label": f"{protocol}\nCausal PIMS", "Violations": dfp["CausalPIMS_Violations"].mean()})
        pv = pd.DataFrame(pv_rows)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
        ax.bar(np.arange(len(pv)), pv["Violations"].values, color="mediumpurple", edgecolor="black")
        ax.set_xticks(np.arange(len(pv)))
        ax.set_xticklabels(pv["Label"].values, rotation=15, ha="right", fontsize=THESIS_TICK_FONTSIZE)
        _apply_axis_text(ax, title="Monotonicity Violations Before and After PIMS Calibration", ylabel="Violation Count")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_pims_diagnostics_summary.png"), dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to generate PIMS violations plot: {e}")

    # 5. QGWO Gate Weights (fig_qgwo_gate_weights_5seeds.png)
    try:
        gw_df = five_res["gate_df"].copy()
        w_cols = [c for c in gw_df.columns if c.startswith("W_")]
        if w_cols:
            x_pos = np.arange(len(gw_df))
            bottom = np.zeros(len(gw_df))
            fig, ax = plt.subplots(figsize=(9, 5), dpi=FIG_DPI)
            for col in w_cols:
                vals = gw_df[col].fillna(0).values
                ax.bar(x_pos, vals, bottom=bottom, label=col.replace("W_", ""), edgecolor="black")
                bottom += vals
            ax.set_xticks(x_pos)
            ax.set_xticklabels(gw_df["Fold"].values, rotation=0, fontsize=THESIS_TICK_FONTSIZE)
            _apply_axis_text(ax, title="QGWO Gate Weight Distribution - 5 Seeds", xlabel="Seed Split", ylabel="Gate Weight")
            ax.legend(fontsize=THESIS_LEGEND_FONTSIZE)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_qgwo_gate_weights_5seeds.png"), dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to generate gate weight plot: {e}")

    # 6. Trajectories for LOBO (fig_lobo_rul_prediction_trajectories.png)
    try:
        pred_path = os.path.join(out_dir, "lobo_predictions.csv")
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            folds = sorted(pred_df["Fold"].dropna().unique().tolist())
            ncols = 3
            nrows = int(np.ceil(len(folds) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), dpi=FIG_DPI)
            axes = np.atleast_1d(axes).ravel()
            for ax, fold in zip(axes, folds):
                g = pred_df[pred_df["Fold"] == fold].sort_values("Cycle_Index")
                ax.plot(g["Cycle_Index"], g["True_RUL"], label="True RUL", color="black", linewidth=2)
                ax.plot(g["Cycle_Index"], g["Pred_PrePIMS"], label="Pre-PIMS", color="orange", linestyle="--", linewidth=1.2)
                ax.plot(g["Cycle_Index"], g["Pred_CausalPIMS"], label="Causal PIMS", color="green", linestyle="-.", linewidth=1.5)
                _apply_axis_text(ax, title=f"Battery {fold}")
                ax.grid(True, linestyle=":", alpha=0.4)
            for ax in axes[len(folds):]:
                ax.axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=THESIS_LEGEND_FONTSIZE)
            fig.suptitle("LOBO RUL Prediction Trajectories", fontsize=THESIS_TITLE_FONTSIZE, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(out_dir, "fig_lobo_rul_prediction_trajectories.png"), dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to generate trajectories plot: {e}")

# =============================================================================
# 11. Model Training & Serialization for Deployment
# =============================================================================
def train_and_save_final_model(df_raw: pd.DataFrame, output_path: str, params: Params, seed: int = 42):
    print("\n" + "=" * 80)
    print(f"[DEPLOYMENT] Fitting final models for deployment on full dataset...")
    print("=" * 80)

    clean_params = fit_cleaning_params(df_raw)
    df_clean = apply_cleaning(df_raw, clean_params)
    df_all = engineer_features(df_clean)

    clock_feats, full_feats, kan_feats, _, _ = get_feature_sets(df_all)
    baseline_feats = [c for c in CLOCK_BASELINE_FEATURES if c in df_all.columns]
    residual_feats = full_feats if (USE_LEGACY_RESIDUAL_BANK or USE_SINGLE_RESIDUAL_BRANCH or USE_QGWO_ENSEMBLE_BRANCHES) else []

    y = df_all["RUL"].values.astype(float)

    # 1. Ridge Baseline
    sc_base = StandardScaler()
    X_base = sc_base.fit_transform(df_all[baseline_feats].values)
    base_model = Ridge(alpha=params.base_alpha)
    base_model.fit(X_base, y)

    max_clip = float(y.max() * 1.1)
    yb = np.clip(base_model.predict(X_base), 0.0, max_clip)
    r_base = y - yb

    # 2. Gated Residual Ensemble (GroupKFold split models for robustness)
    groups = df_all["Battery_ID"].values
    sc = StandardScaler()
    X_res = sc.fit_transform(df_all[residual_feats].values)

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    et_models = []
    lgbm_models = []
    huber_models = []

    oof_et = np.zeros(len(df_all))
    oof_lgbm = np.zeros(len(df_all))
    oof_huber = np.zeros(len(df_all))

    for fold, (ti, vi) in enumerate(gkf.split(X_res, r_base, groups=groups)):
        et = ExtraTreesRegressor(
            n_estimators=240,
            max_depth=int(params.et_depth),
            min_samples_leaf=int(params.et_leaf),
            n_jobs=-1,
            random_state=seed + fold * 17,
        )
        et.fit(X_res[ti], r_base[ti])
        et_models.append(et)
        oof_et[vi] = et.predict(X_res[vi])

        lgbm = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=max(0.003, float(params.lgbm_learning_rate)),
            max_depth=int(params.lgbm_max_depth),
            num_leaves=max(8, min(31, 2 ** int(params.lgbm_max_depth))),
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=float(params.lgbm_reg_lambda),
            min_child_samples=20,
            objective="huber",
            random_state=seed + fold * 19,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_res[ti], r_base[ti])
        lgbm_models.append(lgbm)
        oof_lgbm[vi] = lgbm.predict(X_res[vi])

        huber = HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=700)
        try:
            huber.fit(X_res[ti], r_base[ti])
            huber_models.append(huber)
            oof_huber[vi] = huber.predict(X_res[vi])
        except Exception:
            huber_models.append(None)
            oof_huber[vi] = 0.0

    et_gate = float(params.lambda_ET)
    lgbm_gate = float(params.lambda_LGBM)
    huber_gate = float(params.lambda_Huber)

    gated_sum_tr = et_gate * oof_et + lgbm_gate * oof_lgbm + huber_gate * oof_huber
    y_gated = np.clip(yb + gated_sum_tr, 0.0, max_clip)

    # 3. PhysicsKAN Refiner
    sc_kan = StandardScaler()
    X_kan = sc_kan.fit_transform(df_all[kan_feats].values)

    tau = np.clip(df_all["Tau_rm20"].values if "Tau_rm20" in df_all.columns else df_all["Tau"].values, 0.0, 1.0)
    amp = 3.0 + 17.0 * tau

    r_kan = y - y_gated
    kan_model = None
    kan_state_dict = None
    kan_pred = np.zeros(len(df_all))

    if TORCH_OK and len(kan_feats) > 0:
        print("[INFO] PyTorch available. Training PhysicsKANRefiner on remaining residual...")
        kan_model, kan_pred, _ = train_physics_kan_refiner(
            X_kan, r_kan, X_kan, amp, amp,
            epochs=25,
            lr=0.015,
            seed=seed
        )
        if kan_model is not None:
            kan_state_dict = kan_model.state_dict()
    else:
        print("[INFO] PyTorch not available or kan_feats empty. Skipping KAN training.")

    y_full = np.clip(y_gated + params.lambda_KAN * kan_pred, 0.0, max_clip)

    # 4. Piecewise Post-Calibrator
    post_cal = fit_train_only_post_calibrator(
        df_all,
        y,
        y_full,
        mode=POST_CALIBRATION_MODE
    )

    # 5. Serialize Model Package
    model_pkg = {
        "clean_params": clean_params,
        "sc_base": sc_base,
        "base_model": base_model,
        "sc": sc,
        "et_models": et_models,
        "lgbm_models": lgbm_models,
        "huber_models": huber_models,
        "et_gate": et_gate,
        "lgbm_gate": lgbm_gate,
        "huber_gate": huber_gate,
        "sc_kan": sc_kan,
        "kan_state_dict": kan_state_dict,
        "lambda_KAN": float(params.lambda_KAN),
        "post_cal": post_cal,
        "max_clip": max_clip,
        "clock_feats": baseline_feats,
        "residual_feats": residual_feats,
        "kan_feats": kan_feats,
    }

    joblib.dump(model_pkg, output_path)
    print(f"[OK] Model package serialized and saved successfully to '{output_path}'.")

# =============================================================================
# 12. Main Entry point
# =============================================================================
def main():
    import sys
    ensure_dir(OUTPUT_DIR)
    seed_everything(42)

    dataset_path = find_dataset_path()
    print(f"[INFO] Model variant: {MODEL_VARIANT}")
    print(f"[INFO] Dataset path: {dataset_path}")
    print(f"[INFO] Torch: {TORCH_OK}, device={DEVICE}")

    raw = pd.read_csv(dataset_path)
    raw_with_id = attach_battery_id(raw)

    default_params = Params()

    # Save details
    save_json(asdict(default_params), os.path.join(OUTPUT_DIR, "default_params.json"))

    final_only = "--final-only" in sys.argv
    if final_only:
        print("[INFO] --final-only flag detected. Skipping 5-seed and LOBO evaluations. Fitting final deployment model directly...")
        train_and_save_final_model(raw_with_id, "probms_model.pkl", default_params)
        return

    # Execute training split evaluations
    five_res = run_five_seeds(raw_with_id, default_params, OUTPUT_DIR)
    lobo_res = run_lobo(raw_with_id, default_params, OUTPUT_DIR)

    # Summaries and figures
    master = save_master_summary(five_res, lobo_res, OUTPUT_DIR)
    ablation_summary = save_ablation_summary(five_res, OUTPUT_DIR)
    generate_essential_figures(five_res, lobo_res, ablation_summary, OUTPUT_DIR)

    # Print final validation metrics to stdout
    print("\n" + "=" * 80)
    print("[VALIDATION MASTER SUMMARY]")
    print("=" * 80)
    print(master[["Experiment", "Test_RMSE_Mean", "Test_RMSE_Std", "Test_MAE_Mean", "Test_R2_Mean"]].to_string(index=False))

    # Train final deployment model and save package to pkl
    train_and_save_final_model(raw_with_id, "probms_model.pkl", default_params)

if __name__ == "__main__":
    main()