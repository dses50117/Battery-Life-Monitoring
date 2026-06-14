# -*- coding: utf-8 -*-
"""
Q-PEAK thesis-clean naming script
Variant: QPEAK_Final_ET_LGBM_Huber_PhysicsKAN_CausalPIMS

Common fixes:
1. Unified baseline features:
   Cycle_Index, Cycle_log, Cum_Ah, Cum_Energy, Cum_Ah_log1p, Cum_Energy_log1p
2. QGWO objective = final reported output: full_causal_pims
3. QGWO tunes lambda_KAN
4. Legacy residual ensemble is disabled and replaced by QGWO-gated residual ensemble.
5. HGB remains removed.
6. Causal Tau and train-only cleaning are retained.
"""

import os, sys, math, json, time, random, shutil, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from scipy.optimize import nnls as _legacy_nnls
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, HuberRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TORCH_OK = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    TORCH_OK = False

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
    "/kaggle/input/datasets/ignaciovinuales/battery-remaining-useful-life-rul/Battery_RUL.csv",
    "/kaggle/input/battery-remaining-useful-life-rul/Battery_RUL.csv",
    "/kaggle/input/battery-rul/Battery_RUL.csv",
    "/mnt/data/Battery_RUL.csv",
]

OUTPUT_DIR = f"/kaggle/working/qpeak_{MODEL_VARIANT}_results"
ZIP_PATH = f"/kaggle/working/qpeak_{MODEL_VARIANT}_results.zip"

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
            params["clip_hi"][c] = float(vals.quantile(0.999))
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
        return np.zeros(len(X_tr)), np.zeros(len(X_te))

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
        return model(Xt, at).detach().cpu().numpy(), model(Xte, ate).detach().cpu().numpy()


# =============================================================================
# 6. Model pipeline
# =============================================================================
@dataclass
class Params:
    base_alpha: float = 15.0
    # LightGBM residual learner settings used in the QGWO-gated residual ensemble.
    lgbm_max_depth: int = 5
    lgbm_learning_rate: float = 0.009
    lgbm_reg_lambda: float = 22.0
    et_depth: int = 8
    et_leaf: int = 9
    lambda_ET: float = 0.12   # ExtraTrees gate
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
    """Create the single residual learner used before KAN.

    The residual learner is trained on y - y_base through inner GroupKFold OOF.
    Its output is multiplied by params.lambda_ET before being added to the baseline.
    """
    kind = SINGLE_RESIDUAL_BRANCH_KIND

    if kind == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=60 if fast_mode else 240,
            max_depth=int(params.et_depth),
            min_samples_leaf=int(params.et_leaf),
            n_jobs=-1,
            random_state=seed,
        )

    if kind == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=180 if fast_mode else 900,
            learning_rate=params.lgbm_learning_rate,
            max_depth=int(params.lgbm_max_depth),
            subsample=0.70,
            colsample_bytree=0.70,
            reg_lambda=params.lgbm_reg_lambda,
            objective="reg:pseudohubererror",
            tree_method="hist",
            device=XGB_DEVICE,
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
                # Robust fallback: if Huber fails to converge on a fold, keep zero residual for that fold.
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

    if USE_SINGLE_RESIDUAL_BRANCH and len(residual_feats) > 0:
        groups = df_tr["Battery_ID"].values
        r_base_tr = y_tr - yb_tr

        sc = StandardScaler()
        Xtr = sc.fit_transform(df_tr[residual_feats].values)
        Xte = sc.transform(df_te[residual_feats].values)

        n_splits = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)

        oof_branch = np.zeros(len(df_tr), dtype=float)
        pred_branch = np.zeros(len(df_te), dtype=float)

        for fold, (ti, vi) in enumerate(gkf.split(Xtr, r_base_tr, groups=groups)):
            model = _make_single_residual_model(params, seed + fold * 17, fast_mode=fast_mode)
            model.fit(Xtr[ti], r_base_tr[ti])
            oof_branch[vi] = model.predict(Xtr[vi])
            pred_branch += model.predict(Xte) / n_splits

        gate = float(params.lambda_ET)
        y_branch_raw_tr = np.clip(yb_tr + oof_branch, 0.0, max_clip)
        y_branch_raw_te = np.clip(yb_te + pred_branch, 0.0, max_clip)
        y_gated_tr = np.clip(yb_tr + gate * oof_branch, 0.0, max_clip)
        y_gated_te = np.clip(yb_te + gate * pred_branch, 0.0, max_clip)

        branch_name = SINGLE_RESIDUAL_BRANCH_KIND
        indiv_preds = {
            f"{branch_name}_Raw": y_branch_raw_te,
            f"{branch_name}_Gated": y_gated_te,
        }

        return {
            "y_gated_tr": y_gated_tr,
            "y_gated_te": y_gated_te,
            "gate_weights": np.array([gate], dtype=float),
            "gate_names": [f"{branch_name}_Gate"],
            "indiv_preds": indiv_preds,
            "residual_feats": residual_feats,
            "branch_oof": oof_branch,
            "branch_test": pred_branch,
        }

    if (not USE_LEGACY_RESIDUAL_BANK) or len(residual_feats) == 0:
        return {
            "y_gated_tr": yb_tr.copy(),
            "y_gated_te": yb_te.copy(),
            "gate_weights": np.zeros(0),
            "gate_names": [],
            "indiv_preds": {},
            "residual_feats": [],
        }

    groups = df_tr["Battery_ID"].values
    r_base_tr = y_tr - yb_tr

    sc = StandardScaler()
    Xtr = sc.fit_transform(df_tr[residual_feats].values)
    Xte = sc.transform(df_te[residual_feats].values)

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    oof_xgb = np.zeros(len(df_tr))
    oof_et = np.zeros(len(df_tr))
    oof_lin = np.zeros(len(df_tr))

    pred_xgb = np.zeros(len(df_te))
    pred_et = np.zeros(len(df_te))
    pred_lin = np.zeros(len(df_te))

    n_bags = 1 if fast_mode else 2
    xgb_estimators = 180 if fast_mode else 1000
    et_estimators = 60 if fast_mode else 200

    for fold, (ti, vi) in enumerate(gkf.split(Xtr, r_base_tr, groups=groups)):
        oof_xgb_fold = np.zeros(len(vi))
        pred_xgb_fold = np.zeros(len(df_te))

        for bag in range(n_bags):
            clf_xgb = xgb.XGBRegressor(
                n_estimators=xgb_estimators,
                learning_rate=params.lgbm_learning_rate,
                max_depth=int(params.lgbm_max_depth),
                subsample=0.7,
                colsample_bytree=0.7,
                reg_lambda=params.lgbm_reg_lambda,
                objective="reg:pseudohubererror",
                tree_method="hist",
                device=XGB_DEVICE,
                n_jobs=-1,
                random_state=seed + fold * 17 + bag,
            )
            clf_xgb.fit(Xtr[ti], r_base_tr[ti], verbose=False)
            oof_xgb_fold += clf_xgb.predict(Xtr[vi]) / n_bags
            pred_xgb_fold += clf_xgb.predict(Xte) / (n_splits * n_bags)

        oof_xgb[vi] = oof_xgb_fold
        pred_xgb += pred_xgb_fold

        clf_et = ExtraTreesRegressor(
            n_estimators=et_estimators,
            max_depth=int(params.et_depth),
            min_samples_leaf=int(params.et_leaf),
            n_jobs=-1,
            random_state=seed + fold,
        )
        clf_et.fit(Xtr[ti], r_base_tr[ti])
        oof_et[vi] = clf_et.predict(Xtr[vi])
        pred_et += clf_et.predict(Xte) / n_splits

        clf_lin = BayesianRidge()
        clf_lin.fit(Xtr[ti], r_base_tr[ti])
        oof_lin[vi] = clf_lin.predict(Xtr[vi])
        pred_lin += clf_lin.predict(Xte) / n_splits

    P_tr = np.column_stack([oof_xgb, oof_et, oof_lin, np.ones(len(df_tr))])
    P_te = np.column_stack([pred_xgb, pred_et, pred_lin, np.ones(len(df_te))])

    w, _ = _legacy_nnls(P_tr, r_base_tr)

    y_gated_tr = np.clip(yb_tr + P_tr @ w, 0.0, max_clip)
    y_gated_te = np.clip(yb_te + P_te @ w, 0.0, max_clip)

    indiv_preds = {
        "XGB": np.clip(yb_te + pred_xgb, 0.0, max_clip),
        "ExtraTrees": np.clip(yb_te + pred_et, 0.0, max_clip),
        "BayesianRidge": np.clip(yb_te + pred_lin, 0.0, max_clip),
        "LegacyResidualEnsemble": y_gated_te,
    }

    return {
        "y_gated_tr": y_gated_tr,
        "y_gated_te": y_gated_te,
        "gate_weights": w,
        "gate_names": ["XGB", "ExtraTrees", "BayesianRidge", "Bias"],
        "indiv_preds": indiv_preds,
        "residual_feats": residual_feats,
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
    kan_tr, kan_te = train_physics_kan_refiner(
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
    }




# =============================================================================
# Extra reporting helpers for Chapter 4
# =============================================================================
def add_stage_metric_rows(rows, protocol, fold, stage, y_train, pred_train, y_test, pred_test, extra=None):
    """Append Train/Test RMSE, MAE, MSE and R2 for one modelling stage."""
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


def save_stage_summary_tables(stage_df: pd.DataFrame, out_dir: str, prefix: str):
    safe_to_csv(stage_df, os.path.join(out_dir, f"{prefix}_stage_metrics_long.csv"))
    summary = summarize_stage_metrics(stage_df)
    safe_to_csv(summary, os.path.join(out_dir, f"{prefix}_stage_metrics_summary.csv"))
    return summary

# =============================================================================
# 7. QGWO tuning
# =============================================================================
def params_from_array(arr: np.ndarray, init_params: Params) -> Params:
    if USE_QGWO_ENSEMBLE_BRANCHES:
        return Params(
            base_alpha=float(arr[0]),
            lgbm_max_depth=init_params.lgbm_max_depth,
            lgbm_learning_rate=init_params.lgbm_learning_rate,
            lgbm_reg_lambda=init_params.lgbm_reg_lambda,
            et_depth=init_params.et_depth,
            et_leaf=init_params.et_leaf,
            lambda_ET=float(arr[1]),       # ExtraTrees gate
            lambda_LGBM=float(arr[2]),
            lambda_Huber=float(arr[3]),
            lambda_KAN=float(arr[4]),
        )
    if USE_SINGLE_RESIDUAL_BRANCH:
        return Params(
            base_alpha=float(arr[0]),
            lgbm_max_depth=init_params.lgbm_max_depth,
            lgbm_learning_rate=init_params.lgbm_learning_rate,
            lgbm_reg_lambda=init_params.lgbm_reg_lambda,
            et_depth=int(np.round(arr[1])),
            et_leaf=int(np.round(arr[2])),
            lambda_ET=float(arr[3]),
            lambda_KAN=float(arr[4]),
        )
    if USE_LEGACY_RESIDUAL_BANK:
        return Params(
            base_alpha=float(arr[0]),
            lgbm_max_depth=int(np.round(arr[1])),
            lgbm_learning_rate=float(arr[2]),
            lgbm_reg_lambda=float(arr[3]),
            et_depth=int(np.round(arr[4])),
            et_leaf=int(np.round(arr[5])),
            lambda_ET=init_params.lambda_ET,
            lambda_KAN=float(arr[6]),
        )

    return Params(
        base_alpha=float(arr[0]),
        lgbm_max_depth=init_params.lgbm_max_depth,
        lgbm_learning_rate=init_params.lgbm_learning_rate,
        lgbm_reg_lambda=init_params.lgbm_reg_lambda,
        et_depth=init_params.et_depth,
        et_leaf=init_params.et_leaf,
        lambda_ET=init_params.lambda_ET,
        lambda_KAN=float(arr[1]),
    )


def make_qgwo_bounds(init_params: Params):
    if USE_QGWO_ENSEMBLE_BRANCHES:
        # Keep this first replacement test clean: tune gates and KAN weight only.
        # Model hyperparameters are fixed to avoid turning this into a broad AutoML search.
        return [
            (10.0, 18.0),   # base_alpha
            (0.00, 0.35),   # ExtraTrees gate
            (0.00, 0.35),   # LightGBM gate
            (0.00, 0.20),   # HuberRegressor gate
            (0.70, 1.30),   # KAN weight
        ]
    if USE_SINGLE_RESIDUAL_BRANCH:
        center = np.array([init_params.base_alpha, init_params.et_depth, init_params.et_leaf, init_params.lambda_ET, init_params.lambda_KAN], dtype=float)
        # Main hypothesis test:
        # QGWO directly searches the branch gate and KAN weight against full_causal_pims.
        spans = np.array([3.0, 2.0, 3.0, 0.15, 0.25], dtype=float)
        bounds = [(max(1e-3, c - s), c + s) for c, s in zip(center, spans)]
        bounds[0] = (10.0, 18.0)   # base_alpha
        bounds[1] = (5.0, 12.0)    # et_depth
        bounds[2] = (5.0, 20.0)    # et_leaf
        bounds[3] = (0.0, 0.45)    # ExtraTrees residual gate
        bounds[4] = (0.70, 1.30)   # KAN weight
        return bounds
    if USE_LEGACY_RESIDUAL_BANK:
        center = np.array([
            init_params.base_alpha,
            init_params.lgbm_max_depth,
            init_params.lgbm_learning_rate,
            init_params.lgbm_reg_lambda,
            init_params.et_depth,
            init_params.et_leaf,
            init_params.lambda_KAN,
        ], dtype=float)

        spans = np.array([3.0, 1.0, 0.004, 8.0, 2.0, 3.0, 0.20], dtype=float)
        bounds = [(max(1e-3, c - s), c + s) for c, s in zip(center, spans)]
        bounds[-1] = (0.4, 1.2)
        return bounds

    center = np.array([init_params.base_alpha, init_params.lambda_KAN], dtype=float)
    spans = np.array([3.0, 0.20], dtype=float)
    bounds = [(max(1e-3, c - s), c + s) for c, s in zip(center, spans)]
    bounds[-1] = (0.4, 1.2)
    return bounds


def tune_qgwo_train_only(
    df_train: pd.DataFrame,
    residual_feats: List[str],
    baseline_feats: List[str],
    kan_feats: List[str],
    seed: int,
    init_params: Params,
    objective_track: str = QGWO_OBJECTIVE_TRACK,
) -> Tuple[Params, float, pd.DataFrame]:

    bounds = make_qgwo_bounds(init_params)

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
# 8. Experiment runners
# =============================================================================
def collect_ablation_rows(y_te, base_pred, gated_residual_pred, qgwo_gated_residual_pred, full_pred, causal_full_pred) -> Dict[str, float]:
    return {
        "Baseline_RMSE": calc_metrics(y_te, base_pred)["RMSE"],
        "ResidualEnsemble_RMSE": calc_metrics(y_te, gated_residual_pred)["RMSE"],
        "QGWO_GatedResidual_RMSE": calc_metrics(y_te, qgwo_gated_residual_pred)["RMSE"],
        "QGWO_GatedResidual_PhysicsKAN_RMSE": calc_metrics(y_te, full_pred)["RMSE"],
        "QPEAK_Final_CausalPIMS_RMSE": calc_metrics(y_te, causal_full_pred)["RMSE"],
    }


def run_five_seeds(raw_with_id: pd.DataFrame, default_params: Params, out_dir: str) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"[RUN] 5-seed Group Split | {MODEL_VARIANT}")
    print("=" * 80)

    batteries = np.array(sorted(raw_with_id["Battery_ID"].unique()))

    pre_test, off_test, causal_test = init_metric_dict(), init_metric_dict(), init_metric_dict()

    seed_rows, split_rows, ablation_rows, gate_rows, pims_rows = [], [], [], [], []
    stage_metric_rows = []
    prediction_records, qgwo_history_records = [], []

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

        tuned_params, qgwo_obj, qhist = default_params, None, pd.DataFrame()

        if ENABLE_QGWO_5SEED:
            tuned_params, qgwo_obj, qhist = tune_qgwo_train_only(
                dtr,
                residual_feats,
                baseline_feats,
                kan_feats,
                s,
                default_params,
                objective_track=QGWO_OBJECTIVE_TRACK,
            )

            if SAVE_QGWO_HISTORY:
                qhist["Seed"] = s
                qgwo_history_records.append(qhist)

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
            "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES),
            "Seed": s,
            "Train_Batteries": ";".join([f"B{x+1:02d}" for x in sorted(train_batts)]),
            "Test_Batteries": ";".join([f"B{x+1:02d}" for x in sorted(test_batts)]),
            "PrePIMS_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
            "CausalPIMS_MAE": m_cau["MAE"],
            "CausalPIMS_R2": m_cau["R2"],
            "QGWO_Obj": qgwo_obj,
            "N_Baseline_Features": len(baseline_feats),
            "N_Residual_Features": len(residual_feats),
            "N_KAN_Features": len(kan_feats),
            **{f"Param_{k}": v for k, v in asdict(tuned_params).items()},
        })

        split_rows.append({
            "Seed": s,
            "Train_Batteries": seed_rows[-1]["Train_Batteries"],
            "Test_Batteries": seed_rows[-1]["Test_Batteries"],
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

        # Chapter 4 long-format Train/Test metrics for each modelling stage
        stage_extra = {"Seed": s, "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES)}
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

        gate_row = {"Fold": f"Seed_{s}", "Seed": s, "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES)}
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
            "CausalPIMS_Violations_Raw": monotonic_violation_count(dte, cau_raw),
            "CausalPIMS_Violations": monotonic_violation_count(dte, cau),
            "Pre_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE_Raw": m_cau_raw["RMSE"],
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

        print(f"seed={s} | pre={m_pre['RMSE']:.4f} | offline={m_off['RMSE']:.4f} | causal={m_cau['RMSE']:.4f}")

    seed_df = pd.DataFrame(seed_rows)
    split_df = pd.DataFrame(split_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    gate_df = pd.DataFrame(gate_rows)
    pims_df = pd.DataFrame(pims_rows)
    stage_df = pd.DataFrame(stage_metric_rows)
    stage_summary_df = save_stage_summary_tables(stage_df, out_dir, "group_split_5seed")

    safe_to_csv(seed_df, os.path.join(out_dir, "group_split_5seed_summary.csv"))
    safe_to_csv(split_df, os.path.join(out_dir, "group_split_5seed_assignments.csv"))
    safe_to_csv(ablation_df, os.path.join(out_dir, "group_split_5seed_stage_ablation.csv"))
    safe_to_csv(gate_df, os.path.join(out_dir, "group_split_5seed_qgwo_gate_weights.csv"))
    safe_to_csv(pims_df, os.path.join(out_dir, "group_split_5seed_pims_diagnostics.csv"))

    if SAVE_PREDICTIONS and prediction_records:
        safe_to_csv(pd.concat(prediction_records, ignore_index=True), os.path.join(out_dir, "group_split_5seed_predictions.csv"))

    if SAVE_QGWO_HISTORY and qgwo_history_records:
        safe_to_csv(pd.concat(qgwo_history_records, ignore_index=True), os.path.join(out_dir, "group_split_5seed_qgwo_history.csv"))

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



# =============================================================================
# 8b. LOBO train-only post calibration
#     These calibrators are report/prediction-time only. They use outer-train
#     batteries only and never inspect the held-out LOBO battery labels.
# =============================================================================
TRAIN_ONLY_POST_CALIBRATION_ENABLED = True
POST_CALIBRATION_MODE = "TrainOnlyRobustResidualCurve"
POST_CALIBRATION_MAX_ABS = 10.0
POST_CALIBRATION_SHRINK = 0.6


def _weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if sample_weight is None:
        return np.quantile(values, quantiles)
    sample_weight = np.asarray(sample_weight, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight)
    cdf = cdf / (cdf[-1] + 1e-12)
    return np.interp(quantiles, cdf, values)


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

    # Smooth adjacent offsets to avoid noisy bin jumps.
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
    """Fit train-only calibration object from outer-train predictions."""
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    pred_train = np.asarray(pred_train, dtype=float).reshape(-1)
    resid = y_train - pred_train

    if not TRAIN_ONLY_POST_CALIBRATION_ENABLED:
        return {"enabled": False, "mode": "none"}

    if mode == "ConservativeResidualCurve":
        cal = _fit_piecewise_residual_calibrator(
            pred_train,
            resid,
            n_bins=8,
            max_abs=POST_CALIBRATION_MAX_ABS,
            shrink=POST_CALIBRATION_SHRINK,
        )
        cal.update({"enabled": True, "mode": mode})
        return cal

    if mode == "TrainOnlyRobustResidualCurve":
        # Residual curve plus an upper-RUL tail term.
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

        used_params, qgwo_obj = default_params, None

        if ENABLE_QGWO_LOBO:
            used_params, qgwo_obj, _ = tune_qgwo_train_only(
                dtr,
                residual_feats,
                baseline_feats,
                kan_feats,
                seed,
                default_params,
                objective_track=QGWO_OBJECTIVE_TRACK,
            )

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
            "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES),
            "Battery_ID": int(b),
            "Battery_Label": label,
            "PrePIMS_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE_Raw": m_cau_raw["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
            "CausalPIMS_MAE": m_cau["MAE"],
            "CausalPIMS_R2": m_cau["R2"],
            "TrainOnly_PostCalibration_Mode": POST_CALIBRATION_MODE,
            "TrainOnly_PostCalibration_MeanCorrection": float(np.mean(post_corr)),
            "TrainOnly_PostCalibration_MaxAbsCorrection": float(np.max(np.abs(post_corr))) if len(post_corr) else 0.0,
            "TrainOnly_PostCalibration_Improvement_RMSE": m_cau_raw["RMSE"] - m_cau["RMSE"],
            "Hard_Case_Label": bool(int(b) in HARD_BATTERY_IDS),
            "QGWO_Obj": qgwo_obj,
        })

        # Chapter 4 long-format Train/Test metrics for LOBO stages
        stage_extra = {"Battery_ID": int(b), "Battery_Label": label, "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES)}
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "01_Baseline_default",
            out["y_tr"], out["tracks"]["baseline_offline_pims"][0],
            out["y_te"], out["tracks"]["baseline_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "02_ResidualEnsemble_default",
            out["y_tr"], out["tracks"]["gated_residual_offline_pims"][0],
            out["y_te"], out["tracks"]["gated_residual_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "03_Full_PrePIMS",
            out["y_tr"], out["tracks"]["full_pre"][0],
            out["y_te"], out["tracks"]["full_pre"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "04_Full_OfflinePIMS",
            out["y_tr"], out["tracks"]["full_offline_pims"][0],
            out["y_te"], out["tracks"]["full_offline_pims"][1],
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "05_Full_CausalPIMS_Raw",
            out["y_tr"], out["tracks"]["full_causal_pims"][0],
            out["y_te"], cau_raw,
            stage_extra,
        )
        add_stage_metric_rows(
            stage_metric_rows, "LOBO", label, "06_QPEAK_Final_PostCalibrated_CausalPIMS",
            out["y_tr"], out["tracks"]["full_causal_pims"][0],
            out["y_te"], cau,
            stage_extra,
        )

        gate_row = {"Fold": label, "Battery_ID": int(b), "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES)}
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
            "CausalPIMS_Violations_Raw": monotonic_violation_count(dte, cau_raw),
            "CausalPIMS_Violations": monotonic_violation_count(dte, cau),
            "Pre_RMSE": m_pre["RMSE"],
            "OfflinePIMS_RMSE": m_off["RMSE"],
            "CausalPIMS_RMSE_Raw": m_cau_raw["RMSE"],
            "CausalPIMS_RMSE": m_cau["RMSE"],
        })

        if SAVE_PREDICTIONS:
            tmp = dte[["Battery_ID", "Cycle_Index", "RUL"]].copy()
            tmp["Fold"] = label
            tmp["True_RUL"] = y_te
            tmp["Pred_PrePIMS"] = pre
            tmp["Pred_OfflinePIMS"] = off
            tmp["Pred_CausalPIMS_Raw"] = cau_raw
            tmp["Pred_CausalPIMS"] = cau
            tmp["TrainOnly_PostCalibration_Correction"] = post_corr
            prediction_records.append(tmp)

        print(f"{label} | pre={m_pre['RMSE']:.4f} | offline={m_off['RMSE']:.4f} | causal_raw={m_cau_raw['RMSE']:.4f} | causal_cal={m_cau['RMSE']:.4f}")

    rows_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    pims_df = pd.DataFrame(pims_rows)
    stage_df = pd.DataFrame(stage_metric_rows)
    stage_summary_df = save_stage_summary_tables(stage_df, out_dir, "lobo")

    safe_to_csv(rows_df, os.path.join(out_dir, "lobo_summary.csv"))
    safe_to_csv(gate_df, os.path.join(out_dir, "lobo_qgwo_gate_weights.csv"))
    safe_to_csv(pims_df, os.path.join(out_dir, "lobo_pims_diagnostics.csv"))

    if SAVE_PREDICTIONS and prediction_records:
        safe_to_csv(pd.concat(prediction_records, ignore_index=True), os.path.join(out_dir, "lobo_predictions.csv"))

    vals = rows_df["CausalPIMS_RMSE"].values.astype(float)
    robust_rows = [
        {
            "Model_Variant": MODEL_VARIANT,
            "Protocol": "All_LOBO",
            "N": len(vals),
            "RMSE_Mean": np.mean(vals),
            "RMSE_Std": np.std(vals),
            "RMSE_Median": np.median(vals),
        }
    ]

    if "B05" in set(rows_df["Battery_Label"]):
        vals_wo = rows_df.loc[rows_df["Battery_Label"] != "B05", "CausalPIMS_RMSE"].values.astype(float)
        robust_rows.append({
            "Model_Variant": MODEL_VARIANT,
            "Protocol": "LOBO_excluding_B05",
            "N": len(vals_wo),
            "RMSE_Mean": np.mean(vals_wo),
            "RMSE_Std": np.std(vals_wo),
            "RMSE_Median": np.median(vals_wo),
        })

    safe_to_csv(pd.DataFrame(robust_rows), os.path.join(out_dir, "lobo_robustness_summary.csv"))

    return {
        "pre_test": pre_test,
        "off_test": off_test,
        "causal_test": causal_test,
        "rows": rows_df,
        "gate_df": gate_df,
        "pims_df": pims_df,
        "stage_df": stage_df,
        "stage_summary_df": stage_summary_df,
    }


# =============================================================================
# 9. Summaries and figures
# =============================================================================
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


def make_essential_figures(five_res: Dict[str, Any], lobo_res: Dict[str, Any], ablation_summary: pd.DataFrame, out_dir: str):

    seed_df = five_res["seed_df"]
    plt.figure(figsize=(8, 5))
    plt.plot(seed_df["Seed"], seed_df["PrePIMS_RMSE"], marker="o", label="Pre-PIMS")
    plt.plot(seed_df["Seed"], seed_df["OfflinePIMS_RMSE"], marker="s", label="Offline PIMS")
    plt.plot(seed_df["Seed"], seed_df["CausalPIMS_RMSE"], marker="^", label="Causal PIMS")
    plt.xlabel("Seed")
    plt.ylabel("Test RMSE")
    plt.title(f"5-seed Group Split RMSE - {MODEL_VARIANT}")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_group_split_5seed_rmse_summary.png"), dpi=220)
    plt.close()

    lobo_df = lobo_res["rows"]
    vals = lobo_df["CausalPIMS_RMSE"].values
    x = np.arange(len(lobo_df))

    plt.figure(figsize=(11, 5))
    plt.bar(x, vals, edgecolor="black")
    plt.axhline(np.mean(vals), linestyle="--", label=f"Mean={np.mean(vals):.2f}")
    plt.axhline(np.median(vals), linestyle=":", label=f"Median={np.median(vals):.2f}")
    plt.xticks(x, lobo_df["Battery_Label"].values)
    plt.xlabel("Battery")
    plt.ylabel("RMSE")
    plt.title(f"LOBO RMSE by Battery - {MODEL_VARIANT}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_lobo_rmse_by_battery.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(ablation_summary))
    plt.bar(x, ablation_summary["Mean_RMSE"].values, edgecolor="black")
    for i, v in enumerate(ablation_summary["Mean_RMSE"].values):
        plt.text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(x, ablation_summary["Stage"].values, rotation=20, ha="right")
    plt.ylabel("Average Test RMSE")
    plt.title(f"Stage Ablation - {MODEL_VARIANT}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_stage_ablation_rmse.png"), dpi=220)
    plt.close()

    # Fig: train/test RMSE by stage for 5-seed
    try:
        st = five_res.get("stage_summary_df", pd.DataFrame())
        st = st[(st["Protocol"] == "FiveSeed") & (st["Set"].isin(["Train", "Test"]))].copy()
        if not st.empty:
            stages = list(st["Stage"].drop_duplicates())
            x = np.arange(len(stages))
            width = 0.38
            train_vals = [st[(st["Stage"] == s) & (st["Set"] == "Train")]["RMSE_Mean"].mean() for s in stages]
            test_vals = [st[(st["Stage"] == s) & (st["Set"] == "Test")]["RMSE_Mean"].mean() for s in stages]
            plt.figure(figsize=(12, 5))
            plt.bar(x - width / 2, train_vals, width, edgecolor="black", label="Train")
            plt.bar(x + width / 2, test_vals, width, edgecolor="black", label="Test")
            plt.xticks(x, stages, rotation=25, ha="right")
            plt.ylabel("Mean RMSE")
            plt.title(f"Train/Test RMSE by Stage - {MODEL_VARIANT}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_group_split_5seed_stage_train_test_rmse.png"), dpi=220)
            plt.close()
    except Exception as e:
        print(f"[WARN] stage train/test figure failed: {e}")

    # Fig: PIMS violation comparison
    try:
        pv_rows = []
        for protocol, dfp in [("FiveSeed", five_res.get("pims_df", pd.DataFrame())), ("LOBO", lobo_res.get("pims_df", pd.DataFrame()))]:
            if not dfp.empty:
                pv_rows.append({"Protocol": protocol, "Track": "Pre-PIMS", "Violations": dfp["Pre_Violations"].mean()})
                pv_rows.append({"Protocol": protocol, "Track": "Offline PIMS", "Violations": dfp["OfflinePIMS_Violations"].mean()})
                pv_rows.append({"Protocol": protocol, "Track": "Causal PIMS", "Violations": dfp["CausalPIMS_Violations"].mean()})
        pv = pd.DataFrame(pv_rows)
        safe_to_csv(pv, os.path.join(out_dir, "pims_violation_summary.csv"))
        if not pv.empty:
            labels = [f"{r.Protocol}\n{r.Track}" for r in pv.itertuples()]
            plt.figure(figsize=(9, 5))
            plt.bar(np.arange(len(pv)), pv["Violations"].values, edgecolor="black")
            plt.xticks(np.arange(len(pv)), labels, rotation=20, ha="right")
            plt.ylabel("Mean violation count")
            plt.title("Monotonicity Violations Before/After PIMS")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_pims_diagnostics_summary.png"), dpi=220)
            plt.close()
    except Exception as e:
        print(f"[WARN] PIMS violation figure failed: {e}")


def save_protocol_audit(raw: pd.DataFrame, raw_with_id: pd.DataFrame, out_dir: str):
    rows = [
        {"Item": "Model variant", "Value": MODEL_VARIANT},
        {"Item": "Use legacy residual ensemble", "Value": bool(USE_LEGACY_RESIDUAL_BANK)},
        {"Item": "Use single residual learner", "Value": bool(USE_SINGLE_RESIDUAL_BRANCH)},
        {"Item": "Use QGWO-gated residual ensemble", "Value": bool(USE_QGWO_ENSEMBLE_BRANCHES)},
        {"Item": "Ensemble branch kinds", "Value": ";".join(ENSEMBLE_BRANCH_KINDS)},
        {"Item": "Single residual branch kind", "Value": SINGLE_RESIDUAL_BRANCH_KIND},
        {"Item": "Raw rows", "Value": int(len(raw))},
        {"Item": "Raw columns", "Value": int(raw.shape[1])},
        {"Item": "Inferred batteries", "Value": int(raw_with_id["Battery_ID"].nunique())},
        {"Item": "Causal Tau", "Value": bool(USE_CAUSAL_TAU_IN_MAIN)},
        {"Item": "Hard battery local refine", "Value": bool(ENABLE_HARD_BATTERY_LOCAL_REFINE)},
        {"Item": "HGB included", "Value": False},
        {"Item": "QGWO objective track", "Value": QGWO_OBJECTIVE_TRACK},
        {"Item": "QGWO tunes lambda_KAN", "Value": True},
        {"Item": "Baseline features", "Value": ";".join(CLOCK_BASELINE_FEATURES)},
        {"Item": "Residual ensemble", "Value": "Legacy static ensemble disabled; QGWO-gated residual ensemble branches: " + ";".join(ENSEMBLE_BRANCH_KINDS)},
        {"Item": "Main final output", "Value": "Baseline/QGWO + gated ET/LGBM/Huber residual ensemble + Tau-gated PhysicsKAN + causal PIMS"},
    ]

    safe_to_csv(pd.DataFrame(rows), os.path.join(out_dir, "protocol_audit.csv"))

    bsum = raw_with_id.groupby("Battery_ID").agg(
        N=("RUL", "count"),
        Cycle_Min=("Cycle_Index", "min"),
        Cycle_Max=("Cycle_Index", "max"),
        RUL_Min=("RUL", "min"),
        RUL_Max=("RUL", "max"),
    ).reset_index()

    bsum["Battery_Label"] = bsum["Battery_ID"].apply(lambda x: f"B{int(x)+1:02d}")
    safe_to_csv(bsum, os.path.join(out_dir, "battery_summary.csv"))



# =============================================================================
# 10. Post-hoc reports for Chapter 4
#     Report-only outputs. These functions do NOT modify training, tuning, model
#     prediction, or PIMS. They only read finished outputs and create extra CSVs
#     and figures for Chapter 4 / appendix.
# =============================================================================
def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def calc_extra_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    base = calc_metrics(y_true, y_pred)
    y_range = float(np.max(y_true) - np.min(y_true))
    y_mean = float(np.mean(y_true))
    base["NRMSE_range"] = float(base["RMSE"] / (y_range + 1e-9))
    base["NRMSE_mean"] = float(base["RMSE"] / (abs(y_mean) + 1e-9))
    base["MAPE_percent_safe"] = safe_mape(y_true, y_pred)
    return base


def save_prediction_extra_metrics(out_dir: str):
    """
    Add NRMSE and safe MAPE from saved prediction files.
    This is report-only and does not alter model results.
    """
    rows = []
    files = [
        ("FiveSeed", os.path.join(out_dir, "group_split_5seed_predictions.csv")),
        ("LOBO", os.path.join(out_dir, "lobo_predictions.csv")),
    ]
    tracks = [
        ("PrePIMS", "Pred_PrePIMS"),
        ("OfflinePIMS", "Pred_OfflinePIMS"),
        ("CausalPIMS", "Pred_CausalPIMS"),
    ]

    for protocol, path in files:
        if not os.path.exists(path):
            rows.append({
                "Protocol": protocol,
                "Warning": f"{os.path.basename(path)} not found. Set SAVE_PREDICTIONS=True."
            })
            continue

        df = pd.read_csv(path)
        for track_name, col in tracks:
            if col not in df.columns:
                continue
            m = calc_extra_metrics(df["True_RUL"].values, df[col].values)
            rows.append({"Protocol": protocol, "Group": "ALL", "Track": track_name, **m})

        group_col = "Seed" if protocol == "FiveSeed" else "Fold"
        if group_col in df.columns:
            for group_value, g in df.groupby(group_col):
                for track_name, col in tracks:
                    if col not in g.columns:
                        continue
                    m = calc_extra_metrics(g["True_RUL"].values, g[col].values)
                    rows.append({
                        "Protocol": protocol,
                        "Group": group_value,
                        "Track": track_name,
                        **m,
                    })

    out = pd.DataFrame(rows)
    safe_to_csv(out, os.path.join(out_dir, "extra_metric_summary_nrmse_mape.csv"))
    return out


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str], title: str, save_path: str, max_cols: int = 30):
    """Simple matplotlib correlation heatmap without seaborn."""
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        safe_to_csv(
            pd.DataFrame([{"Warning": "Not enough columns for correlation heatmap."}]),
            save_path.replace(".png", "_warning.csv"),
        )
        return

    if len(cols) > max_cols:
        cols = cols[:max_cols]

    corr = df[cols].corr(numeric_only=True)
    safe_to_csv(corr.reset_index().rename(columns={"index": "Feature"}), save_path.replace(".png", ".csv"))

    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(cols)), cols, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_correlation_reports(raw_with_id: pd.DataFrame, out_dir: str):
    """
    Add raw and engineered feature correlation heatmaps.
    These figures are descriptive reports only and are not used in training.
    """
    raw_num_cols = [
        c for c in raw_with_id.select_dtypes(include=[np.number]).columns
        if c not in ["Battery_ID"]
    ]

    plot_correlation_heatmap(
        raw_with_id,
        raw_num_cols,
        "Raw Feature Correlation Heatmap",
        os.path.join(out_dir, "fig_raw_feature_correlation_heatmap.png"),
        max_cols=25,
    )

    # Full-data engineering is used only for descriptive analysis. It is not fed
    # back into model training or evaluation.
    df_desc = engineer_features(apply_cleaning(raw_with_id, params=None))
    preferred_cols = [
        "RUL",
        "Cycle_Index", "Cycle_log",
        "Cap", "IR", "CV_Ratio", "CC_Time", "Vdrop",
        "Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "CC_Time_EMA", "Vdrop_EMA",
        "Cap_v1", "IR_v1", "CV_Ratio_v1", "Vdrop_v1",
        "CC_Ratio_EMA", "CC_minus_CV_EMA",
        "Tau", "Tau_rm20", "Tau_rs20", "Tau_slope20",
        "Cum_Ah_log1p", "Cum_Energy_log1p",
    ]
    preferred_cols = [c for c in preferred_cols if c in df_desc.columns]

    plot_correlation_heatmap(
        df_desc,
        preferred_cols,
        "Engineered Physics Feature Correlation Heatmap",
        os.path.join(out_dir, "fig_engineered_feature_correlation_heatmap.png"),
        max_cols=30,
    )

    corr = df_desc[preferred_cols].corr(numeric_only=True)
    if "RUL" in corr.columns:
        top = (
            corr["RUL"]
            .drop(labels=["RUL"], errors="ignore")
            .sort_values(key=lambda s: np.abs(s), ascending=False)
            .reset_index()
        )
        top.columns = ["Feature", "PearsonCorrWithRUL"]
        safe_to_csv(top, os.path.join(out_dir, "correlation_top_features_with_rul.csv"))


def plot_prediction_trajectories(out_dir: str):
    """
    Plot hard-case LOBO trajectories and residual diagnostics.
    Requires SAVE_PREDICTIONS=True.
    """
    path = os.path.join(out_dir, "lobo_predictions.csv")
    if not os.path.exists(path):
        safe_to_csv(
            pd.DataFrame([{"Warning": "lobo_predictions.csv not found. Set SAVE_PREDICTIONS=True."}]),
            os.path.join(out_dir, "trajectory_plot_warning.csv"),
        )
        return

    df = pd.read_csv(path)
    if "Fold" not in df.columns:
        safe_to_csv(
            pd.DataFrame([{"Warning": "Fold column not found in lobo_predictions.csv."}]),
            os.path.join(out_dir, "trajectory_plot_warning.csv"),
        )
        return

    target_folds = ["B05", "B13", "B14", "B01"]
    for fold in target_folds:
        g = df[df["Fold"] == fold].copy()
        if g.empty:
            continue
        g = g.sort_values("Cycle_Index")

        plt.figure(figsize=(10, 5))
        plt.plot(g["Cycle_Index"], g["True_RUL"], label="True RUL", linewidth=2)
        plt.plot(g["Cycle_Index"], g["Pred_PrePIMS"], label="Pre-PIMS", linewidth=1.6)
        plt.plot(g["Cycle_Index"], g["Pred_CausalPIMS"], label="Causal PIMS", linewidth=1.6)
        plt.xlabel("Cycle Index")
        plt.ylabel("RUL")
        plt.title(f"LOBO Prediction Trajectory - {fold}")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_lobo_{fold}_trajectory.png"), dpi=220)
        plt.close()

    # Parity / residual diagnostics for appendix.
    if {"True_RUL", "Pred_CausalPIMS"}.issubset(df.columns):
        plt.figure(figsize=(6, 6))
        plt.scatter(df["True_RUL"], df["Pred_CausalPIMS"], s=6, alpha=0.35)
        min_v = float(min(df["True_RUL"].min(), df["Pred_CausalPIMS"].min()))
        max_v = float(max(df["True_RUL"].max(), df["Pred_CausalPIMS"].max()))
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5)
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title("LOBO True vs Predicted RUL - Causal PIMS")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_lobo_true_vs_pred_scatter.png"), dpi=220)
        plt.close()

        resid = df["True_RUL"].values - df["Pred_CausalPIMS"].values
        plt.figure(figsize=(8, 5))
        plt.hist(resid, bins=50, edgecolor="black")
        plt.xlabel("Residual = True RUL - Predicted RUL")
        plt.ylabel("Count")
        plt.title("LOBO Residual Distribution - Causal PIMS")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_lobo_residual_histogram.png"), dpi=220)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(df["Pred_CausalPIMS"], resid, s=6, alpha=0.35)
        plt.axhline(0, linestyle="--", linewidth=1.5)
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual")
        plt.title("LOBO Residual vs Predicted RUL - Causal PIMS")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_lobo_residual_vs_predicted.png"), dpi=220)
        plt.close()


def plot_qgwo_gate_weight_figures(five_res: Dict[str, Any], lobo_res: Dict[str, Any], out_dir: str):
    """Plot QGWO branch-gate weights for appendix-level diagnostics."""
    for name, df in [
        ("5seeds", five_res.get("gate_df", pd.DataFrame())),
        ("lobo", lobo_res.get("gate_df", pd.DataFrame())),
    ]:
        if df.empty:
            continue
        weight_cols = [c for c in df.columns if c.startswith("W_")]
        if not weight_cols:
            continue

        plot_df = df.copy()
        x = np.arange(len(plot_df))
        bottom = np.zeros(len(plot_df))
        plt.figure(figsize=(11, 5))
        for c in weight_cols:
            vals = plot_df[c].fillna(0).values.astype(float)
            plt.bar(x, vals, bottom=bottom, label=c.replace("W_", ""), edgecolor="black")
            bottom += vals
        labels = plot_df["Fold"].values if "Fold" in plot_df.columns else x
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("QGWO Gate Weight")
        plt.title(f"QGWO Gate Weights - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_qgwo_gate_weights_{name}.png"), dpi=220)
        plt.close()


def make_fold_local_feature_importance(raw_with_id: pd.DataFrame, out_dir: str, seed: int = 42):
    """
    Fold-local LightGBM feature importance for explainability.
    This trains a separate explanatory model on seed-specific outer train data
    only. It does NOT affect Q-PEAK predictions.
    """
    try:
        batteries = np.array(sorted(raw_with_id["Battery_ID"].unique()))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_i, te_i = next(gss.split(batteries.reshape(-1, 1), groups=batteries))
        train_batts = list(map(int, batteries[tr_i]))
        test_batts = list(map(int, batteries[te_i]))

        dtr, dte = prepare_fold_features(raw_with_id, train_batts, test_batts)
        _, full_feats, _, _, _ = get_feature_sets(pd.concat([dtr, dte], ignore_index=True))
        full_feats = _safe_features(dtr, full_feats)
        if len(full_feats) == 0:
            return

        sc = StandardScaler()
        Xtr = sc.fit_transform(dtr[full_feats].values)
        ytr = dtr["RUL"].values.astype(float)

        # Diagnostic-only explanatory model. This does not affect Q-PEAK training or predictions.
        clf = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=20.0,
            min_child_samples=20,
            objective="huber",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(Xtr, ytr)

        fi = pd.DataFrame({"Feature": full_feats, "Importance": clf.feature_importances_})
        fi = fi.sort_values("Importance", ascending=False)
        safe_to_csv(fi, os.path.join(out_dir, f"foldlocal_lightgbm_feature_importance_seed{seed}.csv"))

        top = fi.head(20).iloc[::-1]
        plt.figure(figsize=(9, 7))
        plt.barh(top["Feature"], top["Importance"], edgecolor="black")
        plt.xlabel("LightGBM Feature Importance")
        plt.title(f"Fold-local Feature Importance - Seed {seed}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_foldlocal_feature_importance_seed{seed}.png"), dpi=220)
        plt.close()

    except Exception as e:
        safe_to_csv(pd.DataFrame([{"Error": str(e)}]), os.path.join(out_dir, "foldlocal_feature_importance_error.csv"))


def save_literature_comparison_table(master_df: pd.DataFrame, out_dir: str):
    """
    Literature-level comparison table. Values from prior studies should be
    treated as non-identical benchmark references because split protocols and
    preprocessing are not always the same.
    """
    rows = []

    try:
        fs = master_df[master_df["Experiment"] == "FiveSeed_CausalPIMS"].iloc[0]
        rows.append({
            "Study": "This study",
            "Dataset": "HNEI Battery_RUL.csv",
            "Protocol": "5-seed battery-level Group Split",
            "Model": "Q-PEAK: QGWO + Tau-gated KAN + Causal PIMS",
            "RMSE": fs.get("Test_RMSE_Mean", np.nan),
            "MAE": fs.get("Test_MAE_Mean", np.nan),
            "MSE": fs.get("Test_MSE_Mean", np.nan),
            "R2": fs.get("Test_R2_Mean", np.nan),
            "Notes": "Strict battery-level group split; mean over five seeds."
        })
    except Exception:
        pass

    try:
        lo = master_df[master_df["Experiment"] == "LOBO_CausalPIMS"].iloc[0]
        rows.append({
            "Study": "This study",
            "Dataset": "HNEI Battery_RUL.csv",
            "Protocol": "LOBO",
            "Model": "Q-PEAK: QGWO + Tau-gated KAN + Causal PIMS",
            "RMSE": lo.get("Test_RMSE_Mean", np.nan),
            "MAE": lo.get("Test_MAE_Mean", np.nan),
            "MSE": lo.get("Test_MSE_Mean", np.nan),
            "R2": lo.get("Test_R2_Mean", np.nan),
            "Notes": "Leave-one-battery-out validation."
        })
    except Exception:
        pass

    rows.extend([
        {
            "Study": "Sekhar et al. (2023)",
            "Dataset": "HNEI",
            "Protocol": "Reported ML comparison",
            "Model": "Selected ML algorithms",
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "R2": np.nan,
            "Notes": "Reports MSE, RMSE, MAE, R2 and execution time; fill exact best values after final table verification."
        },
        {
            "Study": "Sravanthi and Sekhar (2025)",
            "Dataset": "HNEI",
            "Protocol": "Reported ML regression comparison",
            "Model": "Bagging Regressor",
            "RMSE": 3.782,
            "MAE": 2.099,
            "MSE": 14.307,
            "R2": 0.999,
            "Notes": "Reported best model in uploaded paper; protocol may not match group split."
        },
        {
            "Study": "Paneru et al.",
            "Dataset": "Battery RUL dataset",
            "Protocol": "Cross-validation / explainable AI framework",
            "Model": "Two-level ensemble learning",
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "R2": np.nan,
            "Notes": "Use exact table values after final verification; includes SHAP / interpretability analysis."
        },
        {
            "Study": "Battery-Insight-PSO",
            "Dataset": "NASA + HNEI",
            "Protocol": "Reported train/test evaluation",
            "Model": "PSO-XGBoost",
            "RMSE": np.nan,
            "MAE": np.nan,
            "MSE": np.nan,
            "R2": np.nan,
            "Notes": "Reports PSO-XGBoost and feature importance; fill exact HNEI RUL values after final verification."
        },
    ])

    out = pd.DataFrame(rows)
    safe_to_csv(out, os.path.join(out_dir, "literature_comparison_table.csv"))
    return out


def save_runtime_summary(start_time: float, out_dir: str):
    elapsed = time.time() - start_time
    rows = [{
        "Model_Variant": MODEL_VARIANT,
        "Elapsed_Minutes": elapsed / 60.0,
        "Torch_OK": TORCH_OK,
        "Device": str(DEVICE),
        "Use_GatedResidualEnsemble": bool(USE_QGWO_ENSEMBLE_BRANCHES),
        "Use_Single_Residual_Learner": bool(USE_SINGLE_RESIDUAL_BRANCH),
        "Single_Residual_Learner_Kind": SINGLE_RESIDUAL_BRANCH_KIND,
        "QGWO_Wolves": GLOBAL_QGWO_WOLVES,
        "QGWO_Iters": GLOBAL_QGWO_ITERS,
        "Save_Predictions": bool(SAVE_PREDICTIONS),
    }]
    safe_to_csv(pd.DataFrame(rows), os.path.join(out_dir, "runtime_summary.csv"))


def save_ch4_posthoc_reports(
    raw_with_id: pd.DataFrame,
    five_res: Dict[str, Any],
    lobo_res: Dict[str, Any],
    master_df: pd.DataFrame,
    out_dir: str,
    start_time: float,
):
    """
    Chapter 4 post-hoc reporting entry point.
    It only creates extra outputs after all predictions have been computed.
    """
    save_prediction_extra_metrics(out_dir)
    save_correlation_reports(raw_with_id, out_dir)
    plot_prediction_trajectories(out_dir)
    plot_qgwo_gate_weight_figures(five_res, lobo_res, out_dir)
    make_fold_local_feature_importance(raw_with_id, out_dir, seed=FIVE_SEEDS[0])
    save_literature_comparison_table(master_df, out_dir)
    save_runtime_summary(start_time, out_dir)


# =============================================================================
# 11. Main
# =============================================================================
def main():
    start = time.time()
    ensure_dir(OUTPUT_DIR)
    seed_everything(42)

    dataset_path = find_dataset_path()
    print(f"[INFO] Model variant: {MODEL_VARIANT}")
    print(f"[INFO] USE_LEGACY_RESIDUAL_ENSEMBLE: {USE_LEGACY_RESIDUAL_BANK}")
    print(f"[INFO] USE_SINGLE_RESIDUAL_BRANCH: {USE_SINGLE_RESIDUAL_BRANCH}, kind={SINGLE_RESIDUAL_BRANCH_KIND}")
    print(f"[INFO] USE_QGWO_ENSEMBLE_BRANCHES: {USE_QGWO_ENSEMBLE_BRANCHES}, branches={ENSEMBLE_BRANCH_KINDS}")
    print(f"[INFO] Dataset path: {dataset_path}")
    print(f"[INFO] Torch: {TORCH_OK}, device={DEVICE}")

    raw = pd.read_csv(dataset_path)
    raw_with_id = attach_battery_id(raw)

    default_params = Params()

    save_protocol_audit(raw, raw_with_id, OUTPUT_DIR)
    save_json(asdict(default_params), os.path.join(OUTPUT_DIR, "default_params.json"))
    save_json({
        "MODEL_VARIANT": MODEL_VARIANT,
        "USE_LEGACY_RESIDUAL_BANK": USE_LEGACY_RESIDUAL_BANK,
        "USE_SINGLE_RESIDUAL_BRANCH": USE_SINGLE_RESIDUAL_BRANCH,
        "USE_QGWO_ENSEMBLE_BRANCHES": USE_QGWO_ENSEMBLE_BRANCHES,
        "ENSEMBLE_BRANCH_KINDS": ENSEMBLE_BRANCH_KINDS,
        "SINGLE_RESIDUAL_BRANCH_KIND": SINGLE_RESIDUAL_BRANCH_KIND,
        "SINGLE_RESIDUAL_BRANCH_DESCRIPTION": SINGLE_RESIDUAL_BRANCH_DESCRIPTION,
        "USE_CAUSAL_TAU_IN_MAIN": USE_CAUSAL_TAU_IN_MAIN,
        "ENABLE_QGWO_5SEED": ENABLE_QGWO_5SEED,
        "ENABLE_QGWO_LOBO": ENABLE_QGWO_LOBO,
        "QGWO_OBJECTIVE_TRACK": QGWO_OBJECTIVE_TRACK,
        "ENABLE_HARD_BATTERY_LOCAL_REFINE": ENABLE_HARD_BATTERY_LOCAL_REFINE,
        "HGB_INCLUDED": False,
        "CLOCK_BASELINE_FEATURES": CLOCK_BASELINE_FEATURES,
        "FIVE_SEEDS": FIVE_SEEDS,
        "SAVE_PREDICTIONS": SAVE_PREDICTIONS,
        "SAVE_QGWO_HISTORY": SAVE_QGWO_HISTORY,
        "TRAIN_ONLY_POST_CALIBRATION_ENABLED": TRAIN_ONLY_POST_CALIBRATION_ENABLED,
        "POST_CALIBRATION_MODE": POST_CALIBRATION_MODE,
        "POST_CALIBRATION_MAX_ABS": POST_CALIBRATION_MAX_ABS,
        "POST_CALIBRATION_SHRINK": POST_CALIBRATION_SHRINK,
        "POSTHOC_REPORTS": True,
        "LITERATURE_COMPARISON_TABLE": True,
        "CORRELATION_HEATMAPS": True,
        "TRAJECTORY_PLOTS": True,
        "FOLD_LOCAL_FEATURE_IMPORTANCE": True,
        "EXTRA_METRICS_NRMSE_MAPE": True,
    }, os.path.join(OUTPUT_DIR, "run_config.json"))

    five_res = run_five_seeds(raw_with_id, default_params, OUTPUT_DIR)
    lobo_res = run_lobo(raw_with_id, default_params, OUTPUT_DIR)

    master = save_master_summary(five_res, lobo_res, OUTPUT_DIR)
    ablation_summary = save_ablation_summary(five_res, OUTPUT_DIR)
    make_essential_figures(five_res, lobo_res, ablation_summary, OUTPUT_DIR)

    # Chapter 4 post-hoc reports: extra figures/tables only, no effect on model results.
    save_ch4_posthoc_reports(
        raw_with_id=raw_with_id,
        five_res=five_res,
        lobo_res=lobo_res,
        master_df=master,
        out_dir=OUTPUT_DIR,
        start_time=start,
    )

    manifest = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            path = os.path.join(root, f)
            manifest.append({
                "file": os.path.relpath(path, OUTPUT_DIR),
                "size_bytes": int(os.path.getsize(path)),
            })

    safe_to_csv(pd.DataFrame(manifest).sort_values("file"), os.path.join(OUTPUT_DIR, "output_manifest.csv"))

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    shutil.make_archive(ZIP_PATH.replace(".zip", ""), "zip", OUTPUT_DIR)

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("[DONE]")
    print("=" * 80)
    print(f"Model variant: {MODEL_VARIANT}")
    print(f"USE_LEGACY_RESIDUAL_ENSEMBLE: {USE_LEGACY_RESIDUAL_BANK}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Zip file: {ZIP_PATH}")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")
    print("\n[MASTER SUMMARY]")
    cols = ["Experiment", "Test_RMSE_Mean", "Test_RMSE_Std", "Test_RMSE_Median", "Test_MAE_Mean", "Test_R2_Mean"]
    print(master[cols].to_string(index=False))


if __name__ == "__main__":
    main()
