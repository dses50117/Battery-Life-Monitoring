# ===================================================================================
#  本研究核心數據與圖表：
# 1. 原始資料清洗報告 
# 2. 原始特徵熱力圖 
# 3. 轉換映射表 
# 4. 最終特徵清單 
# 5. 物理特徵熱力圖  
# 6. 最佳超參數表 
# 7. 5-Seed 訓練測試指標 (RMSE, MAE, R2)
# 8. 5-Seed 詳細消融實驗矩陣與階梯圖 
# 9. 14-Fold LOBO 盲測與異常排除
# 10. 全局特徵重要性 
# 11. 殘差分佈圖 
# 12. SHAP 歸因圖
# 13. 殘差散點圖 
# 14. 物理單調性退化曲線證明圖
# ===================================================================================
import os, sys, warnings, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def install_package(pkg: str):
    try: __import__(pkg)  # 嘗試讀取套件
    except:               # 如果讀不到（代表沒安裝）
        import subprocess
        # 呼叫系統指令 pip install 安裝它，-q 代表安靜模式（不顯示安裝過程）
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_package("xgboost")
install_package("shap")
install_package("tabulate")
import xgboost as xgb
import shap

from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#「防崩潰」檢查
TORCH_OK = True
try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception: TORCH_OK = False   #如果這台電腦沒裝 PyTorch，程式不會直接報錯當機，而是把 TORCH_OK 設為 False

#如果有 NVIDIA 顯示卡且驅動正常 (cuda) $\rightarrow$ 使用 GPU 運算。否則, 使用 CPU 運算。
DEVICE = torch.device("cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu") if TORCH_OK else None

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)   #固定 Python 內建隨機,  NumPy 矩陣隨機
    if TORCH_OK:
        torch.manual_seed(seed)   # 固定 PyTorch CPU 隨機
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)   #固定 PyTorch GPU 隨機

def calc_metrics(y_true, y_pred):    #定義工具函數，一次產出論文表格需要的所有數據
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

# ===================================================================================
# 1. 原始資料清理與報告
# ===================================================================================
def clean_raw_data_with_report(df: pd.DataFrame):
    print("\n" + "="*80)
    print(" [Req 1] 原始資料檢查與異常值剔除報告")
    print("="*80)
    orig_count = len(df)
    print(f"   [-] 原始資料總筆數: {orig_count} 筆")
    
    df = df.copy().dropna(subset=["RUL"])
    print(f"   [-] 剔除 RUL 缺失值後: {len(df)} 筆")
    
    for c in ["Discharge Time (s)", "Time at 4.15V (s)", "Charging time (s)", "Time constant current (s)"]:
        if c in df.columns: 
            df[c] = pd.to_numeric(df[c], errors="coerce")
            upper_limit = df[c].quantile(0.999)
            df[c] = df[c].clip(lower=0.0, upper=upper_limit)
            
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"   [-] 處理無限值與感測器極端值 (Clip 99.9%) 後最終筆數: {len(df)} 筆")
    print(f"   [+] 總計清洗掉的雜訊/缺失比例: {((orig_count - len(df))/orig_count)*100:.2f}%")
    return df

# ===================================================================================
# 2. 物理感知特徵工程模塊 (回歸最強的動態滑動視窗版本)
# ===================================================================================
def ema_group(series, alpha=0.1): return series.ewm(alpha=alpha, adjust=False).mean()
def robust_mad_scale(a: np.ndarray):
    med = np.median(a); mad = np.median(np.abs(a - med)) + 1e-6
    return 1.4826 * mad

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()
    c = pd.to_numeric(df.get("Cycle_Index", np.zeros(len(df))), errors="coerce").fillna(0).values
    df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])
    df = df.sort_values(["Battery_ID", "Cycle_Index"]).reset_index(drop=True)
    df["Cycle_Index"] = df.groupby("Battery_ID").cumcount()

    df["Cap"] = df.get("Discharge Time (s)", 0.0).astype(float)
    df["IR"]  = (4.2 - df.get("Max. Voltage Dischar. (V)", 4.2)).astype(float)
    df["CV_Ratio"] = (df.get("Time at 4.15V (s)", 0.0) / (df.get("Charging time (s)", 1.0) + 1e-6)).astype(float)
    df["CC_Time"] = df.get("Time constant current (s)", 0.0).astype(float)

    df["Energy"] = (df["Cap"] * 3.6).clip(lower=0)
    df["Cum_Ah"] = df.groupby("Battery_ID")["Cap"].cumsum()
    df["Cum_Energy"] = df.groupby("Battery_ID")["Energy"].cumsum()
    
    for col in ["Cap", "IR", "CV_Ratio", "CC_Time"]:
        init = df.groupby("Battery_ID")[col].transform("first")
        df[f"{col}_Ret"] = df[col] / (init + 1e-6)
        df[f"{col}_EMA"] = df.groupby("Battery_ID")[f"{col}_Ret"].transform(lambda x: ema_group(x, 0.1))
        df[f"{col}_v1"] = df.groupby("Battery_ID")[f"{col}_EMA"].diff(10).fillna(0.0)

    H = 1.0 * df["Cap_EMA"] + 0.35 * (1.0 / (df["IR_EMA"] + 1e-6)) - 0.25 * df["CV_Ratio_EMA"]
    df["_H"] = H
    H_init = df.groupby("Battery_ID")["_H"].transform(lambda x: x.iloc[:10].mean() if len(x) >= 10 else x.mean())
    H_scale = np.clip(df.groupby("Battery_ID")["_H"].transform(lambda x: robust_mad_scale(x.iloc[:10].mean() - x.values)).values, 1e-3, None)
    df["Tau"] = (1.0 / (1.0 + np.exp(-(H_init - df["_H"]).values / H_scale))).clip(0.0, 1.0)
    df["Cycle_log"] = np.log1p(df["Cycle_Index"].astype(float))
    df["Cum_Ah_log1p"] = np.log1p(df["Cum_Ah"])
    df["Cum_Energy_log1p"] = np.log1p(df["Cum_Energy"])

    for col in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "Tau"]:
        g = df.groupby("Battery_ID")[col]
        for w in [10, 20]:
            df[f"{col}_rm{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_rs{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0.0)
            df[f"{col}_slope{w}"] = g.transform(lambda x: ((x - x.shift(w)).fillna(0.0)) / (w + 1e-6))
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ===================================================================================
# 3. 物理感知 KAN 神經網路與殘差管線
# ===================================================================================
if TORCH_OK:
    class KANLinear(nn.Module):
        def __init__(self, in_features, out_features, grid_size=3, spline_order=2):
            super().__init__()
            self.in_features, self.out_features, self.spline_order = in_features, out_features, spline_order
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
                d1 = (self.grid[:, k:-1] - self.grid[:, :-(k+1)] + 1e-6)
                d2 = (self.grid[:, k+1:] - self.grid[:, 1:-k] + 1e-6)
                bases = (x - self.grid[:, :-(k+1)]) / d1 * bases[:, :, :-1] + (self.grid[:, k+1:] - x) / d2 * bases[:, :, 1:]
            return bases.contiguous()

        def forward(self, x):
            base_out = torch.nn.functional.linear(torch.nn.functional.silu(x), self.base_weight)
            scaled_w = (self.spline_weight * self.spline_scaler.unsqueeze(-1)).view(self.out_features, -1)
            return base_out + torch.nn.functional.linear(self.b_splines(x).view(x.size(0), -1), scaled_w)

    class PhysicsKAN(nn.Module):
        def __init__(self, in_dim, hidden_dim=4): 
            super().__init__()
            self.kan1 = KANLinear(in_dim, hidden_dim)
            self.kan2 = KANLinear(hidden_dim, 1)
        def forward(self, x, amp): return torch.tanh(self.kan2(self.kan1(x)).squeeze(-1)) * amp

def train_physics_kan(X_tr, r_tr, X_te, amp_tr, amp_te, epochs=25, lr=0.015, kan_seed=42):
    if not TORCH_OK: return np.zeros(len(X_tr)), np.zeros(len(X_te))
    torch.manual_seed(kan_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(kan_seed)
    model = PhysicsKAN(X_tr.shape[1], hidden_dim=4).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    crit = nn.HuberLoss(delta=1.0)
    Xt, yt, at = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE), torch.tensor(r_tr, dtype=torch.float32, device=DEVICE), torch.tensor(amp_tr, dtype=torch.float32, device=DEVICE)
    Xte, ate = torch.tensor(X_te, dtype=torch.float32, device=DEVICE), torch.tensor(amp_te, dtype=torch.float32, device=DEVICE)
    dl = DataLoader(TensorDataset(Xt, yt, at), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb, ab in dl:
            opt.zero_grad(); crit(model(xb, ab), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad(): return model(Xt, at).detach().cpu().numpy(), model(Xte, ate).detach().cpu().numpy()

def apply_pims(df_subset, preds):
    preds = np.asarray(preds, float).reshape(-1)
    out = np.zeros_like(preds)
    for b in np.unique(df_subset["Battery_ID"].values):
        idx = (df_subset["Battery_ID"].values == b)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        out[idx] = iso.fit_transform(df_subset.loc[idx, "Cycle_Index"].values.astype(float), preds[idx])
    return out

def run_pipeline_full(df_tr, df_te, feats, current_seed, params, use_nnls=True, use_kan=True):
    y_tr, y_te = df_tr["RUL"].values.astype(float), df_te["RUL"].values.astype(float)
    groups = df_tr["Battery_ID"].values

    clock_feats = ["Cycle_Index", "Cum_Ah", "Cum_Energy"] if "Cum_Energy" in df_tr.columns else ["Cycle_Index", "Cum_Ah"]
    sc_clock = StandardScaler()
    Xtr_c = sc_clock.fit_transform(df_tr[clock_feats].values)
    Xte_c = sc_clock.transform(df_te[clock_feats].values)

    base = Ridge(alpha=params['base_alpha'])
    base.fit(Xtr_c, y_tr)
    max_clip = y_tr.max() * 1.1 
    yb_tr = np.clip(base.predict(Xtr_c), 0.0, max_clip)
    yb_te = np.clip(base.predict(Xte_c), 0.0, max_clip)
    r_base_tr = y_tr - yb_tr

    sc = StandardScaler()
    Xtr = sc.fit_transform(df_tr[feats].values)
    Xte = sc.transform(df_te[feats].values)

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    oof_xgb, pred_xgb = np.zeros(len(df_tr)), np.zeros(len(df_te))
    oof_lin, pred_lin = np.zeros(len(df_tr)), np.zeros(len(df_te))
    oof_et,  pred_et  = np.zeros(len(df_tr)), np.zeros(len(df_te))
    oof_hgb, pred_hgb = np.zeros(len(df_tr)), np.zeros(len(df_te))

    N_BAGS = 3 if use_kan else 1

    for fold, (ti, vi) in enumerate(gkf.split(Xtr, r_base_tr, groups=groups)):
        oof_xgb_fold = np.zeros(len(vi)); pred_xgb_fold = np.zeros(len(df_te))
        for b in range(N_BAGS):
            clf_xgb = xgb.XGBRegressor(
                n_estimators=2500, learning_rate=params['xgb_lr'], max_depth=params['xgb_depth'], 
                subsample=0.7, colsample_bytree=0.7, reg_lambda=params['xgb_lambda'], 
                objective="reg:pseudohubererror", tree_method="hist", n_jobs=-1, 
                random_state=current_seed + fold*10 + b
            )
            clf_xgb.fit(Xtr[ti], r_base_tr[ti], verbose=False)
            oof_xgb_fold += clf_xgb.predict(Xtr[vi]) / N_BAGS
            pred_xgb_fold += clf_xgb.predict(Xte) / (n_splits * N_BAGS)
        oof_xgb[vi] = oof_xgb_fold; pred_xgb += pred_xgb_fold

        clf_et = ExtraTreesRegressor(n_estimators=400, max_depth=params['et_depth'], min_samples_leaf=params['et_leaf'], n_jobs=-1, random_state=current_seed + fold)
        clf_et.fit(Xtr[ti], r_base_tr[ti])
        oof_et[vi] = clf_et.predict(Xtr[vi])
        pred_et += clf_et.predict(Xte) / n_splits

        clf_hgb = HistGradientBoostingRegressor(loss="squared_error", max_depth=params['hgb_depth'], learning_rate=0.015, min_samples_leaf=12, l2_regularization=params['hgb_l2'], random_state=current_seed + fold)
        clf_hgb.fit(Xtr[ti], r_base_tr[ti])
        oof_hgb[vi] = clf_hgb.predict(Xtr[vi])
        pred_hgb += clf_hgb.predict(Xte) / n_splits

        clf_lin = BayesianRidge()
        clf_lin.fit(Xtr[ti], r_base_tr[ti])
        oof_lin[vi] = clf_lin.predict(Xtr[vi])
        pred_lin += clf_lin.predict(Xte) / n_splits

    P_tr_ext = np.column_stack([oof_xgb, oof_et, oof_hgb, oof_lin])
    P_te_ext = np.column_stack([pred_xgb, pred_et, pred_hgb, pred_lin])
    
    if use_nnls:
        P_tr_nnls = np.column_stack([P_tr_ext, np.ones(len(df_tr))])
        P_te_nnls = np.column_stack([P_te_ext, np.ones(len(df_te))])
        w, _ = nnls(P_tr_nnls, r_base_tr)
        y_hat_trees_tr = yb_tr + P_tr_nnls @ w
        y_hat_trees_te = yb_te + P_te_nnls @ w
    else:
        y_hat_trees_tr = yb_tr + np.mean(P_tr_ext, axis=1)
        y_hat_trees_te = yb_te + np.mean(P_te_ext, axis=1)

    if not use_kan:
        tr_pred = np.clip(apply_pims(df_tr, y_hat_trees_tr), 0.0, max_clip)
        te_pred = np.clip(apply_pims(df_te, y_hat_trees_te), 0.0, max_clip)
        return y_tr, tr_pred, y_te, te_pred

    r_calib_tr = y_tr - y_hat_trees_tr
    kan_feats = [c for c in ["Tau_rm20", "CV_Ratio_EMA_rm20", "Cap_v1"] if c in df_tr.columns]
    sc_kan = StandardScaler()
    Xtr_k, Xte_k = sc_kan.fit_transform(df_tr[kan_feats].values), sc_kan.transform(df_te[kan_feats].values)

    tau_tr = np.clip(df_tr["Tau_rm20"].values if "Tau_rm20" in df_tr.columns else df_tr["Tau"].values, 0.0, 1.0)
    tau_te = np.clip(df_te["Tau_rm20"].values if "Tau_rm20" in df_te.columns else df_te["Tau"].values, 0.0, 1.0)
    amp_tr = 3.0 + 17.0 * tau_tr; amp_te = 3.0 + 17.0 * tau_te

    kan_tr_total, kan_te_total = np.zeros(len(df_tr)), np.zeros(len(df_te))
    for b in range(N_BAGS):
        kan_tr, kan_te = train_physics_kan(Xtr_k, r_calib_tr, Xte_k, amp_tr, amp_te, epochs=25, lr=0.015, kan_seed=current_seed + b*99)
        kan_tr_total += kan_tr / N_BAGS
        kan_te_total += kan_te / N_BAGS

    tr_pred = np.clip(apply_pims(df_tr, y_hat_trees_tr + kan_tr_total), 0.0, max_clip)
    te_pred = np.clip(apply_pims(df_te, y_hat_trees_te + kan_te_total), 0.0, max_clip)
    return y_tr, tr_pred, y_te, te_pred

# ===================================================================================
# 4. 主程式執行與output 
# ===================================================================================
if __name__ == "__main__":
    path = "/kaggle/input/battery-remaining-useful-life-rul/Battery_RUL.csv"
    if not os.path.exists(path):
        fallback_path = "Battery_RUL.csv"
        if os.path.exists(fallback_path): path = fallback_path
        else: sys.exit(f"⚠️ 找不到資料集: {path}")

    # --- [Req 1] 資料清理 ---
    df_raw = pd.read_csv(path)
    df_clean = clean_raw_data_with_report(df_raw)
    
    # --- [Req 2] 原始資料熱力圖 ---
    print("\n [Req 2] Generating Raw Feature Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    raw_corr_cols = [c for c in df_clean.columns if df_clean[c].dtype in [float, int] and c != "Battery_ID"]
    sns.heatmap(df_clean[raw_corr_cols].corr(), annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Figure A: HNEI Raw Dataset Feature Correlation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- [Req 3 & 4] 特徵工程映射與清單 ---
    print("\n" + "="*80)
    print(" [Req 3] 特徵工程轉換映射說明 (Feature Engineering Mapping)")
    print("="*80)
    print("1. [Discharge Time] -> 容量衰退率 (Cap_EMA) 與絕對時鐘 (Cum_Ah_log1p)")
    print("2. [Max. Voltage Dischar.] -> 內阻代理指標 (IR_EMA = 4.2 - V)")
    print("3. [Time at 4.15V / Charging time] -> 極化效應指標 (CV_Ratio_EMA)")
    print("4. [Cycle_Index] -> 平滑化對數時鐘 (Cycle_log)")
    print("5. 捨棄具備高度共線性的多餘電壓/時間特徵，降維除噪。")
    print("6. [動態特徵] 針對 Cap, IR, CV_Ratio 提取 10/20 週期之均值(_rm), 標準差(_rs), 斜率(_slope)。")
    
    df_all = engineer_features(df_clean)
    base_feats = [c for c in ["Cap_EMA","IR_EMA","CV_Ratio_EMA","Tau","CC_Time_EMA"] if c in df_all.columns]
    base_feats += [c for c in ["Cap_v1","IR_v1","CV_Ratio_v1"] if c in df_all.columns]
    for col in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "Tau"]:
        for w in [10, 20]:
            for suf in ["_rm", "_rs", "_slope"]:
                if f"{col}{suf}{w}" in df_all.columns: base_feats.append(f"{col}{suf}{w}")
    clock_plus = [c for c in ["Cum_Ah_log1p","Cum_Energy_log1p","Cycle_Index","Cycle_log"] if c in df_all.columns]
    S1_FEATS = sorted(list(set(base_feats + clock_plus)))
    batts = np.array(sorted(df_all["Battery_ID"].unique()))

    print(f"\n [Req 4] 最終入模之物理感知特徵清單 (共 {len(S1_FEATS)} 項):")
    print(", ".join(S1_FEATS))

    # --- [Req 5] 工程後特徵熱力圖 ---
    print("\n [Req 5] Generating Engineered Features Correlation Heatmap...")
    top_corr_feats = df_all[S1_FEATS + ["RUL"]].corr()["RUL"].abs().sort_values(ascending=False).index[:16]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_all[top_corr_feats].corr(), annot=True, cmap='RdYlGn', fmt=".2f", annot_kws={"size": 8})
    plt.title('Figure B: Engineered Physics-Informed Features Correlation (Top 15)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- [Req 6] 最佳超參數表 ---
    best_params = {
        'base_alpha': 12.4408, 'xgb_depth': 3, 'xgb_lr': 0.0055, 'xgb_lambda': 37.2796,
        'et_depth': 7, 'et_leaf': 12, 'hgb_depth': 2, 'hgb_l2': 5.1204
    }
    baseline_params = {
        'base_alpha': 15.0, 'xgb_depth': 3, 'xgb_lr': 0.01, 'xgb_lambda': 10.0,
        'et_depth': 7, 'et_leaf': 10, 'hgb_depth': 3, 'hgb_l2': 5.0
    }
    print("\n" + "="*80)
    print(" [Req 6] NSGA-II 帕雷托最佳超參數配置 (Best Hyperparameters)")
    print("="*80)
    for k, v in best_params.items(): print(f"   - {k}: {v}")

    all_residuals = []
    all_preds = []
    
    # --- [Req 7 & 8] 5-Seed 訓練測試評估與消融實驗 ---
    print("\n" + "="*80)
    print(" [Req 7 & 8] 執行 5-Seed 盲測與消融實驗... 預計 3-5 分鐘")
    print("="*80)
    best_seeds = [42, 43, 44, 45, 46]
    metrics_train_full = {"RMSE":[], "MAE":[], "MSE":[], "R2":[]}
    metrics_test_full = {"RMSE":[], "MAE":[], "MSE":[], "R2":[]}
    ablation_results = {"1. Base": [], "2. + NSGA-II": [], "3. + NNLS": [], "4. + KAN (Full)": []}

    for s in best_seeds:
        seed_everything(s)
        gss_s = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
        tr_i, te_i = next(gss_s.split(df_all, groups=df_all["Battery_ID"]))
        dtr, dte = df_all.iloc[tr_i].copy().reset_index(drop=True), df_all.iloc[te_i].copy().reset_index(drop=True)
        
        _, _, y_te_abl, p_te_base = run_pipeline_full(dtr, dte, S1_FEATS, s, baseline_params, use_nnls=False, use_kan=False)
        ablation_results["1. Base"].append(calc_metrics(y_te_abl, p_te_base)["RMSE"])
        
        _, _, _, p_te_nsga = run_pipeline_full(dtr, dte, S1_FEATS, s, best_params, use_nnls=False, use_kan=False)
        ablation_results["2. + NSGA-II"].append(calc_metrics(y_te_abl, p_te_nsga)["RMSE"])
        
        _, _, _, p_te_nnls = run_pipeline_full(dtr, dte, S1_FEATS, s, best_params, use_nnls=True, use_kan=False)
        ablation_results["3. + NNLS"].append(calc_metrics(y_te_abl, p_te_nnls)["RMSE"])
        
        y_tr, p_tr, y_te, p_te = run_pipeline_full(dtr, dte, S1_FEATS, s, best_params, use_nnls=True, use_kan=True)
        
        tr_m = calc_metrics(y_tr, p_tr)
        te_m = calc_metrics(y_te, p_te)
        
        for m in ["RMSE", "MAE", "MSE", "R2"]:
            metrics_train_full[m].append(tr_m[m])
            metrics_test_full[m].append(te_m[m])
            
        ablation_results["4. + KAN (Full)"].append(te_m["RMSE"])
        all_residuals.extend(y_te - p_te)
        all_preds.extend(p_te)
        print(f"   [Seed {s}] Test RMSE: {te_m['RMSE']:.4f}")

    print("\n--- [Req 7] 5-Seed 平均模型指標 (Train vs Test) ---")
    df_metrics = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MSE", "R2"],
        "Train (Mean ± Std)": [f"{np.mean(metrics_train_full[m]):.4f} ± {np.std(metrics_train_full[m]):.4f}" for m in ["RMSE", "MAE", "MSE", "R2"]],
        "Test (Mean ± Std)": [f"{np.mean(metrics_test_full[m]):.4f} ± {np.std(metrics_test_full[m]):.4f}" for m in ["RMSE", "MAE", "MSE", "R2"]]
    })
    print(df_metrics.to_markdown(index=False))

    print("\n--- [Req 8] 5-Seed 詳細消融實驗矩陣 (Ablation Matrix) ---")
    abl_detailed = pd.DataFrame(ablation_results)
    abl_detailed.index = [f"Seed {s}" for s in best_seeds]
    abl_detailed.loc["Average RMSE"] = abl_detailed.mean()
    print(abl_detailed.to_markdown())

    # --- 繪製 Figure 2: 階梯圖 ---
    print("\n Generating [Figure 2]: Ablation Study Bar Chart...")
    plt.figure(figsize=(10, 6))
    configs = list(ablation_results.keys())
    avg_rmses = abl_detailed.loc["Average RMSE"].values
    sns.barplot(x=configs, y=avg_rmses, palette="Reds_r", edgecolor='black')
    for i, v in enumerate(avg_rmses):
        plt.text(i, v + 0.05, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)
    plt.title('Figure 2: Ablation Study - Contribution of Proposed Modules', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Average Test RMSE', fontweight='bold', fontsize=12)
    plt.ylim(min(avg_rmses) - 0.2, max(avg_rmses) + 0.3)
    plt.tight_layout()
    plt.show()

    # --- [Req 9] 14-Fold LOBO 極限盲測 ---
    print("\n" + "="*80)
    print(" [Req 9] 執行 14-Fold LOBO 極限盲測 (揭示 Battery 4 異常分佈)...")
    print("="*80)
    lobo_metrics = []
    
    seed_everything(45)
    for test_b in batts:
        dtr, dte = df_all[df_all["Battery_ID"] != test_b].copy().reset_index(drop=True), df_all[df_all["Battery_ID"] == test_b].copy().reset_index(drop=True)
        _, _, y_te_lobo, p_te_lobo = run_pipeline_full(dtr, dte, S1_FEATS, 45, best_params, use_nnls=True, use_kan=True)
        m = calc_metrics(y_te_lobo, p_te_lobo)
        m['Battery_ID'] = test_b
        lobo_metrics.append(m)
        print(f"   - LOBO Battery {test_b} | RMSE: {m['RMSE']:.4f}")

    df_lobo = pd.DataFrame(lobo_metrics)
    worst_batt = df_lobo.loc[df_lobo['RMSE'].idxmax()]['Battery_ID']
    df_lobo_excl = df_lobo[df_lobo['Battery_ID'] != worst_batt]

    print("\n--- LOBO 總體平均指標 ---")
    print(f"包含所有電池 (14 顆) -> RMSE: {df_lobo['RMSE'].mean():.4f} | MAE: {df_lobo['MAE'].mean():.4f} | R2: {df_lobo['R2'].mean():.4f}")
    print(f"排除極端異常電池 (扣除 Batt {int(worst_batt)}) -> RMSE: {df_lobo_excl['RMSE'].mean():.4f} | MAE: {df_lobo_excl['MAE'].mean():.4f} | R2: {df_lobo_excl['R2'].mean():.4f}")

    # --- [Req 10 & 12] 特徵重要度排名與 SHAP ---
    print("\n [Req 10 & 12] Generating Feature Importance & SHAP Plots...")
    y_full = df_all["RUL"].values.astype(float)
    X_full = df_all[S1_FEATS].values
    sc_f = StandardScaler()
    X_df_f = pd.DataFrame(sc_f.fit_transform(X_full), columns=S1_FEATS)
    
    clf_xgb_shap = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=best_params['xgb_lr'], max_depth=best_params['xgb_depth'], 
        subsample=0.8, colsample_bytree=0.8, reg_lambda=best_params['xgb_lambda'], 
        objective="reg:pseudohubererror", tree_method="hist", n_jobs=-1, random_state=42
    )
    clf_xgb_shap.fit(X_df_f, y_full, verbose=False)

    plt.figure(figsize=(10, 8))
    pd.Series(clf_xgb_shap.feature_importances_, index=S1_FEATS).sort_values(ascending=True)[-15:].plot(kind='barh', color='#2ca02c', edgecolor='black')
    plt.title('Figure 3: Global Feature Importance (Top 15)', fontsize=15, fontweight='bold'); plt.tight_layout(); plt.show()

    explainer = shap.TreeExplainer(clf_xgb_shap)
    X_samp = X_df_f.sample(2000, random_state=42) if len(X_df_f)>2000 else X_df_f
    shap_vals = explainer.shap_values(X_samp)
    plt.figure(figsize=(10, 8))
    plt.title('Figure 4: SHAP Summary Plot (Physical Feature Impact)', fontsize=15, fontweight='bold', pad=20)
    shap.summary_plot(shap_vals, X_samp, show=False, max_display=15); plt.tight_layout(); plt.show()

    # --- [Req 11] 殘差分佈圖 ---
    print("\n [Req 11] Generating Residual Distribution Plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(all_residuals, kde=True, color='purple', bins=50)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.title('Figure E: Residual Distribution (5-Seed Test Sets)', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error (True - Predicted)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.tight_layout(); plt.show()

    # --- [Req 13] 殘差 vs 預測值散點圖 ---
    print("\n [Req 13] Generating Residuals vs. Predicted Plot...")
    plt.figure(figsize=(8, 5))
    plt.scatter(all_preds, all_residuals, alpha=0.5, color='teal', edgecolor='k')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Figure F: Residuals vs. Predicted Values', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted RUL', fontweight='bold')
    plt.ylabel('Residuals', fontweight='bold')
    plt.tight_layout(); plt.show()

    # --- [Req 14] 物理單調性退化曲線證明 ---
    print("\n [Bonus] Generating Monotonicity Physical Proof Plot...")
    seed_everything(45)
    gss_b = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=45)
    tr_i, te_i = next(gss_b.split(df_all, groups=df_all["Battery_ID"]))
    dtr, dte = df_all.iloc[tr_i].copy().reset_index(drop=True), df_all.iloc[te_i].copy().reset_index(drop=True)
    _, _, y_te_b, p_te_b = run_pipeline_full(dtr, dte, S1_FEATS, 45, best_params, use_nnls=True, use_kan=True)
    
    dte['True_RUL'] = y_te_b
    dte['Pred_RUL'] = p_te_b
    target_batt = dte['Battery_ID'].value_counts().idxmax()
    plot_df = dte[dte['Battery_ID'] == target_batt].sort_values("Cycle_Index")

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["Cycle_Index"], plot_df["True_RUL"], label="True RUL", color='black', linewidth=2)
    plt.plot(plot_df["Cycle_Index"], plot_df["Pred_RUL"], label="Predicted RUL (Monotonic)", color='red', linestyle='-.', linewidth=2.5)
    plt.title(f'Figure G: Physical Monotonicity Proof (Battery {target_batt})', fontsize=15, fontweight='bold')
    plt.xlabel('Cycle Index', fontweight='bold')
    plt.ylabel('Remaining Useful Life (RUL)', fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(); plt.show()

    print("\n所有數據與圖表已全數產出完畢！")