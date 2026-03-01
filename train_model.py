import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. 物理感知特徵工程與離群值清洗
# ==========================================
def engineer_features_research(df_clean):
    df = df_clean.copy()
    
    # [關鍵修正] 物理離群值清洗：過濾感測器錯誤導致的「百萬秒」異常放電
    # 根據物理邏輯，正常放電時間不應超過 10,000s
    df["Discharge Time (s)"] = df["Discharge Time (s)"].clip(lower=0, upper=10000)
    
    c = pd.to_numeric(df.get("Cycle_Index", np.zeros(len(df))), errors="coerce").fillna(0).values
    df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])
    df = df.sort_values(["Battery_ID", "Cycle_Index"]).reset_index(drop=True)
    df["Cycle_Index"] = df.groupby("Battery_ID").cumcount()
    
    df["Cap"] = df.get("Discharge Time (s)", 0.0).astype(float)
    df["IR"]  = (4.2 - df.get("Max. Voltage Dischar. (V)", 4.2)).astype(float)
    df["CV_Ratio"] = (df.get("Time at 4.15V (s)", 0.0) / (df.get("Charging time (s)", 1.0) + 1e-6)).astype(float)
    
    for col in ["Cap", "IR", "CV_Ratio"]:
        init = df.groupby("Battery_ID")[col].transform("first")
        df[f"{col}_Ret"] = df[col] / (init + 1e-6)
        df[f"{col}_EMA"] = df.groupby("Battery_ID")[f"{col}_Ret"].transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean())
        for w in [10, 20]:
            df[f"{col}_rm{w}"] = df.groupby("Battery_ID")[f"{col}_EMA"].transform(lambda x: x.rolling(w, min_periods=1).mean())
    
    df["Tau"] = (1.0 / (1.0 + np.exp(-df["Cap_EMA"]))).clip(0.0, 1.0)
    df["Cum_Ah_log1p"] = np.log1p(df.groupby("Battery_ID")["Cap"].cumsum())
    return df.fillna(0.0)

# ==========================================
# 2. 訓練並封裝全套 NNLS 集成架構
# ==========================================
def train_and_export():
    if not os.path.exists("Battery_RUL.csv"):
        print("❌ 找不到數據檔案：Battery_RUL.csv")
        return

    df_raw = pd.read_csv("Battery_RUL.csv")
    df_all = engineer_features_research(df_raw)
    
    S1_FEATS = ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "Cap_rm20", "IR_rm20", "Tau", "Cum_Ah_log1p"]
    clock_feats = ["Cycle_Index", "Cum_Ah_log1p"]
    y_col = "RUL"
    
    # [第一層] 線性時鐘背景模型
    sc_clock = StandardScaler()
    X_c_sc = sc_clock.fit_transform(df_all[clock_feats])
    base_model = Ridge(alpha=12.44).fit(X_c_sc, df_all[y_col])
    yb_full = base_model.predict(X_c_sc)
    
    # [第二層] 物理殘差修正集成
    sc_phys = StandardScaler()
    X_p_sc = sc_phys.fit_transform(df_all[S1_FEATS])
    res_full = df_all[y_col] - yb_full
    
    # 建立四個集成子模型
    xgb_model = xgb.XGBRegressor(n_estimators=2500, learning_rate=0.0055, max_depth=3, subsample=0.7, reg_lambda=37.27, random_state=42)
    et_model = ExtraTreesRegressor(n_estimators=400, max_depth=7, min_samples_leaf=12, random_state=42)
    hgb_model = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.015, l2_regularization=5.1204, random_state=42)
    lin_model = BayesianRidge()
    
    print("⏳ 正在訓練全套集成引擎 (NNLS Optimized)...")
    xgb_model.fit(X_p_sc, res_full)
    et_model.fit(X_p_sc, res_full)
    hgb_model.fit(X_p_sc, res_full)
    lin_model.fit(X_p_sc, res_full)
    
    # 計算 NNLS 權重
    P_full = np.column_stack([
        xgb_model.predict(X_p_sc), et_model.predict(X_p_sc), 
        hgb_model.predict(X_p_sc), lin_model.predict(X_p_sc), np.ones(len(df_all))
    ])
    w_full, _ = nnls(P_full, res_full)
    
    # 封裝模型包
    model_package = {
        "sc_clock": sc_clock,
        "sc_phys": sc_phys,
        "base_model": base_model,
        "xgb_model": xgb_model,
        "et_model": et_model,
        "hgb_model": hgb_model,
        "lin_model": lin_model,
        "nnls_w": w_full,
        "s1_feats": S1_FEATS,
        "clock_feats": clock_feats,
        "max_clip": float(df_all[y_col].max())
    }
    
    joblib.dump(model_package, "probms_model.pkl")
    print("✅ probms_model.pkl 儲存成功！具備強健化 SOH 基準與多層集成架構。")

if __name__ == "__main__":
    train_and_export()