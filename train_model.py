import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import os

print("⏳ 開始訓練 PRO-BMS 機器學習模型 (Ridge + XGBoost)...")

# ==========================================
# 1. 取得並處理數據
# ==========================================
file_path = "Battery_RUL.csv"
if not os.path.exists(file_path):
    print(f"❌ 找不到訓練用資料集：{file_path}，請確認檔案位置。")
    exit(1)

print("   [-] 載入原始資料集並執行特徵工程...")
df = pd.read_csv(file_path)
c = pd.to_numeric(df["Cycle_Index"], errors="coerce").fillna(0).values
df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])

# 基礎特徵
df["Cap"] = df.get("Discharge Time (s)", 0.0).astype(float)
df["IR"]  = (4.2 - df.get("Max. Voltage Dischar. (V)", 4.2)).astype(float)
df["CV_Ratio"] = (df.get("Time at 4.15V (s)", 0.0) / (df.get("Charging time (s)", 1.0) + 1e-6)).astype(float)
df["CC_Time"] = df.get("Time constant current (s)", 0.0).astype(float)

# 時鐘特徵
df["Cycle_Index"] = df.groupby("Battery_ID").cumcount()
df["Cum_Ah"] = df.groupby("Battery_ID")["Cap"].cumsum()
df["Cum_Ah_log1p"] = np.log1p(df["Cum_Ah"])

# 物理特徵工程 (EMA, Diff)
for col in ["Cap", "IR", "CV_Ratio", "CC_Time"]:
    init = df.groupby("Battery_ID")[col].transform("first")
    df[f"{col}_Ret"] = df[col] / (init + 1e-6)
    df[f"{col}_EMA"] = df.groupby("Battery_ID")[f"{col}_Ret"].transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean())
    df[f"{col}_v1"] = df.groupby("Battery_ID")[f"{col}_EMA"].diff(10).fillna(0.0)

for col in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA"]:
    g = df.groupby("Battery_ID")[col]
    for w in [10, 20]:
        df[f"{col}_rm{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"{col}_rs{w}"] = g.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0.0)
        df[f"{col}_slope{w}"] = g.transform(lambda x: ((x - x.shift(w)).fillna(0.0)) / (w + 1e-6))

df["Max_SOH"] = df.groupby("Battery_ID")["Cap_EMA"].transform("max")
df["SOH_Percentage"] = (df["Cap_EMA"] / df["Max_SOH"]) * 100
df["Temp_Proxy"] = df["Max. Voltage Dischar. (V)"] * 0.5 + 20 
df["IR_Proxy"] = df["IR"]

df_all = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ==========================================
# 2. 定義入模特徵
# ==========================================
base_feats = [c for c in ["Cap_EMA","IR_EMA","CV_Ratio_EMA","CC_Time_EMA"] if c in df_all.columns]
base_feats += [c for c in ["Cap_v1","IR_v1","CV_Ratio_v1"] if c in df_all.columns]
for col in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA"]:
    for w in [10, 20]:
        for suf in ["_rm", "_rs", "_slope"]:
            if f"{col}{suf}{w}" in df_all.columns: base_feats.append(f"{col}{suf}{w}")
clock_plus = ["Cum_Ah_log1p", "Cycle_Index"]
feats = sorted(list(set(base_feats + clock_plus)))

print(f"   [-] 提取完成，共 {len(feats)} 項混合物理工程特徵。")

# ==========================================
# 3. 執行基線模型訓練 (Ridge + XGBoost)
# ==========================================
y = df_all["RUL"].values.astype(float)
X = df_all[feats].values

print("   [-] 訓練第一階段老化基線模型 (Ridge Regression)...")
sc_clock = StandardScaler()
X_c = sc_clock.fit_transform(df_all[clock_plus].values)
base = Ridge(alpha=12.44)
base.fit(X_c, y)
yb = np.clip(base.predict(X_c), 0.0, y.max() * 1.1)
r_base = y - yb

print("   [-] 訓練第二階段殘差校正模型 (XGBoost Regressor)...")
sc = StandardScaler()
X_sc = sc.fit_transform(X)
clf_xgb = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.01, max_depth=3,
    subsample=0.7, colsample_bytree=0.7, reg_lambda=37.2,
    objective="reg:pseudohubererror", tree_method="hist", n_jobs=-1, random_state=42
)
clf_xgb.fit(X_sc, r_base, verbose=False)

print("   [-] 建立 XGBoost SHAP 歸因解釋器...")
explainer = shap.TreeExplainer(clf_xgb)

# ==========================================
# 4. 打包與儲存
# ==========================================
model_package = {
    "features": feats, 
    "clock_features": clock_plus,
    "sc_clock": sc_clock, 
    "base": base,
    "sc": sc, 
    "clf_xgb": clf_xgb, 
    "explainer": explainer, 
    "max_clip": y.max() * 1.1
}

save_path = "probms_model.pkl"
joblib.dump(model_package, save_path)
print(f"✅ 模型訓練與特徵打包完成！模型已儲存至：{save_path}")
