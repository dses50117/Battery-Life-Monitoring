import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import train_model

df_raw = pd.read_csv("Battery_RUL.csv")
df_all = train_model.engineer_features_research(df_raw)

S1_FEATS = ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "Cap_rm20", "IR_rm20", "Tau", "Cum_Ah_log1p"]
clock_feats = ["Cycle_Index", "Cum_Ah_log1p"]

for test_b in range(14):
    df_tr = df_all[df_all["Battery_ID"] != test_b]
    df_te = df_all[df_all["Battery_ID"] == test_b]
    
    sc_clock = StandardScaler()
    Xtr_c_sc = sc_clock.fit_transform(df_tr[clock_feats])
    Xte_c_sc = sc_clock.transform(df_te[clock_feats])
    
    base_model = Ridge(alpha=12.44).fit(Xtr_c_sc, df_tr["RUL"])
    yb_tr = base_model.predict(Xtr_c_sc)
    yb_te = base_model.predict(Xte_c_sc)
    
    sc_phys = StandardScaler()
    Xtr_p_sc = sc_phys.fit_transform(df_tr[S1_FEATS])
    Xte_p_sc = sc_phys.transform(df_te[S1_FEATS])
    
    res = df_tr["RUL"] - yb_tr
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.0055, max_depth=3, random_state=42).fit(Xtr_p_sc, res)
    
    pred_te = yb_te + xgb_model.predict(Xte_p_sc)
    rmse = np.sqrt(mean_squared_error(df_te["RUL"], pred_te))
    print(f"LOBO Battery {test_b} RMSE: {rmse:.4f}")
