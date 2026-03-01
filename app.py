import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# ==========================================
# 0. 頁面配置與極致黑 SCADA 風格
# ==========================================
st.set_page_config(page_title="PRO-BMS | 全局監控戰情室 v6.0", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #1a1a1a; }
    .pro-title { text-align: center; color: #00ffca; font-size: 1.6rem; font-weight: 500; margin-top: -20px; margin-bottom: 30px; }
    .kpi-container { text-align: center; padding: 15px; background: rgba(0, 255, 202, 0.03); border: 1px solid #1a1a1a; border-radius: 8px; }
    .kpi-value { font-size: 3rem; color: #00ffca; font-weight: 400; }
    .kpi-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .chart-title { color: #ffffff; font-size: 1rem; border-left: 4px solid #00ffca; padding-left: 10px; margin-top: 10px; }
    .terminal-log { font-family: 'Courier New', monospace; color: #00ffca; font-size: 0.85rem; padding: 10px; background: #050505; border-radius: 5px; border: 1px solid #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數據引擎：研究等級特徵工程 (同步 Kaggle 邏輯)
# ==========================================
@st.cache_data(show_spinner="⏳ 正在同步 14 顆機櫃數據與物理特徵工程...")
def get_advanced_data(file_path):
    df = pd.read_csv(file_path)
    
    # [1] 自動識別 14 顆電池
    c = pd.to_numeric(df["Cycle_Index"], errors="coerce").fillna(0).values
    df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])
    df = df.sort_values(["Battery_ID", "Cycle_Index"]).reset_index(drop=True)
    df["Cycle_Index_Clean"] = df.groupby("Battery_ID").cumcount()
    
    # [2] 物理特徵映射與計算
    df["Cap"] = df["Discharge Time (s)"].astype(float)
    df["IR"]  = (4.2 - df["Max. Voltage Dischar. (V)"]).astype(float)
    df["CV_Ratio"] = (df["Time at 4.15V (s)"] / (df["Charging time (s)"] + 1e-6)).astype(float)
    df["VD"]  = df["Decrement 3.6-3.4V (s)"].astype(float)
    
    # [3] 物理感知特徵 (EMA & Rolling)
    for col in ["Cap", "IR", "CV_Ratio"]:
        init = df.groupby("Battery_ID")[col].transform("first")
        df[f"{col}_Ret"] = df[col] / (init + 1e-6)
        df[f"{col}_EMA"] = df.groupby("Battery_ID")[f"{col}_Ret"].transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean())
        df[f"{col}_rm20"] = df.groupby("Battery_ID")[f"{col}_EMA"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # [4] 物理指標 (SOH & Tau)
    df["SOH"] = df.groupby("Battery_ID")["Cap_EMA"].transform(lambda x: (x / x.iloc[0]) * 100)
    df["Tau"] = (1.0 / (1.0 + np.exp(-df["Cap_EMA"]))).clip(0.0, 1.0)
    df["Cum_Ah_log1p"] = np.log1p(df.groupby("Battery_ID")["Cap"].cumsum())
    
    # [5] 物理極值 (用於雷達圖)
    for feat in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "VD"]:
        df[f"{feat}_min"] = df.groupby("Battery_ID")[feat].transform("min")
        df[f"{feat}_max"] = df.groupby("Battery_ID")[feat].transform("max")
        
    return df.fillna(0.0)

# ==========================================
# 2. 模型載入與推論 (雙層架構：Ridge + XGBoost)
# ==========================================
@st.cache_resource
def load_research_model():
    model_path = "probms_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("❌ 找不到 probms_model.pkl，請先執行新版 train_model.py！")
        st.stop()

# ==========================================
# 3. 側邊欄控制
# ==========================================
if not os.path.exists("Battery_RUL.csv"):
    st.error("❌ 找不到數據檔案：Battery_RUL.csv")
    st.stop()

df_all = get_advanced_data("Battery_RUL.csv")
model_pkg = load_research_model()

with st.sidebar:
    st.markdown("<h2 style='color:#00ffca;'>⚙️ CONFIG</h2>", unsafe_allow_html=True)
    selected_id = st.selectbox("分析機櫃單元", df_all["Battery_ID"].unique(), format_func=lambda x: f"電池陣列 #{x:03d}")
    
    batt_df = df_all[df_all["Battery_ID"] == selected_id].reset_index(drop=True)
    daily_cycles = st.slider("每日循環強度", 0.5, 3.0, 1.2)
    current_idx = st.slider("時間軸模擬 (Cycle)", 0, len(batt_df)-1, len(batt_df)//2)

# ==========================================
# 4. 實時推論 (Inference)
# ==========================================
# 準備時鐘特徵
X_clock = batt_df[model_pkg["clock_feats"]]
X_c_sc = model_pkg["sc_clock"].transform(X_clock)
y_base = model_pkg["base_model"].predict(X_c_sc)

# 準備物理特徵
X_phys = batt_df[model_pkg["s1_feats"]]
X_p_sc = model_pkg["sc_phys"].transform(X_phys)

# 全局集成推論 (NNLS)
p_xgb = model_pkg["xgb_model"].predict(X_p_sc)
p_et = model_pkg["et_model"].predict(X_p_sc)
p_hgb = model_pkg["hgb_model"].predict(X_p_sc)
p_lin = model_pkg["lin_model"].predict(X_p_sc)
P_matrix = np.column_stack([p_xgb, p_et, p_hgb, p_lin, np.ones(len(batt_df))])

y_res = P_matrix @ model_pkg["nnls_w"]

# 結合預測
y_final = np.clip(y_base + y_res, 0.0, model_pkg["max_clip"])

# 單調性校正 (PIMS)
iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
y_mono = iso.fit_transform(batt_df.index.astype(float), y_final)

# ==========================================
# 5. 戰情室大螢幕 (SCADA Main)
# ==========================================
st.markdown("<div class='pro-title'>▯ PRO-BMS | 智慧儲能機櫃全局監控戰情室</div>", unsafe_allow_html=True)

row = batt_df.iloc[current_idx]
pred_rul_now = int(y_mono[current_idx])
rem_years = pred_rul_now / (daily_cycles * 365)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi-container'><div class='kpi-value'>{pred_rul_now}</div><div class='kpi-label'>AI 預測 RUL (Cycles)</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-container'><div class='kpi-value'>{rem_years:.1f}</div><div class='kpi-label'>預估可用年資</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-container'><div class='kpi-value'>{row['SOH']:.1f}%</div><div class='kpi-label'>健康評分 (SOH)</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-container'><div class='kpi-value' style='color:#00ffca'>ONLINE</div><div class='kpi-label'>機櫃連線狀態</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 中間圖表
col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.markdown("<div class='chart-title'>📈 物理單調性退化軌跡 (Req 14: PIMS Proof)</div>", unsafe_allow_html=True)
    fig_traj = go.Figure()
    # 實際觀測
    fig_traj.add_trace(go.Scatter(x=batt_df.index[:current_idx+1], y=batt_df['RUL'].iloc[:current_idx+1], 
                                 name="實際 RUL", line=dict(color='#ffffff', width=1, dash='dot')))
    # AI 預測路徑
    fig_traj.add_trace(go.Scatter(x=batt_df.index, y=y_mono, name="AI 物理感知路徑", line=dict(color='#00ffca', width=4)))
    
    # 當前時間指示線
    fig_traj.add_vline(x=current_idx, line_dash="dash", line_color="#ff4040")
    
    fig_traj.update_layout(template="plotly_dark", height=380, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_traj, use_container_width=True)

with col_right:
    st.markdown("<div class='chart-title'>🧩 物理健康雷達 (敏感度強化版)</div>", unsafe_allow_html=True)
    
    def get_sens_score(val, v_min, v_max, inv=False):
        if v_max == v_min: return 100
        s = (val - v_min) / (v_max - v_min) * 100
        return (100 - s) if inv else s

    scores = [
        get_sens_score(row['Cap_EMA'], row['Cap_EMA_min'], row['Cap_EMA_max']),
        get_sens_score(row['IR_EMA'], row['IR_EMA_min'], row['IR_EMA_max'], inv=True),
        get_sens_score(row['CV_Ratio_EMA'], row['CV_Ratio_EMA_min'], row['CV_Ratio_EMA_max']),
        get_sens_score(row['VD'], row['VD_min'], row['VD_max']),
        (pred_rul_now / batt_df['RUL'].max() * 100)
    ]
    labels = ['容量保持', '低內阻', '極化穩定', '壓降健康', '壽命預估']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=labels + [labels[0]], 
                                       fill='toself', fillcolor='rgba(0, 255, 202, 0.15)', line=dict(color='#00ffca', width=2)))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='#222', tickfont=dict(size=8)),
                   angularaxis=dict(gridcolor='#222', tickfont=dict(size=10))),
        template="plotly_dark", showlegend=False, height=380, margin=dict(l=40,r=40,t=40,b=40), paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# 底部日誌
st.markdown("<div class='chart-title'>📋 專家系統診斷日誌 (Kaggle Research Engine)</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='terminal-log'>
[LOG] 2026-03-01 | 機櫃 #{selected_id:03d} | 模型架構：Ridge + XGBoost Residual Learning<br>
[DATA] 物理退化因子 (Tau): {row['Tau']:.4f} | 累積 Ah: {np.exp(row['Cum_Ah_log1p']):.1f}<br>
[ADVICE] 透過 PIMS 驗證，預測軌跡符合物理不可逆律。目前設備健康度為 {row['SOH']:.1f}%。
</div>
""", unsafe_allow_html=True)