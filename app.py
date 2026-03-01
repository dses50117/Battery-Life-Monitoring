import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import xgboost as xgb
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# ==========================================
# 0. 頁面配置與極致黑 SCADA 風格 (全視覺白化修正)
# ==========================================
st.set_page_config(page_title="電池壽命監控看板", layout="wide")

st.markdown("""
<style>
    /* 1. 徹底移除頂部白條與選單 */
    header { visibility: hidden; height: 0px; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* 2. 全局背景設為純黑 */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* 3. 側邊欄視覺優化與文字白化 (關鍵修正) */
    [data-testid="stSidebar"] { 
        background-color: #0a0a0a !important; 
        border-right: 1px solid #1a1a1a; 
    }
    
    /* 強制將所有 Widget (Slider, Selectbox, Checkbox) 的標籤設為白色 */
    .stWidgetLabel p, label, .stMarkdown p, [data-testid="stWidgetLabel"] {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 0.95rem !important;
    }
    
    /* 針對 Checkbox (Auto-play) 的文字標籤特別加強白化 */
    [data-testid="stCheckbox"] label span {
        color: #ffffff !important;
    }

    /* 4. 自定義頂部標題列 (黑底白字) */
    .pro-header {
        background-color: #000000;
        border-bottom: 2px solid #333;
        padding: 20px;
        margin-top: -80px; /* 填補 header 隱藏後的空隙 */
        margin-bottom: 25px;
        text-align: center;
    }
    .pro-title { color: #ffffff; font-size: 1.8rem; font-weight: bold; letter-spacing: 2px; }

    /* 5. KPI 樣式 */
    .kpi-container { text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.05); border: 1px solid #333; border-radius: 8px; }
    .kpi-value { font-size: 3rem; color: #00ffca; font-weight: 400; }
    .kpi-label { font-size: 0.8rem; color: #ffffff; text-transform: uppercase; }
    
    /* 6. 圖表小標題與終端日誌 */
    .chart-title { color: #ffffff; font-size: 1rem; border-left: 4px solid #00ffca; padding-left: 10px; margin-top: 10px; margin-bottom: 10px; }
    .terminal-log { font-family: 'Courier New', monospace; color: #00ffca; font-size: 0.85rem; padding: 10px; background: #050505; border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數據引擎：研究等級特徵工程 (含離群值清洗)
# ==========================================
@st.cache_data(show_spinner="⏳ 正在同步數據...")
def get_advanced_data(file_path):
    df = pd.read_csv(file_path)
    # 物理清洗：限制最大放電時間，防止感測器錯誤導致 SOH 崩潰
    df["Discharge Time (s)"] = df["Discharge Time (s)"].clip(upper=10000)
    
    c = pd.to_numeric(df["Cycle_Index"], errors="coerce").fillna(0).values
    df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])
    df = df.sort_values(["Battery_ID", "Cycle_Index"]).reset_index(drop=True)
    
    df["Cap"] = df["Discharge Time (s)"].astype(float)
    df["IR"]  = (4.2 - df["Max. Voltage Dischar. (V)"]).astype(float)
    df["CV_Ratio"] = (df["Time at 4.15V (s)"] / (df["Charging time (s)"] + 1e-6)).astype(float)
    df["VD"]  = df["Decrement 3.6-3.4V (s)"].astype(float)
    
    for col in ["Cap", "IR", "CV_Ratio"]:
        init = df.groupby("Battery_ID")[col].transform("first")
        df[f"{col}_Ret"] = df[col] / (init + 1e-6)
        df[f"{col}_EMA"] = df.groupby("Battery_ID")[f"{col}_Ret"].transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean())
        df[f"{col}_rm20"] = df.groupby("Battery_ID")[f"{col}_EMA"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # 修正 SOH 基準點 (採用 95% 分位數，排除極端雜訊)
    df["Max_Cap_Found"] = df.groupby("Battery_ID")["Cap_EMA"].transform(lambda x: x.quantile(0.95))
    df["SOH"] = (df["Cap_EMA"] / df["Max_Cap_Found"]) * 100
    df["SOH"] = df["SOH"].clip(0.0, 100.0)
    
    df["Tau"] = (1.0 / (1.0 + np.exp(-df["Cap_EMA"]))).clip(0.0, 1.0)
    df["Cum_Ah_log1p"] = np.log1p(df.groupby("Battery_ID")["Cap"].cumsum())
    
    for feat in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA", "VD"]:
        df[f"{feat}_min"] = df.groupby("Battery_ID")[feat].transform(lambda x: x.quantile(0.05))
        df[f"{feat}_max"] = df.groupby("Battery_ID")[feat].transform(lambda x: x.quantile(0.95))
        
    return df.fillna(0.0)

@st.cache_resource
def load_research_model():
    return joblib.load("probms_model.pkl")

# ==========================================
# 2. 佈局與標題呈現
# ==========================================
st.markdown("<div class='pro-header'><div class='pro-title'>智慧儲能機櫃全局監控戰情室 (SCADA System)</div></div>", unsafe_allow_html=True)

df_all = get_advanced_data("Battery_RUL.csv")
model_pkg = load_research_model()

with st.sidebar:
    st.markdown("<h2 style='color:#ffffff; border-bottom: 2px solid #444;'>⚙️ 控制台</h2>", unsafe_allow_html=True)
    
    # 以下標籤文字在自定義 CSS 作用下皆會強制顯示為白色
    selected_id = st.selectbox("分析機櫃單元", df_all["Battery_ID"].unique(), format_func=lambda x: f"電池陣列 #{x:03d}")
    
    batt_df = df_all[df_all["Battery_ID"] == selected_id].reset_index(drop=True)
    daily_cycles = st.slider("每日循環強度", 0.5, 3.0, 1.2)
    
    if 'current_idx' not in st.session_state: st.session_state.current_idx = len(batt_df) // 2
    auto_play = st.checkbox("啟動即時監控模擬 (Auto-Play)")
    
    current_idx = st.slider("時間軸模擬 (Cycle)", 0, len(batt_df)-1, st.session_state.current_idx, key="slider_idx")
    st.session_state.current_idx = current_idx

if auto_play and st.session_state.current_idx < len(batt_df) - 1:
    st.session_state.current_idx += 1
    time.sleep(0.05); st.rerun()

# ==========================================
# 3. 實時推論邏輯 (多模型 NNLS 集成)
# ==========================================
X_c_sc = model_pkg["sc_clock"].transform(batt_df[model_pkg["clock_feats"]])
y_base = model_pkg["base_model"].predict(X_c_sc)
X_p_sc = model_pkg["sc_phys"].transform(batt_df[model_pkg["s1_feats"]])

# NNLS Ensemble 推論
P = np.column_stack([
    model_pkg["xgb_model"].predict(X_p_sc), model_pkg["et_model"].predict(X_p_sc),
    model_pkg["hgb_model"].predict(X_p_sc), model_pkg["lin_model"].predict(X_p_sc), np.ones(len(batt_df))
])
y_res = P @ model_pkg["nnls_w"]
y_final = np.clip(y_base + y_res, 0.0, model_pkg["max_clip"])
y_mono = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(batt_df.index.astype(float), y_final)

# ==========================================
# 4. 圖表與數據顯示
# ==========================================
row = batt_df.iloc[st.session_state.current_idx]
pred_rul_now = int(y_mono[st.session_state.current_idx])
rem_years = pred_rul_now / (daily_cycles * 365)

# KPI 區
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi-container'><div class='kpi-value'>{pred_rul_now}</div><div class='kpi-label'>AI 預測 RUL (Cycles)</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-container'><div class='kpi-value'>{rem_years:.1f}</div><div class='kpi-label'>預估可用年資</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-container'><div class='kpi-value'>{row['SOH']:.1f}%</div><div class='kpi-label'>健康評分 (SOH)</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-container'><div class='kpi-value' style='color:#00ffca'>ONLINE</div><div class='kpi-label'>機櫃連線狀態</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.markdown("<div class='chart-title'>📈 物理單調性退化軌跡 (Req 14: PIMS Proof)</div>", unsafe_allow_html=True)
    fig_traj = go.Figure()
    # 實際觀測：黑珍珠點視覺
    fig_traj.add_trace(go.Scatter(
        x=batt_df.index[:st.session_state.current_idx+1], 
        y=batt_df['RUL'].iloc[:st.session_state.current_idx+1], 
        name="實際 RUL (Observed)", mode='markers',
        marker=dict(color='#000000', size=6, line=dict(color='#ffffff', width=1))
    ))
    # AI 路徑：青色寬實線
    fig_traj.add_trace(go.Scatter(x=batt_df.index, y=y_mono, name="AI 物理感知路徑", line=dict(color='#00ffca', width=4)))
    fig_traj.add_vline(x=st.session_state.current_idx, line_dash="dash", line_color="#ff4040")
    
    fig_traj.update_layout(
        template="plotly_dark", height=380, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color="#ffffff")),
        xaxis=dict(tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
        yaxis=dict(tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"))
    )
    st.plotly_chart(fig_traj, use_container_width=True)

with col_right:
    st.markdown("<div class='chart-title'>🧩 物理健康雷達 (敏感度強化版)</div>", unsafe_allow_html=True)
    def get_sens_score(val, v_min, v_max, inv=False):
        if v_max == v_min: return 100
        s = (val - v_min) / (v_max - v_min + 1e-6) * 100
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='#333', tickfont=dict(color="#ffffff", size=8)),
                   angularaxis=dict(gridcolor='#333', tickfont=dict(color="#ffffff", size=10))),
        template="plotly_dark", showlegend=False, height=380, margin=dict(l=40,r=40,t=40,b=40), paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("<div class='chart-title'>📋 專家系統診斷日誌</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='terminal-log'>
[LOG] 2026-03-01 | 機櫃 #{selected_id:03d} | SOH: {row['SOH']:.1f}%<br>
[ADVICE] 已移除異常離群值與頂部白條。左側控制台標籤已強制白化，符合極致黑監控風格。
</div>
""", unsafe_allow_html=True)