import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from streamlit_autorefresh import st_autorefresh
from sklearn.isotonic import IsotonicRegression

# ==========================================
# 0. 頁面配置與極致黑 SCADA 風格 (Demo 單檔版 CSS)
# ==========================================
st.set_page_config(page_title="電池壽命監控看板", layout="wide")

st.markdown("""
<style>
    /* 隱藏頂部白條與選單 */
    header { visibility: hidden; height: 0px; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* 全局背景設為純黑 */
    .stApp { background-color: #000000; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #1a1a1a; }
    
    /* 強制文字白化 */
    .stWidgetLabel p, label, .stMarkdown p, [data-testid="stWidgetLabel"], [data-testid="stCheckbox"] label span {
        color: #ffffff !important; font-weight: bold !important;
    }

    /* 自訂 UI 元件 */
    .pro-header { background-color: #000000; border-bottom: 2px solid #333; padding: 20px; margin-top: -80px; margin-bottom: 25px; text-align: center; }
    .pro-title { color: #ffffff; font-size: 1.8rem; font-weight: bold; letter-spacing: 2px; }
    .kpi-container { text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.05); border: 1px solid #333; border-radius: 8px; }
    .kpi-value { font-size: 3rem; font-weight: 400; }
    .kpi-label { font-size: 0.8rem; color: #ffffff; text-transform: uppercase; }
    .chart-title { color: #ffffff; font-size: 1rem; border-left: 4px solid #00ffca; padding-left: 10px; margin-top: 10px; margin-bottom: 10px; }
    .terminal-log { font-family: 'Courier New', monospace; color: #00ffca; font-size: 0.85rem; padding: 10px; background: #050505; border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='pro-header'><div class='pro-title'>智慧儲能機櫃全局監控戰情室 (SCADA System)</div></div>", unsafe_allow_html=True)

# ==========================================
# 1. 數據與推論引擎層 (Functions)
# ==========================================
@st.cache_data(show_spinner="⏳ 正在同步數據...", ttl=3600)
def get_advanced_data(file_path):
    df = pd.read_csv(file_path)
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

def infer_battery(batt_df, model_pkg):
    X_c_sc = model_pkg["sc_clock"].transform(batt_df[model_pkg["clock_feats"]])
    y_base = model_pkg["base_model"].predict(X_c_sc)
    X_p_sc = model_pkg["sc_phys"].transform(batt_df[model_pkg["s1_feats"]])

    P = np.column_stack([
        model_pkg["xgb_model"].predict(X_p_sc), model_pkg["et_model"].predict(X_p_sc),
        model_pkg["hgb_model"].predict(X_p_sc), model_pkg["lin_model"].predict(X_p_sc), np.ones(len(batt_df))
    ])
    y_res = P @ model_pkg["nnls_w"]
    y_final = np.clip(y_base + y_res, 0.0, model_pkg["max_clip"])
    return IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(batt_df.index.astype(float), y_final)

# ==========================================
# 2. 啟動與狀態管理
# ==========================================
df_all = get_advanced_data("Battery_RUL.csv")
model_pkg = load_research_model()

with st.sidebar:
    st.markdown("<h2 style='color:#ffffff; border-bottom: 2px solid #444;'>⚙️ 控制台</h2>", unsafe_allow_html=True)
    selected_id = st.selectbox("分析機櫃單元", df_all["Battery_ID"].unique(), format_func=lambda x: f"電池陣列 #{x:03d}")
    batt_df = df_all[df_all["Battery_ID"] == selected_id].reset_index(drop=True)
    daily_cycles = st.slider("每日循環強度", 0.5, 3.0, 1.2)
    
    if 'current_idx' not in st.session_state: 
        st.session_state.current_idx = len(batt_df) // 2
        
    auto_play = st.toggle("啟動即時監控模擬 (Auto-Play)")

if auto_play:
    st_autorefresh(interval=500, limit=len(batt_df) - st.session_state.current_idx, key="auto_refresh")
    
    run_days = st.session_state.current_idx / daily_cycles
    st.sidebar.markdown(f"""
    <div style='padding: 10px; background: rgba(0, 255, 202, 0.1); border: 1px solid #00ffca; border-radius: 5px; text-align: center; margin-top: 10px; margin-bottom: 15px;'>
        <div style='font-size: 0.85rem; color: #ccc;'>▶️ 自動播放進度 (Cycle {st.session_state.current_idx})</div>
        <div style='font-size: 1.4rem; color: #00ffca; font-weight: bold;'>相當於已運轉 {run_days:.1f} 天</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.current_idx < len(batt_df) - 1:
        st.session_state.current_idx += 1
else:
    st.session_state.current_idx = st.sidebar.slider("時間軸模擬 (Cycle)", 0, len(batt_df)-1, st.session_state.current_idx)
    run_days = st.session_state.current_idx / daily_cycles
    st.sidebar.markdown(f"<div style='text-align: right; color: #00ffca; font-size: 0.9rem; margin-top: -15px; margin-bottom: 10px;'>相當於已運轉: <b>{run_days:.1f}</b> 天</div>", unsafe_allow_html=True)

# ==========================================
# 3. 實時推論與警報邏輯
# ==========================================
y_mono = infer_battery(batt_df, model_pkg)
row = batt_df.iloc[st.session_state.current_idx]
pred_rul_now = int(y_mono[st.session_state.current_idx])
rem_years = pred_rul_now / (daily_cycles * 365)

if row['SOH'] < 70.0:
    alarm_status, alarm_color = "CRITICAL (立即停機)", "#ff4040"
elif row['SOH'] < 85.0:
    alarm_status, alarm_color = "WARNING (偏離/注意)", "#ffb300"
else:
    alarm_status, alarm_color = "ONLINE (正常運轉)", "#00ffca"

# ==========================================
# 4. 畫面渲染
# ==========================================
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi-container'><div class='kpi-value' style='color:#00ffca'>{pred_rul_now}</div><div class='kpi-label'>AI 預測 RUL (Cycles)</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-container'><div class='kpi-value' style='color:#00ffca'>{rem_years:.1f}</div><div class='kpi-label'>預估可用年資</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-container'><div class='kpi-value' style='color:{alarm_color}'>{row['SOH']:.1f}%</div><div class='kpi-label'>健康評分 (SOH)</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-container'><div class='kpi-value' style='color:{alarm_color}; font-size:1.4rem; height:4.5rem; display:flex; align-items:center; justify-content:center;'>{alarm_status}</div><div class='kpi-label'>機櫃狀態</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1.2])

with col_left:
    st.markdown("<div class='chart-title'>📈 物理單調性退化軌跡</div>", unsafe_allow_html=True)
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=batt_df.index[:st.session_state.current_idx+1], y=batt_df['RUL'].iloc[:st.session_state.current_idx+1], 
        name="實際 RUL", mode='markers', marker=dict(color='#000000', size=6, line=dict(color='#ffffff', width=1))
    ))
    fig_traj.add_trace(go.Scatter(x=batt_df.index, y=y_mono, name="AI 預測路徑", line=dict(color='#00ffca', width=4)))
    fig_traj.add_vline(x=st.session_state.current_idx, line_dash="dash", line_color="#ff4040")
    
    fig_traj.update_layout(
        template="plotly_dark", height=380, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        legend=dict(font=dict(color='#ffffff'))
    )
    st.plotly_chart(fig_traj, use_container_width=True)

with col_right:
    st.markdown("<div class='chart-title'>🧩 物理健康雷達</div>", unsafe_allow_html=True)
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
    fig_radar.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], theta=labels + [labels[0]], 
        fill='toself', fillcolor='rgba(0, 255, 202, 0.15)', line=dict(color='#00ffca', width=2)
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='#333'), angularaxis=dict(gridcolor='#333')),
        template="plotly_dark", showlegend=False, height=380, margin=dict(l=40,r=40,t=40,b=40), paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

clip_count = len(df_all[(df_all["Battery_ID"] == selected_id) & (df_all["Discharge Time (s)"] >= 10000)])
st.markdown("<div class='chart-title'>📋 專家系統診斷與稽核日誌</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='terminal-log'>
[SYS] 單檔 Demo 模式啟動，使用前端定時器優化效能。<br>
[EVENT] 本機櫃共觸發 {clip_count} 次放電時間上限保護。<br>
[LOG] 2026-03-01 | 機櫃 #{selected_id:03d} | 當前 SOH: {row['SOH']:.1f}% | 狀態判定: <span style='color:{alarm_color}'>{alarm_status}</span>
</div>
""", unsafe_allow_html=True)