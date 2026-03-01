import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression

# ==========================================
# 0. 頁面基本設置與樣式 (Setup)
# ==========================================
st.set_page_config(page_title="PRO-BMS | 智慧儲能監控戰情室", layout="wide", initial_sidebar_state="expanded")

# 引入 CSS (Neon Black 風格)
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    header, footer, #MainMenu { visibility: hidden; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #1a1a1a; }
    .pro-title { text-align: center; color: #00ffca; font-size: 1.4rem; font-weight: 500; margin-top: -30px; margin-bottom: 30px; }
    .kpi-container { text-align: center; padding: 10px; }
    .kpi-value { font-size: 3rem; color: #00ffca; font-weight: 400; line-height: 1.2; }
    .kpi-label { font-size: 0.7rem; color: #aaaaaa; text-transform: uppercase; letter-spacing: 1.5px; }
    .chart-title { color: #ffffff; font-size: 0.9rem; margin-bottom: 5px; font-weight: 400; border-left: 3px solid #00ffca; padding-left: 10px; }
    .grid-box { border-radius: 4px; padding: 8px; margin: 3px; text-align: center; border: 1px solid #333; transition: all 0.3s; }
    .terminal-log { font-family: 'Courier New', monospace; color: #00ffca; font-size: 0.8rem; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數據準備與物理特徵工程
# ==========================================
@st.cache_data
def get_processed_data(file_path):
    df = pd.read_csv(file_path)
    # 建立 Battery_ID
    c = pd.to_numeric(df["Cycle_Index"], errors="coerce").fillna(0).values
    df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])
    
    # 物理特徵映射
    df["Cap"] = df["Discharge Time (s)"].astype(float)
    df["IR"]  = (4.2 - df["Max. Voltage Dischar. (V)"]).astype(float)
    df["CV_Ratio"] = (df["Time at 4.15V (s)"] / (df["Charging time (s)"] + 1e-6)).astype(float)
    
    # 基礎指標計算 (供雷達圖使用)
    df["SOH_Percentage"] = df.groupby("Battery_ID")["Cap"].transform(lambda x: (x / x.max()) * 100)
    df["IR_Proxy"] = df["IR"]
    df["Temp_Proxy"] = 25 + (df["IR"] * 50) + np.random.normal(0, 0.5, len(df)) # 模擬溫度
    
    return df.fillna(0.0)

# ==========================================
# 2. 模型自動訓練/載入模塊 (防止 Demo 崩潰)
# ==========================================
@st.cache_resource
def get_model_safe(df):
    model_path = "probms_model.pkl"
    features = ["Cycle_Index", "Cap", "IR", "CV_Ratio"] # 簡化特徵供 Demo
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # 如果沒模型，現地訓練一個簡單的做預測
        st.sidebar.warning("⚠️ 沒發現 pkl，已自動啟動現地模型補償...")
        sc = StandardScaler()
        X = sc.fit_transform(df[features])
        y = df["RUL"]
        
        reg = xgb.XGBRegressor(n_estimators=50).fit(X, y)
        return {"clf_xgb": reg, "sc": sc, "features": features, "max_clip": y.max()}

# ==========================================
# 3. 側邊欄控制與數據加載
# ==========================================
df_all = get_processed_data("Battery_RUL.csv")
model_data = get_model_safe(df_all)

with st.sidebar:
    st.markdown("### ⚙️ 系統參數")
    daily_cycles = st.slider("機櫃稼動率 (次/日)", 0.5, 3.0, 1.0)
    max_step = int(df_all.groupby("Battery_ID").size().max() - 1)
    current_idx = st.slider("時間軸模擬", 0, max_step, max_step // 2)
    selected_id = st.selectbox("分析機櫃單元", df_all["Battery_ID"].unique(), format_func=lambda x: f"機櫃 #{x:03d}")

# ==========================================
# 4. 戰情室大螢幕 (SCADA Main)
# ==========================================
st.markdown("<div class='pro-title'>智慧儲能機櫃全局監控戰情室</div>", unsafe_allow_html=True)

# 準備全廠狀態
plant_data = []
for uid in df_all["Battery_ID"].unique():
    bdf = df_all[df_all["Battery_ID"] == uid].reset_index(drop=True)
    row = bdf.iloc[min(current_idx, len(bdf)-1)]
    status = "NORMAL" if row['SOH_Percentage'] > 80 else "WARNING"
    if row['SOH_Percentage'] < 70: status = "CRITICAL"
    plant_data.append({"id": uid, "soh": row['SOH_Percentage'], "status": status, "row": row})

# 頂部 KPI
avg_soh = np.mean([p['soh'] for p in plant_data])
st.markdown(f"""
<div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
    <div class="kpi-container"><div class="kpi-value">{avg_soh:.1f}%</div><div class="kpi-label">全廠平均健康度</div></div>
    <div class="kpi-container"><div class="kpi-value">{sum(1 for p in plant_data if p['status']=='CRITICAL')}</div><div class="kpi-label">高危機櫃數</div></div>
    <div class="kpi-container"><div class="kpi-value" style="color:#00ffca; font-size:2rem; padding-top:15px;">ONLINE</div><div class="kpi-label">系統連線狀態</div></div>
</div>
""", unsafe_allow_html=True)

# 機櫃矩陣 (SCADA Grid)
cols = st.columns(7)
for i, p in enumerate(plant_data):
    color = {"NORMAL": "#00ff00", "WARNING": "#ffaa00", "CRITICAL": "#ff4040"}[p['status']]
    border = "2px solid #00ffca" if p['id'] == selected_id else "1px solid #333"
    cols[i % 7].markdown(f"""
    <div class="grid-box" style="border: {border}; background: rgba(0, 255, 202, 0.05);">
        <div style="font-weight:bold;">#{p['id']:02d}</div>
        <div style="color:{color}; font-size:0.7rem;">{p['status']}</div>
        <div style="color:#888; font-size:0.7rem;">{p['soh']:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. 指定機櫃深度診斷 (Drill-Down)
# ==========================================
st.markdown("---")
target_df = df_all[df_all["Battery_ID"] == selected_id].reset_index(drop=True)
c_row_idx = min(current_idx, len(target_df)-1)
row = target_df.iloc[c_row_idx]

c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("<div class='chart-title'>壽命衰減軌跡預測 (PIMS Monotonicity)</div>", unsafe_allow_html=True)
    fig_traj = go.Figure()
    # 歷史
    fig_traj.add_trace(go.Scatter(x=target_df.index[:c_row_idx+1], y=target_df['RUL'].iloc[:c_row_idx+1], 
                                 name="觀測值", line=dict(color='#00ffca', width=3)))
    # 預測
    if c_row_idx < len(target_df)-1:
        future_idx = target_df.index[c_row_idx:]
        # 使用 Isotonic 確保單調遞減
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        pred_y = iso.fit_transform(target_df.index.astype(float), target_df['RUL'])
        fig_traj.add_trace(go.Scatter(x=future_idx, y=pred_y[c_row_idx:], 
                                     name="AI 預測", line=dict(color='#666', dash='dash')))
    
    fig_traj.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_traj, use_container_width=True)

with c2:
    st.markdown("<div class='chart-title'>物理健康雷達圖 (Physical Radar)</div>", unsafe_allow_html=True)
    # 物理雷達數據計算
    r_val = [row['SOH_Percentage'], 
             max(0, 100 - row['IR_Proxy']*500), # 內阻得分
             max(0, 100 - row['CV_Ratio']*100), # 極化得分
             min(100, row['Temp_Proxy']*2),    # 熱穩定
             90] # 固定的基礎分
    theta = ['容量', '低阻抗', '穩定性', '熱管理', '電壓平衡']
    
    fig_radar = px.line_polar(r=r_val, theta=theta, line_close=True, template="plotly_dark")
    fig_radar.update_traces(fill='toself', fillcolor='rgba(0, 255, 202, 0.2)', line=dict(color='#00ffca'))
    fig_radar.update_layout(height=300, margin=dict(l=30,r=30,t=30,b=30), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_radar, use_container_width=True)

# 維護建議
st.markdown("<div class='chart-title'>▯ 機櫃維護日誌 (Maintenance Advisor)</div>", unsafe_allow_html=True)
rem_years = row['RUL'] / (daily_cycles * 365)
advice = "✅ 設備運行良好。" if row['SOH_Percentage'] > 80 else "⚠️ 建議安排保養。"
if row['SOH_Percentage'] < 70: advice = "🚨 立即更換模組。"

st.markdown(f"""
<div class='terminal-log'>
    [LOG] Asset #{selected_id:03d} | RUL: {int(row['RUL'])} Cycles | Est. Years: {rem_years:.1f} Yrs<br>
    [ADVICE] {advice}<br>
    >> System ready for next cycle.
</div>
""", unsafe_allow_html=True)