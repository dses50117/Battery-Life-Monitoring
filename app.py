import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================
# 1. 數據準備與特徵工程 (Data Prep)
# ==========================================
@st.cache_data
def get_processed_data(file_path):
    df = pd.read_csv(file_path)
    # 識別不同電池
    c = pd.to_numeric(df["Cycle_Index"], errors="coerce").fillna(0).values
    df["Battery_ID"] = np.cumsum((c[1:] <= c[:-1]).astype(int).tolist() + [0])
    
    # 物理特徵提取 (物理感知模組)
    df["SOH_Proxy"] = df.groupby("Battery_ID")["Discharge Time (s)"].transform(lambda x: x.ewm(alpha=0.1).mean())
    df["Max_SOH"] = df.groupby("Battery_ID")["SOH_Proxy"].transform("max")
    df["SOH_Percentage"] = (df["SOH_Proxy"] / df["Max_SOH"]) * 100
    
    # 計算內阻代理指標
    df["IR_Proxy"] = (4.2 - df["Max. Voltage Dischar. (V)"]).ewm(alpha=0.1).mean()
    return df

# ==========================================
# 2. UI 配置與交互設計 (Deployment)
# ==========================================
st.set_page_config(page_title="Expert BMS Monitor", layout="wide")

# 工業風深色主題
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { border: 1px solid #464b5d; padding: 20px; border-radius: 10px; background: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# 載入數據
if not os.path.exists("Battery_RUL.csv"):
    st.error("請確保 Battery_RUL.csv 與 app.py 在同一資料夾下")
    st.stop()

full_df = get_processed_data("Battery_RUL.csv")

# 側邊欄控制
st.sidebar.title("🛠️ CRISP-DM 監控後台")
selected_id = st.sidebar.selectbox("選擇電池序號", full_df["Battery_ID"].unique())
daily_cycles = st.sidebar.slider("每日循環次數 (Intensity)", 0.2, 3.0, 1.0)

# 過濾當前電池數據
batt_df = full_df[full_df["Battery_ID"] == selected_id].reset_index(drop=True)
current_idx = st.sidebar.slider("模擬時間線", 0, len(batt_df)-1, len(batt_df)//2)
row = batt_df.iloc[current_idx]

# ==========================================
# 3. 看板核心內容 (Business Value)
# ==========================================
st.title("🔋 電力系統專家：電池健康度與 RUL 監控系統")
st.markdown(f"**電池 ID:** `{selected_id}` | **數據狀態:** 物理感知特徵已同步")

# 第一排：核心 KPI
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
rem_cycles = int(row["RUL"])
rem_years = rem_cycles / (daily_cycles * 365)

kpi1.metric("剩餘循環 (RUL)", f"{rem_cycles} 次")
kpi2.metric("預計剩餘年資", f"{rem_years:.1f} 年")
kpi3.metric("健康狀態 (SOH)", f"{row['SOH_Percentage']:.1f}%")
kpi4.metric("電壓下降代理", f"{row['IR_Proxy']:.3f} V")

st.markdown("---")

# 第二排：圖表分析
col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("📊 壽命退化趨勢 (Monotonic Path)")
    fig = go.Figure()
    # 歷史區
    fig.add_trace(go.Scatter(x=batt_df.index[:current_idx+1], y=batt_df['RUL'][:current_idx+1],
                             name="歷史數據", line=dict(color='#00ffcc', width=4)))
    # 預測區
    fig.add_trace(go.Scatter(x=batt_df.index, y=batt_df['RUL'],
                             name="預測趨勢", line=dict(color='gray', dash='dash', width=1)))
    
    fig.update_layout(template="plotly_dark", xaxis_title="Time Steps", yaxis_title="RUL")
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.subheader("🧩 物理特徵雷達")
    # 展示各項物理指標的當前位置
    radar_data = pd.DataFrame(dict(
        r=[row['SOH_Percentage'], row['IR_Proxy']*200, row['CV_Ratio_EMA']*100 if 'CV_Ratio_EMA' in row else 50],
        theta=['SOH','Resistance','Polarization']))
    fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True, template="plotly_dark")
    st.plotly_chart(fig_radar, use_container_width=True)

# 第三排：原始數據
st.subheader("📋 實時感測器流 (Sensor Stream)")
st.dataframe(batt_df.iloc[max(0, current_idx-5):current_idx+1], use_container_width=True)
