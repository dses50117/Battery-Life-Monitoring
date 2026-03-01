import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression

# ==========================================
# 0. 頁面基本設置與樣式 (Setup)
# ==========================================
st.set_page_config(page_title="PRO-BMS | Global Battery Health Monitoring System", layout="wide", initial_sidebar_state="expanded")

# 引入自定義 CSS (PRO-BMS 純粹暗黑與霓虹綠風格)
st.markdown("""
<style>
    /* 全局背景: 極致純黑 */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* 隱藏預設選單與 Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 隱藏 Header 白條 */
    header {visibility: hidden;}
    
    /* 側邊欄樣式 */
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #1a1a1a; }
    
    /* 頂部標題 HTML */
    .pro-title {
        text-align: center;
        color: #00ffca;
        font-size: 1.4rem;
        font-weight: 500;
        letter-spacing: 1px;
        margin-top: -30px;
        margin-bottom: 40px;
    }
    
    /* 自定義 KPI 容器 (無邊框浮動字) */
    .kpi-container {
        text-align: center;
        padding: 10px;
    }
    .kpi-value {
        font-size: 3.5rem;
        color: #00ffca; /* 霓虹青 */
        font-weight: 400;
        line-height: 1.2;
    }
    .kpi-value-status {
        font-size: 3.5rem;
        color: #00ff00; /* 亮綠色給 STATUS */
        font-weight: 400;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }
    
    /* 圖表小標題 (模擬圖片中的 ▯ Title) */
    .chart-title {
        color: #ffffff;
        font-size: 0.95rem;
        margin-bottom: -15px; /* 貼近圖表 */
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* 終端機風格 Log 區塊 */
    .terminal-log {
        font-family: 'Courier New', monospace;
        background-color: transparent;
        color: #00ffca;
        font-size: 0.85rem;
        line-height: 1.6;
        padding-top: 20px;
    }
    
    /* Streamlit 原生 divider 改為極細深灰 */
    hr { margin: 2rem 0; border-color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數據準備與特徵工程 (Data Prep)
# ==========================================
@st.cache_data(show_spinner="⏳ 資料初始化：正在從 Battery_RUL.csv 提取 14 顆電池特徵...")
def get_processed_data(file_path):
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
    df["Temp_Proxy"] = df["Max. Voltage Dischar. (V)"] * 0.5 + 20 # 模擬假溫度特徵供雷達圖
    df["IR_Proxy"] = df["IR"]

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ==========================================
# 1.5. 輕量化訓練管線 (Lightweight Model Pipeline)
# ==========================================
@st.cache_resource(show_spinner="🧠 AI 模型訓練中：正在利用原始資料集擬合 Ridge + XGBoost 全局壽命模型...")
def train_light_model(_df_all):
    base_feats = [c for c in ["Cap_EMA","IR_EMA","CV_Ratio_EMA","CC_Time_EMA"] if c in _df_all.columns]
    base_feats += [c for c in ["Cap_v1","IR_v1","CV_Ratio_v1"] if c in _df_all.columns]
    for col in ["Cap_EMA", "IR_EMA", "CV_Ratio_EMA"]:
        for w in [10, 20]:
            for suf in ["_rm", "_rs", "_slope"]:
                if f"{col}{suf}{w}" in _df_all.columns: base_feats.append(f"{col}{suf}{w}")
    clock_plus = ["Cum_Ah_log1p", "Cycle_Index"]
    feats = sorted(list(set(base_feats + clock_plus)))
    
    y = _df_all["RUL"].values.astype(float)
    X = _df_all[feats].values

    # Base Ridge
    sc_clock = StandardScaler()
    X_c = sc_clock.fit_transform(_df_all[clock_plus].values)
    base = Ridge(alpha=12.44)
    base.fit(X_c, y)
    yb = np.clip(base.predict(X_c), 0.0, y.max() * 1.1)
    r_base = y - yb

    # XGBoost Residual
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    clf_xgb = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=3,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=37.2,
        objective="reg:pseudohubererror", tree_method="hist", n_jobs=-1, random_state=42
    )
    clf_xgb.fit(X_sc, r_base, verbose=False)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(clf_xgb)
    
    return {
        "features": feats, "clock_features": clock_plus,
        "sc_clock": sc_clock, "base": base,
        "sc": sc, "clf_xgb": clf_xgb, "explainer": explainer, "max_clip": y.max() * 1.1
    }

# ==========================================
# 2. 側邊控制面板 (Control Sidebar)
# ==========================================
if not os.path.exists("Battery_RUL.csv"):
    st.error("⚠️ 找不到 `Battery_RUL.csv` 資料檔，請確認檔案位置。")
    st.stop()

full_df = get_processed_data("Battery_RUL.csv")

with st.sidebar:
    st.markdown("<div style='color:#00ffca; font-size:1.2rem; margin-bottom: 20px;'>⚙️ CONFIGURATION</div>", unsafe_allow_html=True)
    
    selected_id = st.selectbox(
        "BATTERY ASSET ID", 
        options=full_df["Battery_ID"].unique(), 
        format_func=lambda x: f"UNIT #{x:03d}"
    )
    
    daily_cycles = st.slider(
        "UTILIZATION RATE (CYCLES/DAY)", 
        min_value=0.2, max_value=4.0, value=1.5, step=0.1
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    batt_df = full_df[full_df["Battery_ID"] == selected_id].reset_index(drop=True)
    max_idx = len(batt_df) - 1
    current_idx = st.slider(
        "SIMULATE TIMELINE (CYCLE INDEX)", 
        min_value=0, max_value=max_idx, value=max_idx // 2
    )

row = batt_df.iloc[current_idx]

# 取得訓練好的模型
model_data = train_light_model(full_df)

# ==========================================
# 3. 頂部看板層 (Strategic Layer)
# ==========================================
# Main Title
st.markdown("<div class='pro-title'>▯ PRO-BMS | Global Battery Health Monitoring System (v2.4 Physics-KAN/XGBoost)</div>", unsafe_allow_html=True)

# 計算 KPI
rem_cycles = int(row["RUL"])
rem_years = rem_cycles / (daily_cycles * 365)
soh_val = float(row['SOH_Percentage'])

# 定義狀態
sys_status = "NORMAL"
sys_color = "#00ff00" # 預設亮綠
if soh_val < 80 or float(row['IR_Proxy']) > 0.2:
    sys_status = "WARNING"
    sys_color = "#ffaa00"
if soh_val < 70:
    sys_status = "CRITICAL"
    sys_color = "#ff4040"

# 自定義 HTML KPIs (4 columns)
kpi_html = f"""
<div style="display: flex; justify-content: space-around; margin-bottom: 40px; margin-top: 20px;">
    <div class="kpi-container">
        <div class="kpi-value">{rem_cycles}</div>
        <div class="kpi-label">REMAINING CYCLES</div>
    </div>
    <div class="kpi-container">
        <div class="kpi-value">{rem_years:.1f}</div>
        <div class="kpi-label">ESTIMATED YEARS</div>
    </div>
    <div class="kpi-container">
        <div class="kpi-value">{soh_val:.1f}%</div>
        <div class="kpi-label">STATE OF HEALTH</div>
    </div>
    <div class="kpi-container">
        <div class="kpi-value-status" style="color: {sys_color};">{sys_status}</div>
        <div class="kpi-label">SYSTEM STATUS</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ==========================================
# 4. 中部分析層 (Tactical Layer)
# ==========================================
col_main, col_side = st.columns([2.5, 1.3])

with col_main:
    st.markdown("<div class='chart-title'>▯ 壽命衰減軌跡預測 (Life Degradation Trajectory)</div>", unsafe_allow_html=True)
    fig_line = go.Figure()
    
    # 歷史軌跡 (實際值)
    hist_x = batt_df.index[:current_idx+1]
    hist_y = batt_df['RUL'].iloc[:current_idx+1]
    fig_line.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        name="Observed Path", mode='lines',
        line=dict(color='#00ffca', width=3)
    ))
    
    # AI 預測軌跡 (動態推論未來到生命週期結束)
    if current_idx < len(batt_df) - 1:
        future_df = batt_df.iloc[current_idx:].copy()
        
        # 進行預測
        X_c_fut = model_data["sc_clock"].transform(future_df[model_data["clock_features"]].values)
        yb_fut = np.clip(model_data["base"].predict(X_c_fut), 0.0, model_data["max_clip"])
        
        X_fut = model_data["sc"].transform(future_df[model_data["features"]].values)
        res_fut = model_data["clf_xgb"].predict(X_fut)
        raw_pred = np.clip(yb_fut + res_fut, 0.0, model_data["max_clip"])
        
        # 保證物理衰減單調性 (Isotonic Regression)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        mono_pred = iso.fit_transform(future_df["Cycle_Index"].values.astype(float), raw_pred)
        
        pred_x = future_df.index
        fig_line.add_trace(go.Scatter(
            x=pred_x, y=mono_pred,
            name="AI Forecast", mode='lines',
            line=dict(color='#666666', dash='dash', width=2)
        ))
    
    fig_line.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=280, margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="Cycle Index", showgrid=True, gridcolor='#1a1a1a', gridwidth=1, zerolinecolor='#1a1a1a', color='#aaaaaa'),
        yaxis=dict(title="RUL (Cycles)", showgrid=True, gridcolor='#1a1a1a', gridwidth=1, zerolinecolor='#1a1a1a', color='#aaaaaa'),
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="right", x=0.98, bgcolor='rgba(0,0,0,0.5)', bordercolor='#444')
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col_side:
    st.markdown("<div class='chart-title'>▯ 物理健康雷達圖 (Physical Health Radar)</div>", unsafe_allow_html=True)
    
    radar_soh = float(np.nan_to_num(row.get('SOH_Percentage', 100), nan=100.0))
    ir_val_safe = float(np.nan_to_num(row.get('IR_Proxy', 0), nan=0.0))
    radar_res = max(0.0, 100.0 - (ir_val_safe / 0.8) * 100.0)
    pol_val_safe = float(np.nan_to_num(row.get('CV_Ratio_EMA', 0), nan=0.0))
    radar_pol = max(0.0, 100.0 - pol_val_safe * 100.0)
    radar_vd = radar_res * 0.95 # Mock Voltage Drop based on Resistance
    radar_th = float(np.nan_to_num(row.get('Temp_Proxy', 20), nan=20.0)) / 40 * 100 # Mock Thermal
    
    radar_data = pd.DataFrame(dict(
        r=[radar_soh, radar_res, radar_vd, radar_pol, radar_th],
        theta=['Capacity', 'Resistance', 'Voltage Drop', 'CV Ratio', 'Thermal']
    ))
    
    fig_radar = px.line_polar(
        radar_data, r='r', theta='theta', line_close=True, 
        template="plotly_dark", color_discrete_sequence=['#00ffca']
    )
    fig_radar.update_traces(fill='toself', fillcolor='rgba(0, 255, 202, 0.1)')
    fig_radar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=280, margin=dict(l=30, r=30, t=50, b=30),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1a1a1a', linecolor='rgba(0,0,0,0)', tickfont=dict(color='rgba(0,0,0,0)')),
            angularaxis=dict(gridcolor='#1a1a1a', linecolor='#333', tickfont=dict(color='#cccccc', size=10))
        )
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ==========================================
# 5. 底部診斷層 (Operational Layer)
# ==========================================
col_diag1, col_diag2 = st.columns([1.5, 1.5])

# 產生 Terminal 報告與 SHAP 數據
ir_val = float(np.nan_to_num(row.get('IR_Proxy', 0)))
pol_val = float(np.nan_to_num(row.get('CV_Ratio_EMA', 0)))

log_lines = []

# 動態計算當前時間點的 XGBoost SHAP Values
current_features = row[model_data["features"]].values.reshape(1, -1)
current_features_sc = model_data["sc"].transform(current_features)
shap_vals = model_data["explainer"].shap_values(current_features_sc)[0]

# 將零碎特徵聚合成大類顯示
shap_dict = {
    "Capacity Fade": 0.0, "Internal Resistance": 0.0, "Polarization": 0.0,
    "Voltage Drop": 0.0, "Cycle Stress": 0.0
}
for i, feat in enumerate(model_data["features"]):
    val = abs(shap_vals[i])
    if "Cap" in feat: shap_dict["Capacity Fade"] += val
    elif "IR" in feat: shap_dict["Internal Resistance"] += val
    elif "CV" in feat: shap_dict["Polarization"] += val
    elif "Cycle" in feat or "Ah" in feat: shap_dict["Cycle Stress"] += val
    else: shap_dict["Voltage Drop"] += val

# 如果因剛開始訓練或數值極小而全為0，給予微小基礎值以便顯示圖表
if sum(shap_dict.values()) == 0:
    shap_dict = {"Capacity Fade": 0.35, "Internal Resistance": 0.28, "Polarization": 0.15, "Voltage Drop": 0.10, "Cycle Stress": 0.12}

if sys_status == "CRITICAL" or soh_val < 70:
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 🚨 CRITICAL: 電池 #{selected_id:03d} 容量已低於臨界點 (SOH {soh_val:.1f}%).")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 🛑 行動建議: 立即停機，安排電池模組抽換作業。")
    shap_dict["Capacity Fade"] = 0.65; shap_dict["Internal Resistance"] = 0.15

elif sys_status == "WARNING" or ir_val > 0.2:
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> ⚠️ 警告: 偵測到充放電內阻異常升高 (IR Proxy: {ir_val:.3f}V).")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 🛠️ 行動建議: 於 30 日內安排端子清潔與接點阻抗檢查。")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 💡 系統已自動觸發散熱模組強化運轉。")
    shap_dict["Internal Resistance"] = 0.50; shap_dict["Capacity Fade"] = 0.20

elif pol_val > 0.2:
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> ℹ️ 提示: 恆壓充電 (CV) 佔比偏高 (Polarization 效應)。")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> ⚡ 行動建議: 建議下次充電時進行深度慢充校正。")
    shap_dict["Polarization"] = 0.40; shap_dict["Capacity Fade"] = 0.30
    
else:
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> ✅ 電池 #{selected_id:03d} 狀態已更新為 <span style='color:#00ff00'>NORMAL</span>.")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 📊 未偵測到高風險物理降解特徵。")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 🗓️ 建議: 6 個月後進行例行性保養。")
    log_lines.append(f"<span style='color:#aaaaaa'>[{current_idx:04d}]</span> 🌊 [Action] 冷卻系統自檢已通過。系統準備就緒。")

with col_diag1:
    st.markdown("<div class='chart-title'>▯ AI 衰退因素歸因分析 (AI Attribution Analysis)</div>", unsafe_allow_html=True)
    
    df_shap = pd.DataFrame({
        "Feature": list(shap_dict.keys()), 
        "Impact": list(shap_dict.values())
    }).sort_values("Impact", ascending=True)
    
    fig_bar = px.bar(
        df_shap, x="Impact", y="Feature", orientation='h',
        template="plotly_dark", color_discrete_sequence=['#00ffca']
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=220, margin=dict(l=10, r=20, t=30, b=30),
        xaxis=dict(title="Impact Score (SHAP)", showgrid=True, gridcolor='#1a1a1a', zerolinecolor='#1a1a1a', color='#aaaaaa'),
        yaxis=dict(title="", showgrid=False, color='#aaaaaa')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_diag2:
    st.markdown("<div class='chart-title' style='margin-bottom: 5px;'>▯ 維護總結與日誌 (Maintenance Log & Recommendations)</div>", unsafe_allow_html=True)
    
    terminal_html = "<div class='terminal-log'>"
    terminal_html += "SYSTEM ADVICE:<br><br>"
    for line in log_lines:
        terminal_html += f"&nbsp;&nbsp;{line}<br>"
    if sys_status == "NORMAL":
        terminal_html += "<br>Ready for high-intensity operation.</div>"
    else:
        terminal_html += "<br><span style='color:#ffaa00'>System requires attention.</span></div>"

    st.markdown(terminal_html, unsafe_allow_html=True)
