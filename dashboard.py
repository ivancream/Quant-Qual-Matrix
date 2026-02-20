import os
import time
import concurrent.futures
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
# Import Engines (Must be imported after config to ensure path is correct, but here is fine)
from engines import engine_global
from engines import engine_fundamental as fundamental_eng
from engines import engine_technical as ta
from engines import engine_chips
from engines import engine_ai
# ==========================================
# 0. Config & Setup (MUST BE FIRST)
# ==========================================
st.set_page_config(
    page_title="AlphaScan | Quant-Qual Matrix",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Force reload .env to pick up API key changes immediately
load_dotenv(override=True)

# 優先從 st.secrets 取得 Key (Streamlit Cloud 規範)，若無則從環境變數取得
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Missing GEMINI_API_KEY: Please set it in Streamlit Secrets or .env file.")
    st.stop()
# CSS Styling - Dark Mode & Professional Finance Style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    /* Global Base */
    .stApp {
        background-color: #050505;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 95%; /* Fluid layout to use all space */
    }
    /* Typography */
    h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 1rem;
    }
    /* Custom Card Style */
    .metric-card {
        background-color: #1A1A1A;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F9A825;
        margin-bottom: 1rem;
    }
    /* Metrics Appearance */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F9A825;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Inputs & UI Elements */
    .stTextInput > div > div > input {
        background-color: #121212;
        color: #FFFFFF;
        border: 1px solid #333333;
        border-radius: 4px;
        height: 40px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #F9A825;
    }
    /* Action Button (Premium Mobile Style) */
    .stButton > button {
        background: linear-gradient(135deg, #F9A825 0%, #F57F17 100%);
        color: #000000;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        transition: all 0.2s ease;
        width: 100%;
        text-transform: none;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(249, 168, 37, 0.4);
    }
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #050505;
        padding: 4px;
        border-bottom: 1px solid #222222;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        color: #888888;
        font-weight: 500;
        padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #F9A825;
        border-bottom: 2px solid #F9A825;
    }
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: #222222;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #050505;
    }
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 10px;
    }
    /* RWD: Mobile Optimization */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        h1 { font-size: 1.5rem; }
        h2, h3 { font-size: 1.2rem; }
        /* Stack columns vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 auto !important;
            min-width: unset !important;
        }
        /* Adjust tabs for mobile scrolling */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            -webkit-overflow-scrolling: touch;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0 12px;
            font-size: 0.9rem;
        }
    }
    /* 隱藏報鎖底部的 Ask ChatGPT / Ask Google 連結 */
    [data-testid="stException"] footer, 
    [data-testid="stException"] .stExceptionExplanation {
        display: none !important;
    }
    /* 放大 File Uploader 內部灰色區域 */
    [data-testid="stFileUploader"] section {
        padding: 2rem 1rem !important;
        background-color: #1E1E1E !important;
        border-radius: 10px !important;
    }
    /* 置中標題與平均分散 Tabs */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        width: 100%;
        gap: 20px; /* 增加間距 */
        border-bottom: 2px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1; /* 平均分散 */
        text-align: center;
        padding: 1.5rem !important; /* 加大內距 */
        font-size: 1.8rem !important; /* 字體狠命放大 */
        font-weight: 900 !important;
        letter-spacing: 2px; /* 增加字距 */
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFB74D !important;
    }
    /* Remove stForm border */
    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    </style>
""", unsafe_allow_html=True)
# ==========================================
# 1. State Management (Controller)
# ==========================================
# Initialize Session State
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {
        'global': None,
        'fundamental': None,
        'technical': None,
        'chips': None
    }
if 'global_report' not in st.session_state:
    st.session_state.global_report = None
if 'fund_report' not in st.session_state:
    st.session_state.fund_report = None
if 'tech_messages' not in st.session_state:
    st.session_state.tech_messages = []
if 'chips_report' not in st.session_state:
    st.session_state.chips_report = None
if 'chips_chat' not in st.session_state:
    st.session_state.chips_chat = []
if 'global_macro_data' not in st.session_state:
    st.session_state.global_macro_data = None
if 'global_chat' not in st.session_state:
    st.session_state.global_chat = []
if 'global_report_content' not in st.session_state:
    st.session_state.global_report_content = None
if 'fund_chat' not in st.session_state:
    st.session_state.fund_chat = []
if 'tech_history' not in st.session_state:
    st.session_state.tech_history = []
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'details_map' not in st.session_state:
    st.session_state.details_map = None
if 'slope_val' not in st.session_state:
    st.session_state.slope_val = 0.003
if 'vol_mul' not in st.session_state:
    st.session_state.vol_mul = 1.5
# 統一 AI Brain（跨模組記憶系統）
if 'ai_brain' not in st.session_state:
    st.session_state.ai_brain = engine_ai.get_empty_brain()
# API Key Check (已在上方完成初始化，此處可選留作二次確認)
if not api_key:
    st.stop()
# ==========================================
# 2. Global UI Elements
# ==========================================
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0rem; padding-bottom: 0rem;'>全方位個股檢視神器</h1>", unsafe_allow_html=True)
# ==========================================
# 3. Logic & Data Flow (Internal)
# ==========================================
# We no longer have a global Scan button at the top.
# Individual tabs will handle their own symbol input.
if 'active_symbol' not in st.session_state:
    st.session_state.active_symbol = "2330" # Default
active_symbol_display = st.session_state.active_symbol
# ==========================================
# 4. View Components (Tabs)
# ==========================================
tab_names = ["國際大盤", "基本概況", "技術指標", "籌碼分析", "波段多方", "波段空方"]
tabs = st.tabs(tab_names)
# --- Tab 1: Market Intelligence ---
with tabs[0]:
    st.markdown("### 大盤跌不是病，跌了不反彈才是命")
    # --- 1. 各項總經指標 (頂部固定區域) ---
    # 標題與更新按鈕
    h_col1, h_col2 = st.columns([6, 1])
    with h_col1:
        st.markdown("#### 全球總經核心指標")
    with h_col2:
        # 改為點擊按鈕後直接在該區塊內抓取數據
        if st.button("Scan", use_container_width=True):
            with st.spinner("正在進行數據開採"):
                st.session_state.macro_data_cache = {
                    "short": engine_global.get_short_term_data(),
                    "long": engine_global.get_long_term_data()
                }
    # 檢查是否有數據快取，若無則顯示提示，不自動抓取
    if st.session_state.get('macro_data_cache') is not None:
        # 數據解析
        s_data = st.session_state.macro_data_cache["short"]
        l_data = st.session_state.macro_data_cache["long"]
        # Row 1: 短線情緒
        st.markdown("**短線情緒 (決定攻擊力道)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("台幣匯率 (TWD)", s_data.get('twd', 'N/A'))
        c2.metric("VIX 恐慌指數", s_data.get('vix', 'N/A'))
        c3.metric("美債 10 年殖利率", s_data.get('bond', 'N/A'))
        c4.metric("黃金 (Gold)", s_data.get('gold', 'N/A'))
        st.markdown("") # Spacer
        # Row 2: 波段趨勢
        st.markdown("**波段趨勢 (決定持股水位)**")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("美國 CPI", l_data.get('cpi', 'N/A'))
        c6.metric("外銷訂單 (TW)", l_data.get('export', 'N/A'))
        c7.metric("PMI 指數", l_data.get('pmi', 'N/A'))
        c8.metric("景氣燈號", l_data.get('signal', 'N/A'))
    else:
        st.info("目前尚未獲取總經數據。請點擊上方按鈕以啟動數據開採流程。")
    st.divider()
    # --- 2. 各國盤勢概況 ---
    st.markdown("#### 各國盤勢概況")
    # 將選擇選單放在左側，按鈕放在右側，比例調整
    c_select, c_action = st.columns([6, 1])
    with c_select:
        market_select = st.selectbox("選擇市場", list(engine_global.MARKET_MAP.keys()), label_visibility="collapsed")
    with c_action:
        generate_btn = st.button("Analyze", use_container_width=True, type="primary")
    if generate_btn:
        if st.session_state.get('macro_data_cache') is None:
            st.error("請先點擊上方「Refresh」獲取總經指標，再產出分析報告。")
        else:
            with st.status("正在整合各項數據...", expanded=False) as status:
                metrics, m_status = engine_global.get_market_metrics(market_select)
                fx_val = engine_global.get_fx_data(market_select)
                news_summary = engine_global.get_investing_news(market_select)
                # 從快取取得 macro 數據
                s_data = st.session_state.macro_data_cache["short"]
                l_data = st.session_state.macro_data_cache["long"]
                macro_str = f"VIX: {s_data.get('vix')} | Bond: {s_data.get('bond')}"
                commo_str = f"Gold: {s_data.get('gold')} | CPI: {l_data.get('cpi')}"
                report = engine_global.generate_global_report(market_select, metrics, macro_str, fx_val, commo_str, news_summary)
                st.session_state.global_report_content = {
                    "market": market_select,
                    "metrics": metrics,
                    "status": m_status,
                    "rep_text": report
                }
                # 重置該區塊對談
                st.session_state.global_chat = []
                status.update(label="區域分析完成", state="complete")
    # --- 3. 分析結果與對談區 ---
    if st.session_state.global_report_content:
        res = st.session_state.global_report_content
        st.markdown(f"## {res['market']} 深度決策分析")
        # 指數指標列
        m1, m2 = st.columns(2)
        m1.metric("指數水位", res["metrics"])
        m2.metric("短期情緒", res["status"])
        st.markdown(res["rep_text"])
        st.divider()
        st.subheader("宏觀策略對談")
        # 顯示歷史對話
        for msg in st.session_state.global_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if g_chat_in := st.chat_input("詢問有關宏觀趨勢的問題..."):
            st.session_state.global_chat.append({"role": "user", "content": g_chat_in})
            with st.chat_message("user"): st.markdown(g_chat_in)
            with st.chat_message("assistant"):
                with st.spinner("策略師正在思考..."):
                    ans = engine_global.chat_with_global_analyst(g_chat_in, res["rep_text"], st.session_state.global_chat)
                    st.markdown(ans)
            st.session_state.global_chat.append({"role": "assistant", "content": ans})
    else:
        st.info("請選擇上方市場後點擊「Analyze」。")
# --- Tab 2: Quality & Value ---
with tabs[1]:
    st.subheader("本來想短線，賠錢變長線，長線變貢獻")
    # Context-specific Search
    with st.form(key="fund_search_form"):
        c1, c2, _ = st.columns([2, 1, 5])
        with c1: f_input = st.text_input("輸入股票代號", placeholder="e.g. 2330, NVDA", label_visibility="collapsed")
        with c2: f_scan = st.form_submit_button("Scan", use_container_width=True)
    if st.session_state.get('fund_active_symbol'):
        st.caption(f"Currently analyzing: {st.session_state.fund_active_symbol}")
    if f_scan and f_input:
        target = f_input.strip().upper()
        # Fetch name
        name = ta.get_symbol_name(target)
        st.session_state.fund_active_symbol = f"{target} {name}" if name else target
        with st.status(f"正在對 {st.session_state.fund_active_symbol} 進行財報深挖...", expanded=True) as status:
            st.write("讀取財務報表與估值中...")
            ticker_data = fundamental_eng.get_stock_info(target)
            if ticker_data:
                st.session_state.data_cache['fundamental_data'] = ticker_data
                st.write("分析輿情與市場情緒...")
                news_summary = fundamental_eng.get_sentiment_summary(target)
                st.write("撰寫分析報告...")
                report = fundamental_eng.generate_ai_report(target, ticker_data, news_summary)
                st.session_state.fund_report = report
                # 將基本面摘要對 engine_ai Brain
                fund_summary = f"P/E: {ticker_data.get('pe','N/A')} | ROE: {ticker_data.get('roe','N/A')} | 營收成長: {ticker_data.get('growth','N/A')} | 毛利率: {ticker_data.get('margin','N/A')}\n\n{report[:500]}"
                st.session_state.ai_brain = engine_ai.update_context(st.session_state.ai_brain, target, "fundamental", fund_summary)
                status.update(label="基本面分析完成", state="complete", expanded=False)
            else:
                st.error("數據獲取失敗。")
                status.update(label="分析失敗", state="error")
    # Key Metrics Row
    if st.session_state.data_cache.get('fundamental_data'):
        data = st.session_state.data_cache['fundamental_data']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P/E Ratio", str(data.get('pe', '-')))
        c2.metric("ROE", str(data.get('roe', '-')))
        c3.metric("Rev Growth", str(data.get('growth', '-')))
        c4.metric("Gross Margin", str(data.get('margin', '-')))
        # st.caption(data.get('technicals', '')) # 移除技術指標摘要
        st.divider()
    # Report & Chat Section
    if st.session_state.fund_report:
        st.markdown(st.session_state.fund_report)
        st.divider()
        st.subheader("Analyst Q&A")
        # 顯示對話歷史（來自統一 Brain）
        brain = st.session_state.ai_brain
        for msg in brain.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        active_ticker_brain = brain.get("active_ticker") or ""
        if q := st.chat_input("詢問基本面、財報或其他問題...", key="fund_chat_input"):
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                stream, update_fn = engine_ai.stream_chat(brain, active_ticker_brain, q)
                ans = st.write_stream(stream)
                st.session_state.ai_brain = update_fn(ans)
    else:
        st.info("Initiate a scan to generate quality analysis.")
# --- Tab 3: Technical Pulse ---
with tabs[2]:
    st.subheader("被軋是一陣子，被套是一輩子")
    # Context-specific Search
    with st.form(key="tech_search_form"):
        c1, c2, _ = st.columns([2, 1, 5])
        with c1: t_search_input = st.text_input("輸入股票代號", placeholder="e.g. 2330, NVDA", label_visibility="collapsed")
        with c2: t_scan = st.form_submit_button("Scan", use_container_width=True)
    if st.session_state.get('tech_active_symbol'):
         st.caption(f"Currently analyzing: {st.session_state.tech_active_symbol}")
    if t_scan and t_search_input:
        target = t_search_input.strip().upper()
        name = ta.get_symbol_name(target)
        st.session_state.tech_active_symbol = f"{target} {name}" if name else target
        # 同步更新全域 active_symbol 避免 AI 錯亂
        st.session_state.active_symbol = target 
        raw_df = ta.fetch_data(target)
        if raw_df is not None:
            processed_df = ta.calculate_indicators(raw_df)
            st.session_state.data_cache['technical_df'] = processed_df
            # Reset technical chat messages for new symbol
            st.session_state.tech_messages = []
            st.session_state.tech_history = []
        else:
            st.error(f"無法載入 {target} 的數據")
    df = st.session_state.data_cache.get('technical_df')
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        # Vertical Layout: Chart Top -> Strategy Bottom
        # Chart Controls row
        c1, c2 = st.columns(2)
        with c1: sub1 = st.selectbox("Indicator 1", ["Volume", "MACD", "RSI", "KD", "Bias", "OBV"], index=0)
        with c2: sub2 = st.selectbox("Indicator 2", ["Volume", "MACD", "RSI", "KD", "Bias", "OBV"], index=3)
        plot_data = df.tail(150).copy()
        plot_data['DateStr'] = plot_data.index.strftime('%Y-%m-%d')
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        # Candlestick (Row 1)
        fig.add_trace(go.Candlestick(
            x=plot_data['DateStr'],
            open=plot_data['Open'], high=plot_data['High'],
            low=plot_data['Low'], close=plot_data['Close'],
            name='Price', increasing_line_color='#FF5252', decreasing_line_color='#00E676',
            hovertemplate='Close: %{close:.2f}<extra></extra>'
        ), row=1, col=1)
        colors = {'MA5': '#F9A825', 'MA10': '#00E676', 'MA20': '#29B6F6', 'MA60': '#AB47BC'}
        for ma, color in colors.items():
            if ma in plot_data.columns:
                    fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data[ma], mode='lines', 
                                            name=ma, line={'color': color, 'width': 1.2},
                                            hoverinfo='skip'), row=1, col=1)
        if 'BB_Upper' in plot_data.columns:
            # Light Grey Dashed Lines for Bollinger Bands
            fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['BB_Upper'], 
                                     line={'color': '#CCCCCC', 'width': 1, 'dash': 'dash'}, 
                                     name='BB Upper', hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['BB_Lower'], 
                                     line={'color': '#CCCCCC', 'width': 1, 'dash': 'dash'}, 
                                     fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', 
                                     name='BB Lower', hoverinfo='skip'), row=1, col=1)
        def plot_indicator(name, row):
            if name == 'Volume':
                bar_colors = ['#FF5252' if r['Close'] >= r['Open'] else '#00E676' for _, r in plot_data.iterrows()]
                fig.add_trace(go.Bar(x=plot_data['DateStr'], y=plot_data['Volume'], name='Volume', marker_color=bar_colors, opacity=0.8), row=row, col=1)
            elif name == 'RSI':
                fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['RSI'], name='RSI', line={'color': '#F9A825'}), row=row, col=1)
                fig.add_hline(y=70, row=row, col=1, line_dash="dot", line_color="#FF5252", opacity=0.5)
                fig.add_hline(y=30, row=row, col=1, line_dash="dot", line_color="#00E676", opacity=0.5)
            elif name == 'MACD':
                fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['MACD'], name='MACD', line={'color': '#29B6F6'}), row=row, col=1)
                fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['Signal'], name='Signal', line={'color': '#E0E0E0'}), row=row, col=1) # Signal white/grey
                # Color Hist based on value
                hist_colors = ['#FF5252' if v >= 0 else '#00E676' for v in plot_data['Hist']]
                fig.add_trace(go.Bar(x=plot_data['DateStr'], y=plot_data['Hist'], name='Hist', marker_color=hist_colors), row=row, col=1)
            elif name == 'Bias':
                    # Bias with threshold lines
                    fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['Bias_20'], name='Bias (20)', line={'color': '#F9A825'}), row=row, col=1)
                    fig.add_hline(y=0, row=row, col=1, line_dash="solid", line_color="#888888", opacity=0.5)
            elif name == 'KD':
                fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['K'], name='K線', line={'color': '#F9A825'}), row=row, col=1)
                fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['D'], name='D線', line={'color': '#29B6F6'}), row=row, col=1)
                fig.add_hline(y=80, row=row, col=1, line_dash="dot", line_color="#FF5252", opacity=0.3)
                fig.add_hline(y=20, row=row, col=1, line_dash="dot", line_color="#00E676", opacity=0.3)
            elif name == 'OBV':
                fig.add_trace(go.Scatter(x=plot_data['DateStr'], y=plot_data['OBV'], name='OBV', line={'color': '#FFFFFF'}), row=row, col=1)
        plot_indicator(sub1, 2)
        plot_indicator(sub2, 3)
        # Update layout to treat X-axis as category (No Gaps)
        # Explicitly set all x-axes to category to ensure alignment
        # Using shared_xaxes=True in make_subplots usually handles linking, but specifying type='category' is key.
        # We also need to ensure axes are linked properly if shared_axes is on.
        fig.update_layout(
            height=700, 
            dragmode=False, 
            template='plotly_dark', 
            xaxis_rangeslider_visible=False,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            # Move legend to top-left vertical
            legend={
                "orientation": "v", 
                "yanchor": "top", 
                "y": 0.98, 
                "xanchor": "left", 
                "x": 0.01,
                "bgcolor": "rgba(0,0,0,0.5)"
            },
            # Enable crosshair on hover
            hovermode='x unified'
        )
        # Update all x-axes to be category type to match the string dates and remove gaps
        fig.update_xaxes(type='category', nticks=10, showspikes=True, spikemode='across', spikethickness=1, spikecolor='#888888', spikedash='dot')
        # Lock Y-axis to prevent vertical dragging, auto-scale will handle it
        # Also add spike lines for y-axis
        fig.update_yaxes(fixedrange=True, showspikes=True, spikemode='across', spikethickness=1, spikecolor='#888888', spikedash='dot')
        # Enable scroll zoom
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
        # Strategy Pulse Section (Below Chart)
        st.divider()
        sp_col1, sp_col2 = st.columns([3, 1])
        with sp_col1: st.subheader("Strategy pulse")
        with sp_col2: analyze_clicked = st.button("Analyze", use_container_width=True)
        
        if analyze_clicked:
            status_container = st.container()
            with status_container:
                st.write("正在掃描 K 線型態與均線排列...")
                # 這裡原本邏輯很快，加個微小延遲模擬掃描感
                import time
                time.sleep(0.5)
                st.write("計算布林帶寬與關鍵動能指標...")
                time.sleep(0.5)
                st.write("彙整跨模組 Context 並撰寫分析報告...")
                
                # 取得當前正確的代號 (例如華碩 2357)
                current_tech_ticker = st.session_state.get('tech_active_symbol', active_symbol_display).split()[0]
                resp, history_obj = ta.get_initial_analysis(df, current_tech_ticker)
                st.session_state.tech_messages = [{"role": "assistant", "content": resp}]
                st.session_state.tech_history = history_obj
                
                # 技術面摘要對入 Brain
                tech_ticker = st.session_state.get('tech_active_symbol', active_symbol_display).split()[0]
                st.session_state.ai_brain = engine_ai.update_context(
                    st.session_state.ai_brain, tech_ticker, "technical", resp[:600]
                )
                st.success("分析完成！")
        for msg in st.session_state.tech_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        brain = st.session_state.ai_brain
        tech_ticker = st.session_state.get('tech_active_symbol', active_symbol_display).split()[0]
        if t_input := st.chat_input("詢問技術面或跟其他模組交叉解讀...", key="tech_chat_input"):
            with st.chat_message("user"): st.markdown(t_input)
            with st.chat_message("assistant"):
                stream, update_fn = engine_ai.stream_chat(brain, tech_ticker, t_input)
                ans = st.write_stream(stream)
                st.session_state.ai_brain = update_fn(ans)
                st.session_state.tech_messages.append({"role": "user", "content": t_input})
                st.session_state.tech_messages.append({"role": "assistant", "content": ans})
    else:
        st.info("Please click Scan to load technical data.")
# --- Tab 4: Institutional Flow ---
with tabs[3]:
    if "chips_images" not in st.session_state:
        st.session_state.chips_images = []
        
    if_col1, if_col2 = st.columns([3, 1])
    with if_col1: st.subheader("十個主力九個騙，一個在訓練")
    with if_col2: chips_analyze_clicked = st.button("Analyze", use_container_width=True, key="chips_analyze_btn")
    
    c_symbol = st.text_input("分析標的", value="", placeholder="分析標的 (留空則讓 AI 自行辨識)", label_visibility="collapsed")
    uploaded = st.file_uploader("Upload images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded:
        from PIL import Image
        st.session_state.chips_images = []
        for f in uploaded:
            try:
                st.session_state.chips_images.append(Image.open(f))
            except Exception as e:
                st.error(f"圖片讀取失敗: {e}")
                
    if st.session_state.chips_images:
        img_cols = st.columns(len(st.session_state.chips_images))
        for idx, img in enumerate(st.session_state.chips_images):
            img_cols[idx].image(img, caption=f'籌碼圖 {idx+1}', use_column_width=True)

    if chips_analyze_clicked:
        if not uploaded:
            st.warning("請先上傳籌碼圖表！")
        else:
            with st.status("正在進行籌碼深度掃描...", expanded=True) as status:
                st.write("影像特徵提取與 OCR 辨識中...")
                target = c_symbol if c_symbol else active_symbol_display
                
                st.write("正在辨識主力動向與法人進出...")
                report = engine_chips.analyze_chips_image(uploaded, target)
                st.session_state.chips_report = report
                
                st.write("彙整地緣券商資訊與跨模組 Context...")
                # 籌碼面摘要對入 Brain
                chips_ticker = (c_symbol or active_symbol_display).split()[0]
                st.session_state.ai_brain = engine_ai.update_context(
                    st.session_state.ai_brain, chips_ticker, "chips", report[:600]
                )
                status.update(label="籌碼分析完成！", state="complete", expanded=False)
    st.divider()
    if st.session_state.chips_report:
        st.markdown(st.session_state.chips_report)
        st.divider()
        st.subheader("Chips Chat")
        brain = st.session_state.ai_brain
        chips_ticker = st.session_state.ai_brain.get("active_ticker") or active_symbol_display.split()[0]
        for msg in brain.get("chat_history", []):
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if c_in := st.chat_input("詢問籌碼面或跨模組問題...", key="chips_chat_input"):
            with st.chat_message("user"): st.markdown(c_in)
            with st.chat_message("assistant"):
                with st.spinner("Chips analyst is analyzing..."):
                    ans, updated_brain = engine_ai.chat(brain, chips_ticker, c_in)
                    st.session_state.ai_brain = updated_brain
                    st.markdown(ans)
    else:
        st.info("Upload chips distribution charts to begin analysis.")
# --- Tab 5: Band Strategy ---
with tabs[4]:
    from PIL import Image # Import here to avoid top-level clutter if not used elsewhere, though defined in top is better.
    # PIL is already imported implicitly by engine_chips using Image, but for file_uploader we need it.
    # main_app.py imports: import google.generativeai as genai... 
    # Let's assume PIL Image is needed.
    from PIL import Image
    st.subheader("主投買 沿上軌，老公開心 老婆也開心")
    # Input Section
    input_tickers = ""
    with st.form(key='strategy_form'):
        c1, c2 = st.columns([4, 1])
        with c1:
            input_tickers = st.text_input(
                "股票代號", 
                value="", 
                placeholder="輸入股票代號 (空白分隔)",
                label_visibility="collapsed"
            )
        with c2:
            submitted = st.form_submit_button("Scan", type="primary", use_container_width=True)
    if submitted:
        if not input_tickers.strip():
            st.warning("請輸入股票代號！")
        else:
            slope_val = st.session_state.slope_val
            vol_mul = st.session_state.vol_mul
            tickers = [t.strip() for t in input_tickers.replace(',', ' ').split() if t.strip()]
            results_data = []
            details_map = {} 
            progress = st.progress(0)
            status_text = st.status("正在並行掃描中...", expanded=True)
            # 定義單一任務函數 (Wrapper)
            def scan_task(tk):
                # 呼叫有 Cache 的數據抓取
                df_res, real_t_res, pure_id_res, name_res = ta.get_stock_data_with_name(tk)
                if df_res is not None and len(df_res) >= 20:
                    match, q_data_res, full_df_res = ta.BollingerStrategy.analyze(df_res, slope_val, vol_mul)
                    return {
                        "ticker": tk,
                        "success": True,
                        "match": match, # 雖然此處不篩掉非 match，但保留原始邏輯
                        "q_data": q_data_res,
                        "full_df": full_df_res,
                        "real_t": real_t_res,
                        "pure_id": pure_id_res,
                        "name": name_res
                    }
                return {"ticker": tk, "success": False}
            # 使用 ThreadPoolExecutor 並行處理
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # 提交所有任務
                future_to_ticker = {executor.submit(scan_task, t): t for t in tickers}
                completed_count = 0
                total_tickers = len(tickers)
                for future in concurrent.futures.as_completed(future_to_ticker):
                    tk = future_to_ticker[future]
                    try:
                        res = future.result()
                        if res["success"]:
                            # 收集結果
                            row = {
                                "代號": res["pure_id"],
                                "收盤價": res["q_data"]['Close'],
                                "均線多排": "V" if res["q_data"]['Details']['cond_d'] else "-",
                                "開布林": "V" if res["q_data"]['Details']['cond_a'] else "-",
                                "爆量表態": "V" if res["q_data"]['Details']['cond_c'] else "-",
                                "月線斜率": res["q_data"]['MA20_Slope_Pct'],
                                "上軌斜率": res["q_data"]['Upper_Slope_Pct'],
                                "帶寬增長(%)": res["q_data"]['Bandwidth_Chg'],
                                "量比": res["q_data"]['Vol_Ratio'],
                                "上軌乖離(%)": res["q_data"]['Pos_Upper'],
                                "_ticker": res["real_t"],
                                "_name": res["name"]
                            }
                            results_data.append(row)
                            details_map[res["real_t"]] = {
                                "q_data": res["q_data"],
                                "df": res["full_df"],
                                "pure_id": res["pure_id"],
                                "name": res["name"]
                            }
                            status_text.write(f"{tk} 分析完成")
                        else:
                            status_text.write(f"{tk} 數據不足或無效")
                    except Exception as exc:
                        status_text.write(f"{tk} 發生錯誤: {exc}")
                    completed_count += 1
                    progress.progress(completed_count / total_tickers)
            status_text.update(label="掃描完成!", state="complete", expanded=False)
            status_text.text("分析完成！")
            # 將結果存入 Session State 確保互動後不消失
            st.session_state.scan_results = results_data
            st.session_state.details_map = details_map
    # Display Results
    if st.session_state.scan_results:
        results_data = st.session_state.scan_results
        details_map = st.session_state.details_map
        result_df = pd.DataFrame(results_data)
        st.write("### 策略掃描結果")
        event = st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "開布林": st.column_config.Column(width="small", help="上軌噴、下軌開、帶寬增"),
                "爆量表態": st.column_config.Column(width="small", help="收紅表態 + 爆量 1.5 倍 + 價貼上軌"),
                "均線多排": st.column_config.Column(width="small", help="MA5 > MA10 > MA20 > MA60"),
                "收盤價": st.column_config.NumberColumn(format="%.1f"),
                "月線斜率": st.column_config.NumberColumn(format="%.2f%%"),
                "上軌斜率": st.column_config.NumberColumn(format="%.2f%%"),
                "帶寬增長(%)": st.column_config.NumberColumn(format="%.1f%%"),
                "量比": st.column_config.NumberColumn(format="%.1f x"),
                "上軌乖離(%)": st.column_config.NumberColumn(format="%.1f%%"),
                "_ticker": None, # 隱藏此欄位
                "_name": None    # 隱藏此欄位
            }
        )
        if len(event.selection.rows) > 0:
            selected_row_idx = event.selection.rows[0]
            real_ticker = result_df.iloc[selected_row_idx]["_ticker"]
            pure_id = result_df.iloc[selected_row_idx]["代號"]
            comp_name = result_df.iloc[selected_row_idx]["_name"]
            st.divider()
            st.subheader(f"個股詳細分析: {pure_id} {comp_name}")
            detail_data = details_map[real_ticker]
            q_d = detail_data["q_data"]
            df = detail_data["df"]
            # 繪圖 Function (Inline to avoid global scope clutter for now, or use a helper)
            def plot_strategy_chart(df, ticker):
                df['DateStr'] = df['Date'].dt.strftime('%Y-%m-%d')
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                # 1. Candlestick
                fig.add_trace(go.Candlestick(
                    x=df['DateStr'], 
                    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                    name='收盤價', 
                    increasing_line_color='#FF5252', decreasing_line_color='#00E676',
                    increasing_fillcolor='#FF5252', decreasing_fillcolor='#00E676',
                    hovertemplate='%{close:.2f}<extra></extra>'
                ), row=1, col=1)
                # 2. MAs
                colors = {'MA5': '#F9A825', 'MA10': '#00E676', 'MA20': '#29B6F6', 'MA60': '#AB47BC'}
                for ma, color in colors.items():
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(x=df['DateStr'], y=df[ma], mode='lines', 
                                                line={'color': color, 'width': 1.2}, name=ma, hoverinfo='skip'), row=1, col=1)
                # 3. BBands with Fill
                if 'Upper' in df.columns and 'Lower' in df.columns:
                    fig.add_trace(go.Scatter(x=df['DateStr'], y=df['Upper'], 
                                             line={'color': '#CCCCCC', 'width': 1, 'dash': 'dash'}, 
                                             name='上軌', hoverinfo='skip'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['DateStr'], y=df['Lower'], 
                                             line={'color': '#CCCCCC', 'width': 1, 'dash': 'dash'}, 
                                             fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', 
                                             name='下軌', hoverinfo='skip'), row=1, col=1)
                # 4. Volume
                v_colors = ['#FF5252' if r['Close'] >= r['Open'] else '#00E676' for _, r in df.iterrows()]
                fig.add_trace(go.Bar(x=df['DateStr'], y=df['Volume'], marker_color=v_colors, name='成交量', opacity=0.8), row=2, col=1)
                # Volume MA if exists
                if 'Volume_MA5' in df.columns:
                    fig.add_trace(go.Scatter(x=df['DateStr'], y=df['Volume_MA5'], line={'color': '#E0E0E0', 'width': 1.2}, name='量5MA'), row=2, col=1)
                fig.update_layout(
                    title=f'{ticker} 走勢分析 (多方)', 
                    xaxis_rangeslider_visible=False, 
                    height=650, 
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(
                        orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"
                    ),
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                fig.update_xaxes(type='category', showspikes=True, spikemode='across', spikecolor="#888888", spikethickness=1, spikedash='dot', nticks=10)
                fig.update_yaxes(fixedrange=True, showspikes=True, spikemode='across', spikecolor="#888888", spikethickness=1, spikedash='dot', row=1, col=1)
                fig.update_yaxes(fixedrange=True, row=2, col=1)
                return fig
            fig = plot_strategy_chart(df, pure_id)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
            # AI Analysis Section
            st.divider()
            ba_col1, ba_col2 = st.columns([3, 1])
            with ba_col1: st.subheader("籌碼面分析 (AI 綜合解讀)")
            with ba_col2: band_analyze_clicked = st.button("Analyze", type="primary", key=f"btn_{real_ticker}", use_container_width=True)

            st.info(f"您可以上傳多張包含「主力買賣超」、「三大法人」或「分點」的截圖，AI 將為您進行深度彙整。")
            images = []
            uploaded_files = st.file_uploader(f"上傳 {pure_id} 籌碼圖表", type=["jpg", "jpeg", "png"], key=f"uploader_{real_ticker}", accept_multiple_files=True)
            if uploaded_files:
                for f in uploaded_files:
                    try:
                        images.append(Image.open(f))
                    except Exception as e:
                        st.error(f"無法讀取: {e}")
            
            if images:
                cols_count = len(images)
                cols = st.columns(cols_count)
                for idx, img in enumerate(images):
                    cols[idx].image(img, caption=f'圖表 {idx+1}', use_column_width=True)
            
            if band_analyze_clicked:
                if not images:
                    st.warning("請先上傳籌碼圖表！")
                elif "GEMINI_API_KEY" not in os.environ:
                    st.error("Missing API Key")
                else:
                    with st.status(f"正在對 {real_ticker} 進行波段綜合解讀...", expanded=True) as status:
                        st.write("交叉比對技術指標數據...")
                        import time
                        time.sleep(0.5)
                        st.write("掃描上傳之多張截圖特徵...")
                        report = engine_chips.analyze_chips_image(uploaded_files, pure_id, tech_data=q_d)
                        
                        st.write("撰寫分析報告並更新 Brain Context...")
                        # 對入 Brain
                        st.session_state.ai_brain = engine_ai.update_context(
                            st.session_state.ai_brain, real_ticker, "band_long", report[:600]
                        )
                        st.session_state[f"chat_history_{real_ticker}"] = [{"role": "model", "content": report}]
                        status.update(label="綜合解讀完成！", state="complete", expanded=False)
                if f"chat_history_{real_ticker}" in st.session_state:
                    history = st.session_state[f"chat_history_{real_ticker}"]
                    for msg in history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    brain = st.session_state.ai_brain
                    if user_input := st.chat_input("詢問籌碼或跨模組問題...", key=f"band_long_chat_{real_ticker}"):
                        with st.chat_message("user"): st.markdown(user_input)
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                ans, updated_brain = engine_ai.chat(brain, real_ticker, user_input)
                                st.session_state.ai_brain = updated_brain
                                st.markdown(ans)
    elif st.session_state.scan_results == []:
            st.warning("沒有數據可顯示，請檢查股票代號。")
# --- Tab 6: Band Short Strategy ---
with tabs[5]:
    st.subheader("散戶都喜歡買在山的右半邊")
    # Input Section
    s_input_tickers = ""
    with st.form(key='short_strategy_form'):
        c1, c2 = st.columns([4, 1])
        with c1:
            s_input_tickers = st.text_input(
                "股票代號", 
                value="", 
                placeholder="輸入股票代號 (空白分隔)",
                label_visibility="collapsed"
            )
        with c2:
            s_submitted = st.form_submit_button("Scan", type="primary", use_container_width=True)
    if s_submitted:
        if not s_input_tickers.strip():
            st.warning("請代號！")
        else:
            s_tickers = [t.strip() for t in s_input_tickers.replace(',', ' ').split() if t.strip()]
            s_results_data = []
            s_details_map = {} 
            s_progress = st.progress(0)
            s_status_text = st.status("正在並行掃描空方訊號...", expanded=True)
            def short_scan_task(tk):
                df_res, real_t_res, pure_id_res, name_res = ta.get_stock_data_with_name(tk)
                if df_res is not None and len(df_res) >= 60: # Need MA60
                    match, q_data_res, full_df_res = ta.BollingerStrategy.analyze_short(df_res)
                    return {
                        "ticker": tk,
                        "success": True,
                        "match": match, 
                        "q_data": q_data_res,
                        "full_df": full_df_res,
                        "real_t": real_t_res,
                        "pure_id": pure_id_res,
                        "name": name_res
                    }
                return {"ticker": tk, "success": False}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {executor.submit(short_scan_task, t): t for t in s_tickers}
                completed_count = 0
                total_tickers = len(s_tickers)
                for future in concurrent.futures.as_completed(future_to_ticker):
                    tk = future_to_ticker[future]
                    try:
                        res = future.result()
                        if res["success"]:
                            # 收集結果
                            row = {
                                "代號": res["pure_id"],
                                "收盤價": res["q_data"]['Close'],
                                "月線下彎": "V" if res["q_data"]['MA20_Slope_Down'] else "-",
                                "破十日線": "V" if res["q_data"]['MA5_lt_MA10'] else "-",
                                "破月線": "V" if res["q_data"]['MA5_lt_MA20'] else "-",
                                "季線乖離": res["q_data"]['MA60_Bias'],
                                "半年線乖離": res["q_data"]['MA120_Bias'],
                                "_ticker": res["real_t"],
                                "_name": res["name"]
                            }
                            s_results_data.append(row)
                            s_details_map[res["real_t"]] = {
                                "q_data": res["q_data"],
                                "df": res["full_df"],
                                "pure_id": res["pure_id"],
                                "name": res["name"]
                            }
                            s_status_text.write(f"{tk} 分析完成")
                        else:
                            s_status_text.write(f"{tk} 數據不足或無效")
                    except Exception as exc:
                        s_status_text.write(f"{tk} 發生錯誤: {exc}")
                    completed_count += 1
                    s_progress.progress(completed_count / total_tickers)
            s_status_text.update(label="空方策略掃描完成!", state="complete", expanded=False)
            # Store results in distinct session state keys
            st.session_state.short_scan_results = s_results_data
            st.session_state.short_details_map = s_details_map
    # Display Short Results
    if st.session_state.get('short_scan_results'):
        s_results_data = st.session_state.short_scan_results
        s_details_map = st.session_state.short_details_map
        s_result_df = pd.DataFrame(s_results_data)
        st.write("### 策略掃描結果")
        s_event = st.dataframe(
            s_result_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "代號": st.column_config.TextColumn(),
                "收盤價": st.column_config.NumberColumn(format="%.1f"),
                "月線下彎": st.column_config.Column(width="small", help="MA20 斜率 < 0"),
                "破十日線": st.column_config.Column(width="small", help="MA5 < MA10"),
                "破月線": st.column_config.Column(width="small", help="MA5 < MA20"),
                "季線乖離": st.column_config.NumberColumn(format="%.2f%%"),
                "半年線乖離": st.column_config.NumberColumn(format="%.2f%%"),
                "_ticker": None,
                "_name": None
            }
        )
        if len(s_event.selection.rows) > 0:
            s_selected_idx = s_event.selection.rows[0]
            s_real_ticker = s_result_df.iloc[s_selected_idx]["_ticker"]
            s_pure_id = s_result_df.iloc[s_selected_idx]["代號"]
            s_comp_name = s_result_df.iloc[s_selected_idx]["_name"]
            st.divider()
            st.subheader(f"個股詳細分析 (空方): {s_pure_id} {s_comp_name}")
            s_detail_data = s_details_map[s_real_ticker]
            short_df = s_detail_data["df"]
            short_q_data = s_detail_data["q_data"]
            # Short Chart Logic
            def plot_short_chart(df, ticker):
                df['DateStr'] = df['Date'].dt.strftime('%Y-%m-%d')
                s_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                # 1. Candlestick
                s_fig.add_trace(go.Candlestick(
                    x=df['DateStr'], 
                    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                    name='K線', 
                    increasing_line_color='#FF5252', decreasing_line_color='#00E676',
                    increasing_fillcolor='#FF5252', decreasing_fillcolor='#00E676',
                    hovertemplate='%{close:.2f}<extra></extra>'
                ), row=1, col=1)
                # 2. MAs
                colors = {'MA5': '#F9A825', 'MA10': '#00E676', 'MA20': '#29B6F6', 'MA60': '#AB47BC', 'MA120': '#E91E63'}
                for ma, color in colors.items():
                    if ma in df.columns:
                        s_fig.add_trace(go.Scatter(x=df['DateStr'], y=df[ma], mode='lines', 
                                                  line={'color': color, 'width': 1.2}, name=ma, hoverinfo='skip'), row=1, col=1)
                # 3. BBands with Fill
                if 'Upper' in df.columns and 'Lower' in df.columns:
                    s_fig.add_trace(go.Scatter(x=df['DateStr'], y=df['Upper'], 
                                              line={'color': '#CCCCCC', 'width': 1, 'dash': 'dash'}, 
                                              name='上軌', hoverinfo='skip'), row=1, col=1)
                    s_fig.add_trace(go.Scatter(x=df['DateStr'], y=df['Lower'], 
                                              line={'color': '#CCCCCC', 'width': 1, 'dash': 'dash'}, 
                                              fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', 
                                              name='下軌', hoverinfo='skip'), row=1, col=1)
                # 4. Volume
                v_colors = ['#FF5252' if r['Close'] >= r['Open'] else '#00E676' for _, r in df.iterrows()]
                s_fig.add_trace(go.Bar(x=df['DateStr'], y=df['Volume'], marker_color=v_colors, name='成交量', opacity=0.8), row=2, col=1)
                # Volume MA if exists
                if 'Volume_MA5' in df.columns:
                    s_fig.add_trace(go.Scatter(x=df['DateStr'], y=df['Volume_MA5'], line={'color': '#E0E0E0', 'width': 1.2}, name='量5MA'), row=2, col=1)
                s_fig.update_layout(
                    title=f'{ticker} 空方走勢分析', 
                    xaxis_rangeslider_visible=False, 
                    height=650, 
                    template='plotly_dark',
                    hovermode='x unified',
                    legend=dict(
                        orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"
                    ),
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                s_fig.update_xaxes(type='category', showspikes=True, spikemode='across', spikecolor='#888888', spikethickness=1, spikedash='dot', nticks=10)
                s_fig.update_yaxes(fixedrange=True, showspikes=True, spikemode='across', spikecolor='#888888', spikethickness=1, spikedash='dot', row=1, col=1)
                s_fig.update_yaxes(fixedrange=True, row=2, col=1)
                return s_fig
            s_fig = plot_short_chart(short_df, s_pure_id)
            st.plotly_chart(s_fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})
            # AI Analysis Section for Short
            st.divider()
            bs_col1, bs_col2 = st.columns([3, 1])
            with bs_col1: st.subheader("籌碼面分析 (AI 綜合解讀)")
            with bs_col2: short_analyze_clicked = st.button("Analyze", type="primary", key=f"btn_short_{s_real_ticker}", use_container_width=True)

            st.info(f"請上傳包含 {s_pure_id} 主力籌碼或法人賣超的截圖，AI 將進行空方籌碼診斷。")
            s_images = []
            s_uploaded_files = st.file_uploader(f"上傳 {s_pure_id} 籌碼圖表", type=["jpg", "jpeg", "png"], key=f"uploader_short_{s_real_ticker}", accept_multiple_files=True)
            if s_uploaded_files:
                for f in s_uploaded_files:
                    try:
                        s_images.append(Image.open(f))
                    except Exception as e:
                        st.error(f"圖片讀取失敗: {e}")
            
            if s_images:
                s_cols_count = len(s_images)
                s_cols = st.columns(s_cols_count)
                for idx, img in enumerate(s_images):
                    s_cols[idx].image(img, caption=f'籌碼圖 {idx+1}', use_column_width=True)
            
            if short_analyze_clicked:
                if not s_images:
                    st.warning("請先上傳籌碼圖表！")
                elif "GEMINI_API_KEY" not in os.environ:
                    st.error("Missing API Key")
                else:
                    with st.status(f"正在對 {s_real_ticker} 進行空方籌碼診斷...", expanded=True) as status:
                        st.write("檢查均線乖離與下軌位置...")
                        import time
                        time.sleep(0.5)
                        st.write("掃描主力出貨跡象與散戶接盤數據...")
                        s_report = engine_chips.analyze_chips_image(s_uploaded_files, s_pure_id, tech_data=short_q_data, is_short=True)
                        
                        st.write("評估空方威脅等級並更新記憶...")
                        # 對入 Brain
                        st.session_state.ai_brain = engine_ai.update_context(
                            st.session_state.ai_brain, s_real_ticker, "band_short", s_report[:600]
                        )
                        st.session_state[f"chat_history_short_{s_real_ticker}"] = [{"role": "model", "content": s_report}]
                        status.update(label="空方診斷完成！", state="complete", expanded=False)
                if f"chat_history_short_{s_real_ticker}" in st.session_state:
                    s_history = st.session_state[f"chat_history_short_{s_real_ticker}"]
                    for msg in s_history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    brain = st.session_state.ai_brain
                    if s_user_input := st.chat_input("詢問空方籌碼或跨模組問題...", key=f"input_short_{s_real_ticker}"):
                        with st.chat_message("user"): st.markdown(s_user_input)
                        with st.chat_message("assistant"):
                            with st.spinner("AI Thinking..."):
                                s_ans, updated_brain = engine_ai.chat(brain, s_real_ticker, s_user_input)
                                st.session_state.ai_brain = updated_brain
                                st.markdown(s_ans)
    elif st.session_state.get('short_scan_results') == []:
            st.warning("沒有數據可顯示，請輸入股票代號並開始掃描。")
