# engine_technical.py
import os
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from typing import Tuple, List, Dict, Optional, Any
from . import prompts
import pandas_ta as ta

# 常用權值股中文名稱對照表 (當 yfinance 抓不到中文時的備案)
TW_NAMES = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", 
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海",
    "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "2412.TW": "中華電", "2308.TW": "台達電", "6669.TW": "緯穎",
    "3037.TW": "欣興", "2337.TW": "旺宏", "2301.TW": "光寶科",
    "2357.TW": "華碩", "2382.TW": "廣達", "3231.TW": "緯創",
    "2376.TW": "技嘉", "2377.TW": "微星", "2610.TW": "華航",
    "2618.TW": "長榮航", "2834.TW": "臺企銀", "2884.TW": "玉山金",
    "2886.TW": "兆豐金"
}

# ==========================================
# 設定 Gemini API Key
# ==========================================
# 使用 os.getenv 讀取環境變數
# 使用 os.environ.get 讀取環境變數
# GEMINI_API_KEY is configured in main_app.py

def fetch_data(stock_id: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    抓取股價並進行資料清洗
    """
    # 處理台股代碼後綴
    stock_id = str(stock_id).strip()
    if stock_id.isdigit():
        symbol = f"{stock_id}.TW"
    elif not stock_id.endswith(".TW") and not stock_id.endswith(".TWO"):
        symbol = f"{stock_id}.TW"
    else:
        symbol = stock_id
        
    print(f"正在抓取 {symbol}...")
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"下載失敗: {e}")
        return None
    
    if df.empty:
        return None
        
    # 處理 MultiIndex (yfinance 新版問題)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # 嘗試直接降維，若只有單一 Ticker，level 1 通常是 Ticker 名稱
            if symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(0)
        except:
             df.columns = df.columns.get_level_values(0)
    
    # 強制將索引轉為 Datetime 並移除時區資訊 (避免後續繪圖問題)
    if isinstance(df.index, pd.DatetimeIndex):
         df.index = df.index.tz_localize(None)
    else:
         df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # ✅ 關鍵修正：檢查必要欄位是否存在
    # Standardize columns to simplify check
    df.columns = [c.capitalize() for c in df.columns]
    
    required_cols = ['Close', 'Open', 'High', 'Low'] 
    # Volume sometimes is missing in indices, we can handle it
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"缺少必要欄位: {missing_cols}")
        return None

    # 移除任何含有 NaN 的列 (確保沒有空數據日)
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        return None
        
    return df

def get_symbol_name(stock_id: str) -> str:
    """ 嘗試獲取股票名稱 """
    try:
        stock_id = str(stock_id).strip()
        if stock_id.isdigit():
            sym = f"{stock_id}.TW"
        else:
            sym = stock_id
        ticker = yf.Ticker(sym)
        # 嘗試從不同欄位獲取名稱
        name = ticker.info.get('longName') or ticker.info.get('shortName') or ""
        return name
    except:
        return ""

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算技術指標：MA(5,10,20,60), BB(20,2), MACD, RSI, OBV
    """
    data = df.copy()
    
    # 1. 均線 (MA)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean() # 月線
    data['MA60'] = data['Close'].rolling(window=60).mean() # 季線
    
    # 2. 乖離率 (Bias)
    data['Bias_20'] = ((data['Close'] - data['MA20']) / data['MA20']) * 100
    
    # 3. 布林通道 (Bollinger Bands) - 20MA 為中軌
    std20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['MA20'] + (std20 * 2)
    data['BB_Lower'] = data['MA20'] - (std20 * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['MA20']
    
    # 4. MACD (12, 26, 9)
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Hist'] = data['MACD'] - data['Signal']
    
    # 5. RSI (14日)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. OBV (能量潮) - Check if Volume exists
    if 'Volume' in data.columns:
        data['Price_Change'] = data['Close'].diff()
        data['Direction'] = np.where(data['Price_Change'] > 0, 1, 
                                     np.where(data['Price_Change'] < 0, -1, 0))
        data['Direction'] = data['Direction'].fillna(0) # type: ignore
        data['OBV'] = (data['Volume'] * data['Direction']).cumsum()
        data.drop(['Price_Change', 'Direction'], axis=1, inplace=True)
    else:
        # If no volume, fill OBV with 0 or NaN
        data['OBV'] = 0.0
        # Also ensure Volume column exists for plotting
        data['Volume'] = 0.0

    # 7. KD 指標 (9, 3, 3)
    low_9 = data['Low'].rolling(window=9).min()
    high_9 = data['High'].rolling(window=9).max()
    rsv = ((data['Close'] - low_9) / (high_9 - low_9)) * 100
    
    # 初始化 K, D 為 50
    k = 50.0
    d = 50.0
    k_list = []
    d_list = []
    
    for val in rsv:
        if pd.isna(val):
            k_list.append(np.nan)
            d_list.append(np.nan)
        else:
            k = (2/3) * k + (1/3) * val
            d = (2/3) * d + (1/3) * k
            k_list.append(k)
            d_list.append(d)
            
    data['K'] = k_list
    data['D'] = d_list
    
    return data

def get_latest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    提取最後一筆資料供 AI 使用
    """
    last = df.iloc[-1]
    
    summary = {
        "Date": str(last.name.date()),
        "Close": round(last['Close'], 2),
        "MA5": round(last['MA5'], 2),
        "MA20": round(last['MA20'], 2),
        "RSI": round(last['RSI'], 2),
        "K": round(last['K'], 2) if 'K' in last else "N/A",
        "D": round(last['D'], 2) if 'D' in last else "N/A",
        "MACD_Hist": round(last['Hist'], 2),
        "BB_Status": "Upper Band" if last['Close'] > last['BB_Upper'] else ("Lower Band" if last['Close'] < last['BB_Lower'] else "Normal")
    }
    return summary

def get_initial_analysis(df: pd.DataFrame, symbol: str) -> Tuple[str, List]:
    """
    產生第一輪的技術分析 (系統主動發起)
    回傳: (response_text, chat_history)
    """
    # 準備數據
    cols = ['Close', 'Volume', 'MA5', 'MA20', 'MACD', 'Hist', 'RSI', 'BB_Width']
    available_cols = [c for c in cols if c in df.columns]
    data_for_ai = df.tail(5)[available_cols].to_markdown()
    
    # System Prompt (隱藏設定，定義 AI 人設)
    system_instruction = prompts.get_technical_analysis_prompt(symbol, data_for_ai)
    
    try:
        # 使用 generate_content 模式 (stateless) 以避免 session 卡住
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash-latest")
        model = genai.GenerativeModel(model_id)
        
        prompt = f"{system_instruction}\n\n請開始你的技術分析："
        response = model.generate_content(prompt)
        
        # 手動建立初始歷史紀錄 List[Dict]
        # 包含系統指令作為背景知識 (雖然 chat history 通常只有 user/model，
        # 但為了延續 context，我們將 system_instruction 隱含在第一則回應的 context 中，
        # 或是之後 continue_chat 時將其加入 prompt)
        
        # 這裡回傳的 history 結構改成簡單的 list dict
        initial_history = [
            {"role": "user", "content": f"分析 {symbol} 的技術面數據:\n{data_for_ai}"},
            {"role": "model", "content": response.text}
        ]
        
        return response.text, initial_history
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
             return "Error: Model not found. Please try 'gemini-1.5-flash'.", []
        if "429" in error_msg:
             return "Error: Quota exceeded. Please upgrade plan or swap API key.", []
        return f"AI Error: {error_msg}", []

def continue_chat(user_input: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    延續對話 (Stateless 模式)
    history: List of dicts [{'role': 'user'|'model', 'content': text}, ...]
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # 建構包含歷史紀錄的 Prompt
        conversation_context = ""
        for msg in history:
            role_name = "User" if msg['role'] == "user" else "AI Analyst"
            conversation_context += f"{role_name}: {msg['content']}\n\n"
            
        final_prompt = prompts.get_technical_chat_prompt(conversation_context, user_input)
        
        response = model.generate_content(final_prompt)
        
        # 更新歷史紀錄
        new_history = history + [
            {"role": "user", "content": user_input},
            {"role": "model", "content": response.text}
        ]
        
        return response.text, new_history
    except Exception as e:
        return f"Error: {e}", history

class BollingerStrategy:
    @staticmethod
    def calculate_indicators(df):
        if df is None:
            return None
        if df.empty or len(df) < 20:
            return df
            
        df = df.copy()
        try:
            # 基本均線
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA10'] = ta.sma(df['Close'], length=10)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA60'] = ta.sma(df['Close'], length=60)
            df['MA120'] = ta.sma(df['Close'], length=120)
            
            bbands = ta.bbands(df['Close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                upper_col = [c for c in bbands.columns if 'BBU' in c][0]
                mid_col = [c for c in bbands.columns if 'BBM' in c][0]
                lower_col = [c for c in bbands.columns if 'BBL' in c][0]
                
                df['Upper'] = bbands[upper_col]
                df['Lower'] = bbands[lower_col]
                
                df['Bandwidth_Pct'] = (df['Upper'] - df['Lower']) / df['MA20'] * 100
                df['Volume_MA5'] = ta.sma(df['Volume'], length=5)
            else:
                for col in ['Upper', 'Lower', 'Bandwidth_Pct', 'Volume_MA5']:
                    df[col] = np.nan
        except Exception:
            for col in ['Upper', 'Lower', 'Bandwidth_Pct', 'Volume_MA5', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120']:
                df[col] = np.nan
        return df

    @classmethod
    def analyze(cls, df, upper_slope_threshold=0.003, vol_surge_multiplier=1.5):
        df = cls.calculate_indicators(df)
        
        required_cols = ['Upper', 'Lower', 'MA20', 'Bandwidth_Pct', 'Volume_MA5']
        if df is None or not all(col in df.columns for col in required_cols) or len(df) < 2:
            return False, {}, df
            
        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        
        if pd.isna(today['Upper']) or pd.isna(yesterday['Upper']):
            return False, {}, df
        
        today_bw = today['Bandwidth_Pct']
        yesterday_bw = yesterday['Bandwidth_Pct']
        bw_change = (today_bw - yesterday_bw) / yesterday_bw * 100 if yesterday_bw != 0 else 0
        
        cond_a = (today['Upper'] > yesterday['Upper']) and \
                 (today['Lower'] < yesterday['Lower']) and \
                 (today_bw > yesterday_bw)
        
        upper_slope_raw = (today['Upper'] - yesterday['Upper']) / yesterday['Upper']
        upper_slope_pct = upper_slope_raw * 100
        cond_b = upper_slope_raw > upper_slope_threshold
        
        # 月線斜率
        ma20_slope_raw = (today['MA20'] - yesterday['MA20']) / yesterday['MA20'] if yesterday['MA20'] != 0 else 0
        ma20_slope_pct = ma20_slope_raw * 100

        is_red = today['Close'] > today['Open']
        vol_ratio = today['Volume'] / today['Volume_MA5'] if today['Volume_MA5'] > 0 else 0
        is_vol_surge = vol_ratio > vol_surge_multiplier
        
        pos_upper = (today['Close'] / today['Upper']) * 100
        is_touching = pos_upper >= 99.0
        
        cond_c = is_red and is_touching and is_vol_surge
        
        # 4. 均線多排檢查 (MA5 > MA10 > MA20 > MA60)
        # 需確保均線值都存在 (非 NaN)
        has_mas = all(not pd.isna(today[col]) for col in ['MA5', 'MA10', 'MA20', 'MA60'])
        if has_mas:
            cond_d = (today['MA5'] > today['MA10']) and \
                     (today['MA10'] > today['MA20']) and \
                     (today['MA20'] > today['MA60'])
        else:
            cond_d = False
        
        is_match = cond_a and cond_b and cond_c and cond_d
        
        quant_data = {
            "Match": is_match,
            "Close": today['Close'],
            "Bandwidth_Pct": round(today_bw, 1),
            "Bandwidth_Chg": round(bw_change, 1),
            "Upper_Slope_Pct": round(upper_slope_pct, 2),
            "MA20_Slope_Pct": round(ma20_slope_pct, 2),
            "Vol_Ratio": round(vol_ratio, 1),
            "Pos_Upper": round(pos_upper, 1),
            "Is_Red": is_red,
            "Details": {
               "cond_a": cond_a, "cond_b": cond_b, "cond_c": cond_c, "cond_d": cond_d
            }
        }
        return is_match, quant_data, df

    @classmethod
    def analyze_short(cls, df):
        """
        波段空方策略 (支撐乖離版):
        1. 月線下彎 (MA20 Slope < 0)
        2. 破十日線 (ma5 < ma10)
        3. 破月線 (ma5 < ma20)
        4. 季線/半年線乖離率
        """
        df = cls.calculate_indicators(df)
        
        required_cols = ['MA5', 'MA10', 'MA20', 'MA60', 'MA120']
        if df is None or not all(col in df.columns for col in required_cols) or len(df) < 2:
            return False, {}, df
            
        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        
        # 1. 月線下彎
        ma20_slope = (today['MA20'] - yesterday['MA20']) / yesterday['MA20'] if yesterday['MA20'] != 0 else 0
        cond_ma20_down = ma20_slope < 0
        
        # 2. 破十日線 (ma5 < ma10)
        cond_ma5_lt_ma10 = today['MA5'] < today['MA10']
        
        # 3. 破月線 (ma5 < ma20)
        cond_ma5_lt_ma20 = today['MA5'] < today['MA20']
        
        # 4. 季線乖離率
        ma60_val = today['MA60']
        ma60_bias = (today['Close'] - ma60_val) / ma60_val * 100 if ma60_val > 0 else 0
        
        # 5. 半年線乖離率
        ma120_val = today['MA120']
        ma120_bias = (today['Close'] - ma120_val) / ma120_val * 100 if ma120_val > 0 else 0
        
        # 趨勢條件: 基本門檻仍保留收盤在月線下
        is_match = today['Close'] < today['MA20']
        
        quant_data = {
            "Match": is_match,
            "Close": today['Close'],
            "MA60_Bias": round(ma60_bias, 2),
            "MA120_Bias": round(ma120_bias, 2),
            "MA20_Slope_Down": cond_ma20_down,
            "MA5_lt_MA10": cond_ma5_lt_ma10,
            "MA5_lt_MA20": cond_ma5_lt_ma20,
            "Details": {
               "cond_a": cond_ma20_down,
               "cond_b": cond_ma5_lt_ma10,
               "cond_c": cond_ma5_lt_ma20,
               "cond_d": today['MA60'] < today['MA120']
            }
        }
        return is_match, quant_data, df

@st.cache_data(ttl=3600)
def get_stock_data_with_name(ticker):
    """
    獲取股票數據並回傳 (df, real_ticker, display_ticker)
    """
    try:
        t = str(ticker).strip()
        if t.isdigit() and len(t) == 4:
            t = f"{t}.TW"
        
        stock = yf.Ticker(t)
        df = stock.history(period="250d") 
        if df.empty and ".TW" in t:
            t = t.replace(".TW", ".TWO")
            stock = yf.Ticker(t)
            df = stock.history(period="250d")
        
        if not df.empty:
            # 優先從 Ticker.info 抓取名稱，不再檢查中文對照表，直接使用 yf 提供之原始名稱 (通常為英文)
            name = ""
            try:
                info = stock.info
                name = info.get('longName') or info.get('shortName', '')
            except:
                name = ""
            
            # 清理代號顯示 (移除 .TW .TWO)
            pure_id = t.replace(".TW", "").replace(".TWO", "")
            
            return df.reset_index(), t, pure_id, name
        return None, t, ticker, ""
    except Exception:
        return None, ticker, ticker, ""

