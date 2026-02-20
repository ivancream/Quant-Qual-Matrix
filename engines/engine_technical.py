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

# Â∏∏Áî®Ê¨äÂÄºËÇ°‰∏≠Ê??çÁ®±Â∞çÁÖßË°?(??yfinance ?ì‰??∞‰∏≠?áÊ??ÑÂ?Ê°?
TW_NAMES = {
    "2330.TW": "?∞Á???, "2317.TW": "È¥ªÊµ∑", "2454.TW": "?ØÁôºÁß?, 
    "2603.TW": "?∑Ê¶Æ", "2609.TW": "?ΩÊ?", "2615.TW": "?¨Êµ∑",
    "2303.TW": "?ØÈõª", "2881.TW": "ÂØåÈÇ¶??, "2882.TW": "?ãÊ≥∞??,
    "2412.TW": "‰∏≠ËèØ??, "2308.TW": "?∞È???, "6669.TW": "Á∑ØÁ?",
    "3037.TW": "Ê¨??", "2337.TW": "?∫Â?", "2301.TW": "?âÂØ∂Áß?,
    "2357.TW": "?ØÁ¢©", "2382.TW": "Âª??", "3231.TW": "Á∑ØÂâµ",
    "2376.TW": "?Ä??, "2377.TW": "ÂæÆÊ?", "2610.TW": "?ØËà™",
    "2618.TW": "?∑Ê¶Æ??, "2834.TW": "?∫‰??Ä", "2884.TW": "?âÂ±±??,
    "2886.TW": "?ÜË???
}

# ==========================================
# Ë®≠Â? Gemini API Key
# ==========================================
# ‰ΩøÁî® os.getenv ËÆÄ?ñÁí∞Â¢ÉË???
# ‰ΩøÁî® os.environ.get ËÆÄ?ñÁí∞Â¢ÉË???
# GEMINI_API_KEY is configured in main_app.py

def fetch_data(stock_id: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    ?ìÂ??°ÂÉπ‰∏¶ÈÄ≤Ë?Ë≥áÊ?Ê∏ÖÊ?
    """
    # ?ïÁ??∞ËÇ°‰ª?¢ºÂæåÁ∂¥
    stock_id = str(stock_id).strip()
    symbols_to_try = []
    
    if stock_id.isdigit():
        symbols_to_try = [f"{stock_id}.TW", f"{stock_id}.TWO"]
    elif not stock_id.endswith(".TW") and not stock_id.endswith(".TWO"):
        symbols_to_try = [f"{stock_id}.TW", f"{stock_id}.TWO"]
    else:
        symbols_to_try = [stock_id]
        
    df = None
    final_symbol = symbols_to_try[0]
    
    for sym in symbols_to_try:
        try:
            print(f"Ê≠?ú®?óË©¶?ìÂ? {sym}...")
            df = yf.download(sym, period=period, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                final_symbol = sym
                break
        except Exception as e:
            print(f"?óË©¶ {sym} Â§±Ê?: {e}")
            continue
    
    if df is None or df.empty:
        return None
        
    # ?ïÁ? MultiIndex (yfinance ?∞Á??èÈ?)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # ?óË©¶?¥Êé•?çÁ∂≠ÔºåËã•?™Ê??Æ‰? TickerÔºålevel 1 ?öÂ∏∏??Ticker ?çÁ®±
            if symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(0)
        except:
             df.columns = df.columns.get_level_values(0)
    
    # Âº∑Âà∂Â∞áÁ¥¢ÂºïË???Datetime ‰∏¶Áßª?§Ê??ÄË≥áË? (?øÂ?ÂæåÁ?Áπ™Â??èÈ?)
    if isinstance(df.index, pd.DatetimeIndex):
         df.index = df.index.tz_localize(None)
    else:
         df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # ???úÈçµ‰øÆÊ≠£ÔºöÊ™¢?•Â?Ë¶ÅÊ?‰ΩçÊòØ?¶Â???
    # Standardize columns to simplify check
    df.columns = [c.capitalize() for c in df.columns]
    
    required_cols = ['Close', 'Open', 'High', 'Low'] 
    # Volume sometimes is missing in indices, we can handle it
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Áº∫Â?ÂøÖË?Ê¨Ñ‰?: {missing_cols}")
        return None

    # ÁßªÈô§‰ªª‰??´Ê? NaN ?ÑÂ? (Á¢∫‰?Ê≤íÊ?Á©∫Êï∏?öÊó•)
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        return None
        
    return df

def get_symbol_name(stock_id: str) -> str:
    """ ?óË©¶?≤Â??°Á•®?çÁ®± """
    try:
        stock_id = str(stock_id).strip()
        symbols = [f"{stock_id}.TW", f"{stock_id}.TWO"] if stock_id.isdigit() else [stock_id]
        
        for sym in symbols:
            ticker = yf.Ticker(sym)
            try:
                # ?óË©¶ÂæûÁâπÂÆöÊ?‰ΩçÁç≤?ñÂ?Á®?
                info = ticker.info
                name = info.get('longName') or info.get('shortName')
                if name:
                    return name
            except:
                continue
        return ""
    except:
        return ""

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ë®àÁ??ÄË°ìÊ?Ê®ôÔ?MA(5,10,20,60), BB(20,2), MACD, RSI, OBV
    """
    data = df.copy()
    
    # 1. ?áÁ? (MA)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean() # ?àÁ?
    data['MA60'] = data['Close'].rolling(window=60).mean() # Â≠??
    
    # 2. ‰πñÈõ¢??(Bias)
    data['Bias_20'] = ((data['Close'] - data['MA20']) / data['MA20']) * 100
    
    # 3. Â∏ÉÊ??öÈ? (Bollinger Bands) - 20MA ?∫‰∏≠Ëª?
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
    
    # 5. RSI (14??
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. OBV (?ΩÈ?ÊΩ? - Check if Volume exists
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

    # 7. KD ?áÊ? (9, 3, 3)
    low_9 = data['Low'].rolling(window=9).min()
    high_9 = data['High'].rolling(window=9).max()
    rsv = ((data['Close'] - low_9) / (high_9 - low_9)) * 100
    
    # ?ùÂ???K, D ??50
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
    ?êÂ??ÄÂæå‰?Á≠ÜË??ô‰? AI ‰ΩøÁî®
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
    ?¢Á?Á¨¨‰?Ëº™Á??ÄË°ìÂ???(Á≥ªÁµ±‰∏ªÂ??ºËµ∑)
    ?ûÂÇ≥: (response_text, chat_history)
    """
    # Ê∫ñÂ??∏Ê?
    cols = ['Close', 'Volume', 'MA5', 'MA20', 'MACD', 'Hist', 'RSI', 'BB_Width']
    available_cols = [c for c in cols if c in df.columns]
    data_for_ai = df.tail(5)[available_cols].to_markdown()
    
    # System Prompt (?±Ë?Ë®≠Â?ÔºåÂ?Áæ?AI ‰∫∫Ë®≠)
    system_instruction = prompts.get_technical_analysis_prompt(symbol, data_for_ai)
    
    try:
        # ‰ΩøÁî® generate_content Ê®°Â? (stateless) ‰ª•ÈÅø??session ?°‰?
        model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_id)
        
        prompt = f"{system_instruction}\n\nË´ãÈ?Âßã‰??ÑÊ?Ë°ìÂ??êÔ?"
        response = model.generate_content(prompt)
        
        # ?ãÂ?Âª∫Á??ùÂ?Ê≠∑Âè≤Á¥Ä??List[Dict]
        # ?ÖÂê´Á≥ªÁµ±?á‰ª§‰ΩúÁÇ∫?åÊôØ?•Ë? (?ñÁÑ∂ chat history ?öÂ∏∏?™Ê? user/modelÔº?
        # ‰ΩÜÁÇ∫‰∫ÜÂª∂Á∫?contextÔºåÊ??ëÂ? system_instruction ?±Âê´?®Á¨¨‰∏Ä?áÂ??âÁ? context ‰∏≠Ô?
        # ?ñÊòØ‰πãÂ? continue_chat ?ÇÂ??∂Â???prompt)
        
        # ?ôË£°?ûÂÇ≥??history ÁµêÊ??πÊ?Á∞°ÂñÆ??list dict
        initial_history = [
            {"role": "user", "content": f"?ÜÊ? {symbol} ?ÑÊ?Ë°ìÈù¢?∏Ê?:\n{data_for_ai}"},
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
    Âª∂Á?Â∞çË©± (Stateless Ê®°Â?)
    history: List of dicts [{'role': 'user'|'model', 'content': text}, ...]
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Âª∫Ê??ÖÂê´Ê≠∑Âè≤Á¥Ä?ÑÁ? Prompt
        conversation_context = ""
        for msg in history:
            role_name = "User" if msg['role'] == "user" else "AI Analyst"
            conversation_context += f"{role_name}: {msg['content']}\n\n"
            
        final_prompt = prompts.get_technical_chat_prompt(conversation_context, user_input)
        
        response = model.generate_content(final_prompt)
        
        # ?¥Êñ∞Ê≠∑Âè≤Á¥Ä??
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
            # ?∫Êú¨?áÁ?
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
        
        # ?àÁ??úÁ?
        ma20_slope_raw = (today['MA20'] - yesterday['MA20']) / yesterday['MA20'] if yesterday['MA20'] != 0 else 0
        ma20_slope_pct = ma20_slope_raw * 100

        is_red = today['Close'] > today['Open']
        vol_ratio = today['Volume'] / today['Volume_MA5'] if today['Volume_MA5'] > 0 else 0
        is_vol_surge = vol_ratio > vol_surge_multiplier
        
        pos_upper = (today['Close'] / today['Upper']) * 100
        is_touching = pos_upper >= 99.0
        
        cond_c = is_red and is_touching and is_vol_surge
        
        # 4. ?áÁ?Â§öÊ?Ê™¢Êü• (MA5 > MA10 > MA20 > MA60)
        # ?ÄÁ¢∫‰??áÁ??ºÈÉΩÂ≠òÂú® (??NaN)
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
        Ê≥¢ÊÆµÁ©∫ÊñπÁ≠ñÁï• (?ØÊ?‰πñÈõ¢??:
        1. ?àÁ?‰∏ãÂ? (MA20 Slope < 0)
        2. ?¥Â??•Á? (ma5 < ma10)
        3. ?¥Ê?Á∑?(ma5 < ma20)
        4. Â≠??/?äÂπ¥Á∑ö‰??¢Á?
        """
        df = cls.calculate_indicators(df)
        
        required_cols = ['MA5', 'MA10', 'MA20', 'MA60', 'MA120']
        if df is None or not all(col in df.columns for col in required_cols) or len(df) < 2:
            return False, {}, df
            
        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        
        # 1. ?àÁ?‰∏ãÂ?
        ma20_slope = (today['MA20'] - yesterday['MA20']) / yesterday['MA20'] if yesterday['MA20'] != 0 else 0
        cond_ma20_down = ma20_slope < 0
        
        # 2. ?¥Â??•Á? (ma5 < ma10)
        cond_ma5_lt_ma10 = today['MA5'] < today['MA10']
        
        # 3. ?¥Ê?Á∑?(ma5 < ma20)
        cond_ma5_lt_ma20 = today['MA5'] < today['MA20']
        
        # 4. Â≠??‰πñÈõ¢??
        ma60_val = today['MA60']
        ma60_bias = (today['Close'] - ma60_val) / ma60_val * 100 if ma60_val > 0 else 0
        
        # 5. ?äÂπ¥Á∑ö‰??¢Á?
        ma120_val = today['MA120']
        ma120_bias = (today['Close'] - ma120_val) / ma120_val * 100 if ma120_val > 0 else 0
        
        # Ë∂®Âã¢Ê¢ù‰ª∂: ?∫Êú¨?ÄÊ™ª‰?‰øùÁ??∂Áõ§?®Ê?Á∑ö‰?
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
    ?≤Â??°Á•®?∏Ê?‰∏¶Â???(df, real_ticker, display_ticker)
    """
    try:
        t = str(ticker).strip()
        symbols = [f"{t}.TW", f"{t}.TWO"] if t.isdigit() and len(t) == 4 else [t]
        
        df = pd.DataFrame()
        final_t = symbols[0]
        
        for sym in symbols:
            try:
                stock = yf.Ticker(sym)
                df = stock.history(period="250d")
                if not df.empty:
                    final_t = sym
                    break
            except:
                continue
        
        if not df.empty:
            # ?™Â?Âæ?Ticker.info ?ìÂ??çÁ®±
            name = ""
            try:
                stock_obj = yf.Ticker(final_t)
                info = stock_obj.info
                name = info.get('longName') or info.get('shortName', '')
            except:
                name = ""
            
            # Ê∏ÖÁ?‰ª??È°ØÁ§∫ (ÁßªÈô§ .TW .TWO)
            pure_id = final_t.replace(".TW", "").replace(".TWO", "")
            
            return df.reset_index(), final_t, pure_id, name
        return None, final_t, ticker, ""
    except Exception:
        return None, ticker, ticker, ""


