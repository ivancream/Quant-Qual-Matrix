import os
import time
import re
import requests
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional, Union

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from . import prompts

# ==========================================
# 1. 設定與工具函式 (Setup & Utils)
# ==========================================
# 使用 os.getenv 讀取環境變數
# 使用 os.environ.get 讀取環境變數
# GEMINI_API_KEY is configured in main_app.py

def get_model() -> genai.GenerativeModel:
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-flash-latest")
    return genai.GenerativeModel(model_id)

def remove_emojis(text: str) -> str:
    """ 移除 Emoji，保持報告專業度 """
    return re.sub(r'[^\w\s,.:;!?()\[\]{}@#$%^&*\-+=/\\\'"<>~`|]', '', text)

def calculate_technicals_str(hist_df: pd.DataFrame) -> str:
    """ 計算技術指標並回傳格式化字串 (供 AI 閱讀) """
    try:
        if len(hist_df) < 60: return "Data insufficient for technicals"
        
        close = hist_df['Close']
        ma5 = close.rolling(window=5).mean().iloc[-1]
        ma20 = close.rolling(window=20).mean().iloc[-1]
        ma60 = close.rolling(window=60).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        # KD (9, 3, 3)
        low_9 = hist_df['Low'].rolling(window=9).min()
        high_9 = hist_df['High'].rolling(window=9).max()
        rsv = ((close - low_9) / (high_9 - low_9)) * 100
        k = 50.0
        d = 50.0
        for val in rsv:
            if not pd.isna(val):
                k = (2/3) * k + (1/3) * val
                d = (2/3) * d + (1/3) * k
        
        # Trend Status
        trend = "Bullish (Strong)" if ma5 > ma20 > ma60 else ("Bearish (Weak)" if ma5 < ma20 < ma60 else "Consolidation")
        
        return (f"Trend: {trend} | RSI(14): {rsi:.2f} | K: {k:.2f} | D: {d:.2f} | "
                f"MA5: {ma5:.2f} | MA20: {ma20:.2f} | MA60: {ma60:.2f}")
    except:
        return "Technical calculation error"

# ==========================================
# 2. 核心數據獲取 (Data Mining)
# ==========================================
def get_stock_info(stock_code: str) -> Optional[Dict[str, Union[str, float]]]:
    """
    獲取硬數據：股價、PE、ROE、技術指標
    回傳: dict (包含所有關鍵數據)
    """
    result: Dict[str, Union[str, float]] = {}
    
    try:
        # 1. 處理代碼
        code = stock_code.strip().upper()
        if code.isdigit():
            # 優先嘗試台灣上市，若無則嘗試上櫃 (這裡簡化邏輯)
            ticker = yf.Ticker(f"{code}.TW")
            hist = ticker.history(period="3mo")
            if hist.empty:
                ticker = yf.Ticker(f"{code}.TWO")
                hist = ticker.history(period="3mo")
        else:
            ticker = yf.Ticker(code)
            hist = ticker.history(period="3mo")
            
        if hist.empty:
            return None

        # 2. 獲取基本面 (Fundamentals)
        info = ticker.info
        result['name'] = info.get('longName', code)
        
        # Try to get price from info, fallback to history
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price:
            current_price = hist.iloc[-1]['Close']
        
        result['price'] = round(current_price, 2)
        
        # PE / ROE / Growth / Margin / Debt / FCF
        pe = info.get('forwardPE') or info.get('trailingPE')
        result['pe'] = f"{pe:.2f}" if pe else "N/A"
        
        roe = info.get('returnOnEquity')
        result['roe'] = f"{roe*100:.2f}%" if roe else "N/A"
        
        rev_growth = info.get('revenueGrowth')
        result['growth'] = f"{rev_growth*100:.2f}%" if rev_growth else "N/A"

        margin = info.get('grossMargins')
        result['margin'] = f"{margin*100:.2f}%" if margin else "N/A"

        debt = info.get('debtToEquity')
        result['debt_ratio'] = f"{debt:.2f}%" if debt else "N/A"

        fcf = info.get('freeCashflow')
        if fcf:
            result['fcf'] = f"{fcf/1e8:.2f} 億" # 轉成億為單位
        else:
            result['fcf'] = "N/A"
        
        # 3. 計算技術指標 (Technicals)
        result['technicals'] = calculate_technicals_str(hist)
        
        return result
        
    except Exception as e:
        print(f"Error in get_stock_info: {e}")
        return None

# ==========================================
# 3. 消息面爬蟲 (Sentiment Mining)
# ==========================================

def get_anue_news_selenium(keyword: str) -> str:
    """ Anue 鉅亨網爬蟲 """
    news_data = []
    chrome_options = Options()
    # 使用新版 headless 模式，更加穩定
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    # 忽略憑證錯誤與自動化控制提示
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        url = f"https://www.cnyes.com/search/news?keyword={keyword}"
        driver.get(url)
        time.sleep(1.5)
        titles = driver.find_elements(By.TAG_NAME, "h3")
        for title in titles:
            text = title.text.strip()
            if text and len(text) > 5:
                news_data.append(f"- [Anue] {text}")
                if len(news_data) >= 3: break
    except Exception: pass
    finally:
        if driver: driver.quit()
    return "\n".join(news_data) if news_data else "No specific news on Anue."

def get_yahoo_news_selenium(stock_code_full: str) -> str:
    """ Yahoo 股市爬蟲 """
    news_data = []
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = None
    try:
        # 如果是數字代碼，需要加上 .TW 或 .TWO，這裡假設使用者輸入完整代碼或由外部處理
        # 為了保險起見，若只傳入數字，補上 .TW
        target = stock_code_full
        if target.isdigit():
             target = f"{target}.TW"
             
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        url = f"https://tw.stock.yahoo.com/quote/{target}/news"
        driver.get(url)
        time.sleep(1.5)
        titles = driver.find_elements(By.TAG_NAME, "h3")
        for title in titles:
            text = title.text.strip()
            if text and len(text) > 8 and "廣告" not in text:
                news_data.append(f"- [Yahoo] {text}")
                if len(news_data) >= 3: break
    except Exception: pass
    finally:
        if driver: driver.quit()
    return "\n".join(news_data) if news_data else "No specific news on Yahoo."

def get_ptt_sentiment(clean_code: str) -> str:
    """ PTT Stock 版爬蟲 """
    headers = {'User-Agent': 'Mozilla/5.0', 'Cookie': 'over18=1'}
    url = f"https://www.ptt.cc/bbs/Stock/search?q={clean_code}"
    posts = []
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        divs = soup.find_all('div', class_='r-ent')
        for i, div in enumerate(divs):
            try:
                title = div.find('div', class_='title').text.strip()
                date_str = div.find('div', class_='date').text.strip()
                # 簡單判斷是否為近期 (這裡簡化邏輯)
                posts.append(f"- [{date_str}] {title}")
                if i >= 4: # 取得 5 篇後停止
                    break
            except: continue
    except Exception: pass
    return "\n".join(posts) if posts else "Low retail discussion on PTT."

def get_sentiment_summary(stock_code: str) -> Dict[str, str]:
    """
    整合新聞與 PTT 爬蟲
    回傳: dict (包含 'anue', 'yahoo', 'ptt' 三種來源的摘要字串)
    """
    summary = {'anue': '', 'yahoo': '', 'ptt': ''}
    
    # 處理代碼與名稱 (簡單處理，實際可能需要對照表)
    clean_code = stock_code.replace(".TW", "").replace(".TWO", "")
    
    if clean_code.isdigit():
        # 如果是數字代碼，可能需要去查中文名稱來搜新聞 (這裡簡化直接用代碼搜)
        search_keyword = clean_code 
    else:
        search_keyword = clean_code
        
    summary['anue'] = get_anue_news_selenium(search_keyword)
    summary['yahoo'] = get_yahoo_news_selenium(stock_code)
    summary['ptt'] = get_ptt_sentiment(clean_code)
    
    return summary

# ==========================================
# 4. AI 報告生成 (AI Synthesis)
# ==========================================
def generate_ai_report(stock_code: str, data_info: Dict, sentiment_summary: Dict) -> str:
    """
    整合數據與情緒，生成 Markdown 報告
    """
    model = get_model()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Prepare Data Strings
    financials_str = f"""
    * Price: {data_info.get('price')}
    * Fundamentals: PE: {data_info.get('pe')} | ROE: {data_info.get('roe')} | Rev Growth: {data_info.get('growth')}
    * Technicals: {data_info.get('technicals')}
    """

    sentiment_str = f"""
    - News (Anue): {sentiment_summary.get('anue')}
    - News (Yahoo): {sentiment_summary.get('yahoo')}
    - Retail (PTT): {sentiment_summary.get('ptt')}
    """

    prompt = prompts.get_fundamental_report_prompt(stock_code, financials_str, sentiment_str)

    try:
        response = model.generate_content(prompt)
        return remove_emojis(response.text)
    except Exception as e:
        return f"AI Report Generation Error: {e}"

def chat_with_analyst(user_msg: str, context_report: str, history: list = []) -> str:
    """
    讓使用者針對報告進行提問 (支援對話記憶)
    """
    model = get_model()
    
    # 建構對話歷史字串
    history_text = ""
    for msg in history:
        role = "User" if msg['role'] == "user" else "Analyst"
        history_text += f"{role}: {msg['content']}\n\n"
    
    prompt = prompts.get_fundamental_chat_prompt(context_report, history_text, user_msg)
    
    
    try:
        response = model.generate_content(prompt)
        return remove_emojis(response.text)
    except Exception as e:
        return f"Chat Error: {e}"
