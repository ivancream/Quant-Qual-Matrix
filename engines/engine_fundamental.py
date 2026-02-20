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
# 1. è¨­å??‡å·¥?·å‡½å¼?(Setup & Utils)
# ==========================================
# ä½¿ç”¨ os.getenv è®€?–ç’°å¢ƒè???
# ä½¿ç”¨ os.environ.get è®€?–ç’°å¢ƒè???
# GEMINI_API_KEY is configured in main_app.py

def get_model() -> genai.GenerativeModel:
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
    return genai.GenerativeModel(model_id)

def remove_emojis(text: str) -> str:
    """ ç§»é™¤ Emojiï¼Œä??å ±?Šå?æ¥­åº¦ """
    return re.sub(r'[^\w\s,.:;!?()\[\]{}@#$%^&*\-+=/\\\'"<>~`|]', '', text)

def calculate_technicals_str(hist_df: pd.DataFrame) -> str:
    """ è¨ˆç??€è¡“æ?æ¨™ä¸¦?å‚³?¼å??–å?ä¸?(ä¾?AI ?±è?) """
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
# 2. ?¸å??¸æ??²å? (Data Mining)
# ==========================================
def get_stock_info(stock_code: str) -> Optional[Dict[str, Union[str, float]]]:
    """
    ?²å?ç¡¬æ•¸?šï??¡åƒ¹?PE?ROE?æ?è¡“æ?æ¨?
    ?å‚³: dict (?…å«?€?‰é??µæ•¸??
    """
    result: Dict[str, Union[str, float]] = {}
    
    try:
        # 1. ?•ç?ä»?¢¼
        code = stock_code.strip().upper()
        if code.isdigit():
            # ?ªå??—è©¦?°ç£ä¸Šå?ï¼Œè‹¥?¡å??—è©¦ä¸Šæ? (?™è£¡ç°¡å??è¼¯)
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

        # 2. ?²å??ºæœ¬??(Fundamentals)
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
            result['fcf'] = f"{fcf/1e8:.2f} ?? # è½‰æ??„ç‚º?®ä?
        else:
            result['fcf'] = "N/A"
        
        # 3. è¨ˆç??€è¡“æ?æ¨?(Technicals)
        result['technicals'] = calculate_technicals_str(hist)
        
        return result
        
    except Exception as e:
        print(f"Error in get_stock_info: {e}")
        return None

# ==========================================
# 3. æ¶ˆæ¯?¢çˆ¬??(Sentiment Mining)
# ==========================================

def get_anue_news_selenium(keyword: str) -> str:
    """ Anue ?…äº¨ç¶²çˆ¬??"""
    news_data = []
    chrome_options = Options()
    # ä½¿ç”¨?°ç? headless æ¨¡å?ï¼Œæ›´? ç©©å®?
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    # å¿½ç•¥?‘è??¯èª¤?‡è‡ª?•å??§åˆ¶?ç¤º
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
    """ Yahoo ?¡å??¬èŸ² """
    news_data = []
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = None
    try:
        # å¦‚æ??¯æ•¸å­—ä»£ç¢¼ï??€è¦å?ä¸?.TW ??.TWOï¼Œé€™è£¡?‡è¨­ä½¿ç”¨?…è¼¸?¥å??´ä»£ç¢¼æ??±å??¨è???
        # ?ºä?ä¿éšªèµ·è?ï¼Œè‹¥?ªå‚³?¥æ•¸å­—ï?è£œä? .TW
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
            if text and len(text) > 8 and "å»??" not in text:
                news_data.append(f"- [Yahoo] {text}")
                if len(news_data) >= 3: break
    except Exception: pass
    finally:
        if driver: driver.quit()
    return "\n".join(news_data) if news_data else "No specific news on Yahoo."

def get_ptt_sentiment(clean_code: str) -> str:
    """ PTT Stock ?ˆçˆ¬??"""
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
                # ç°¡å–®?¤æ–·?¯å¦?ºè???(?™è£¡ç°¡å??è¼¯)
                posts.append(f"- [{date_str}] {title}")
                if i >= 4: # ?–å? 5 ç¯‡å??œæ­¢
                    break
            except: continue
    except Exception: pass
    return "\n".join(posts) if posts else "Low retail discussion on PTT."

def get_sentiment_summary(stock_code: str) -> Dict[str, str]:
    """
    ?´å??°è???PTT ?¬èŸ²
    ?å‚³: dict (?…å« 'anue', 'yahoo', 'ptt' ä¸‰ç¨®ä¾†æ??„æ?è¦å?ä¸?
    """
    summary = {'anue': '', 'yahoo': '', 'ptt': ''}
    
    # ?•ç?ä»?¢¼?‡å?ç¨?(ç°¡å–®?•ç?ï¼Œå¯¦?›å¯?½é?è¦å??§è¡¨)
    clean_code = stock_code.replace(".TW", "").replace(".TWO", "")
    
    if clean_code.isdigit():
        # å¦‚æ??¯æ•¸å­—ä»£ç¢¼ï??¯èƒ½?€è¦å»?¥ä¸­?‡å?ç¨±ä??œæ–°??(?™è£¡ç°¡å??´æ¥?¨ä»£ç¢¼æ?)
        search_keyword = clean_code 
    else:
        search_keyword = clean_code
        
    summary['anue'] = get_anue_news_selenium(search_keyword)
    summary['yahoo'] = get_yahoo_news_selenium(stock_code)
    summary['ptt'] = get_ptt_sentiment(clean_code)
    
    return summary

# ==========================================
# 4. AI ?±å??Ÿæ? (AI Synthesis)
# ==========================================
def generate_ai_report(stock_code: str, data_info: Dict, sentiment_summary: Dict) -> str:
    """
    ?´å??¸æ??‡æ?ç·’ï??Ÿæ? Markdown ?±å?
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
    è®“ä½¿?¨è€…é?å°å ±?Šé€²è??å? (?¯æ´å°è©±è¨˜æ†¶)
    """
    model = get_model()
    
    # å»ºæ?å°è©±æ­·å²å­—ä¸²
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
