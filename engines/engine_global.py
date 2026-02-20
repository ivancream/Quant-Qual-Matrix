import os
import time
import re
import requests
import xml.etree.ElementTree as ET
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Tuple, Any


# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ==========================================
# è¨­å??‡å¸¸??
# GEMINI_API_KEY is configured in dashboard.py

MARKET_MAP = {
    "TAIEX (TWII)": "^TWII",
    "Nikkei 225 (N225)": "^N225",
    "KOSPI (KS11)": "^KS11",
    "Nasdaq (IXIC)": "^IXIC",
    "PHLX Semi (SOX)": "^SOX"
}

from . import prompts

def get_model() -> genai.GenerativeModel:
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
    return genai.GenerativeModel(model_id)

# ==========================================
# 1. å¸‚å ´?¸æ? (Market Metrics)
# ==========================================
def get_market_metrics(market_name: str) -> Tuple[str, str]:
    """
    ?“å??‡å?å¸‚å ´?„æ??°æ??¸è?æ¼²è?å¹?
    ?å‚³: (metrics_str, status_str)
    """
    symbol = MARKET_MAP.get(market_name, "^TWII")
    try:
        ticker = yf.Ticker(symbol)
        # ??5 å¤©æ•¸?šé¿?é??°å??¥æ?ä¼‘å?å°è‡´?¡æ•¸??
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return "No Data", "Unknown"
            
        last_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        
        metrics = f"{last_close:,.0f} ({pct_change:+.2f}%)"
        
        # ç°¡å–®è¶¨å‹¢?¤æ–·
        status = "Bullish" if pct_change > 0.5 else ("Bearish" if pct_change < -0.5 else "Neutral")
        
        return metrics, status
    except Exception as e:
        return f"Error: {e}", "Error"

# ==========================================
# 2. å®è??¸æ? (Macro & FX)
# ==========================================
def get_short_term_data() -> Dict[str, str]:
    """
    ?­ç??»æ??›é? (Daily):
    1. ?°å¹£?¯ç? (USDTWD)
    2. VIX (?æ??‡æ•¸)
    3. 10å¹´ç??µæ??©ç? (US 10Y)
    4. é»ƒé? (Gold)
    ?å‚³: Dict with keys ('twd', 'vix', 'bond', 'gold')
    """
    data = {"twd": "N/A", "vix": "N/A", "bond": "N/A", "gold": "N/A"}
    try:
        # Batch Fetch
        tickers = yf.Tickers("USDTWD=X ^VIX ^TNX GC=F")
        
        # TWD
        try:
            twd = tickers.tickers['USDTWD=X'].history(period="5d")['Close'].iloc[-1]
            data['twd'] = f"{twd:.2f}"
        except: pass

        # VIX
        try:
            vix = tickers.tickers['^VIX'].history(period="5d")['Close'].iloc[-1]
            data['vix'] = f"{vix:.2f}"
        except: pass

        # Bond
        try:
            bond = tickers.tickers['^TNX'].history(period="5d")['Close'].iloc[-1]
            data['bond'] = f"{bond:.2f}%"
        except: pass

        # Gold
        try:
            gold = tickers.tickers['GC=F'].history(period="5d")['Close'].iloc[-1]
            data['gold'] = f"{gold:,.1f}"
        except: pass

        return data
    except Exception as e:
        print(f"Short term data error: {e}")
        return data

import re

def _fetch_rss_headlines(query: str, max_items: int = 10) -> list:
    """ ?å‚³ list of (pub_date, title) from Google News RSS """
    base_url = "https://news.google.com/rss/search"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    params = {"q": query, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=10)
        root = ET.fromstring(resp.content)
        results = []
        for item in root.findall(".//item")[:max_items]:
            title = item.find("title")
            pub   = item.find("pubDate")
            if title is not None and title.text:
                results.append((pub.text[:16] if pub is not None and pub.text else "", title.text))
        return results
    except:
        return []

def _extract_number_from_headlines(headlines: list, patterns: list) -> str:
    """
    ?—è©¦å¾æ?é¡Œå?è¡¨ä¸­??Regex ?´æ¥?¾å‡º?¸å???
    patterns: list of regex pattern strings
    ?å‚³ç¬¬ä??‹æ‰¾?°ç??¹é?çµæ?ï¼Œæ‰¾ä¸åˆ°?å‚³ None
    """
    for _, title in headlines:
        for pattern in patterns:
            m = re.search(pattern, title)
            if m:
                return m.group(0).strip()
    return None

def get_long_term_data() -> Dict[str, str]:
    """
    æ³¢æ®µ?è‚¡æ°´ä? (Monthly): US CPI, TW Export, PMI, Light Signal
    ç­–ç•¥: Regex å¯¦å??“å??ºä¸»ï¼?AI è£œå??†æ??ºè?
    """
    results = {"cpi": "?€?¥è©¢", "export": "?€?¥è©¢", "pmi": "?€?¥è©¢", "signal": "?€?¥è©¢"}

    # ---- 1. US CPI ----
    headlines = _fetch_rss_headlines("ç¾å? CPI å¹´å???)
    # ?…å??ƒæ…®ä¸­æ?æ¨™é?ä¸­ç? "å¹´å?x%" ??CPIå¹´å?x%" ??CPI x%" ä¹‹é?
    val = _extract_number_from_headlines(headlines, [
        r'å¹´å?[?‡]?\s*([\d.]+\s*%)',
        r'CPI[^\d]*([\d.]+\s*%)',
        r'([\d.]+)\s*%.*CPI',
    ])
    if val:
        results["cpi"] = val
    else:
        # AI ?™æ?
        try:
            news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
            model = get_model()
            r = model.generate_content(f"å¾ä»¥ä¸‹æ–°?æ‰¾?ºç??‹æ?è¿‘ä??ŸCPIå¹´å??‡ç?å¯¦é??¸å€¼ï??´æ¥?å‚³?¸å?å¦?3.2%ï¼Œæ??‰å°±?å‚³ç©ºç™½ï¼š\n{news_text}")
            v = r.text.strip()
            if re.search(r'[\d.]+\s*%', v):
                results["cpi"] = v
        except: pass

    # ---- 2. US PMI ----
    headlines = _fetch_rss_headlines("ç¾å? ISM è£½é€ æ¥­ PMI")
    val = _extract_number_from_headlines(headlines, [
        r'PMI[^\d]*([\d.]+)',
        r'([\d.]+)[^\d]*PMI',
        r'è£½é€ æ¥­[^\d]*([\d.]+)',
    ])
    if val:
        results["pmi"] = val
    else:
        try:
            news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
            model = get_model()
            r = model.generate_content(
                f"å¾ä»¥ä¸‹æ–°?æ‰¾?ºç??‹æ?è¿‘ä??ŸISMè£½é€ æ¥­PMI?„å¯¦?›æ•¸å­—ã€?
                f"?ªè?è¼¸å‡ºç´”æ•¸å­—ï?ä¾‹å?: 48.5ï¼‰ï?ä¸è?ä»»ä??‡å??‚æ‰¾ä¸åˆ°å°±å??³ç©º?½ï?\n{news_text}"
            )
            v = r.text.strip()[:10]  # ?ªæ–·ä¿è­·
            m = re.search(r'[\d.]+', v)
            if m:
                results["pmi"] = m.group(0)
        except: pass

    # ---- 3. ?°ç£å¤–éŠ·è¨‚å–® ----
    headlines = _fetch_rss_headlines("?°ç£ å¤–éŠ·è¨‚å–® å¹´å???)
    val = _extract_number_from_headlines(headlines, [
        r'å¹´[å¢å?æ¸›][?‡]?\s*([\+\-]?[\d.]+\s*%)',
        r'([\+\-]?[\d.]+\s*%).*å¤–éŠ·',
        r'å¤–éŠ·[^\d]*([\d.]+\s*%)',
    ])
    if val:
        results["export"] = val
    else:
        try:
            news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
            model = get_model()
            r = model.generate_content(f"å¾ä»¥ä¸‹æ–°?æ‰¾?ºå°???è¿‘ä??Ÿå??·è??®å¹´å¢ç??„å¯¦?›æ•¸?¼ï??´æ¥?å‚³å¦?+5.2% ??-1.3%ï¼Œæ??‰å°±?å‚³ç©ºç™½ï¼š\n{news_text}")
            v = r.text.strip()
            if re.search(r'[\d.]+\s*%', v):
                results["export"] = v
        except: pass

    # ---- 4. ?°ç£?¯æ°£?ˆè? ----
    headlines = _fetch_rss_headlines("?°ç£ ?¯æ°£?ˆè? ?†æ•¸")
    # ?ˆå‚³?ˆè??ç¨±
    val = _extract_number_from_headlines(headlines, [
        r'((?:ç´…|é»ƒç?|é»ƒ|ç¶ |?|ä½è¿·|?±çµ¡)[?ˆ])[^\d]{0,10}(\d+)[\s?†]',
        r'(\d+)[\s?†].*?((?:ç´…|é»ƒç?|é»ƒ|ç¶ |??[?ˆ])',
    ])
    if val:
        results["signal"] = val
    else:
        # ?ˆåœ¨æ¨™é?è£¡æ‰¾ ???†æ•¸ çµ„å?
        for _, title in headlines[:8]:
            m = re.search(r'((?:ç´…|é»ƒç?|é»ƒ|ç¶ |????[^0-9]{0,15}([0-9]+)[?†å?]', title)
            if m:
                results["signal"] = f"{m.group(1)} {m.group(2)}??
                break
            m2 = re.search(r'([0-9]+)[?†å?][^0-9]{0,5}((?:ç´…|é»ƒç?|é»ƒ|ç¶ |????', title)
            if m2:
                results["signal"] = f"{m2.group(2)} {m2.group(1)}??
                break
        
        if results["signal"] == "?€?¥è©¢":
            try:
                news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
                model = get_model()
                r = model.generate_content(f"å¾ä»¥ä¸‹æ–°?æ‰¾?ºå°???è¿‘ä??Ÿæ™¯æ°???Ÿè??†æ•¸ï¼Œç›´?¥å??³å? ç¶ ç?24????é»ƒç???5?†ï?æ²’æ?å°±å??³ç©º?½ï?\n{news_text}")
                v = r.text.strip()
                if v:
                    results["signal"] = v
            except: pass

    print(f"[Macro Data] {results}")
    return results



def get_fx_data(market_name: str) -> str:
    """
    ?“å??€?Ÿå ±?Šæ??€?„åŒ¯??(USDTWD, JPY, HKD)
    """
    try:
        if "TAIEX" in market_name or "TW" in market_name: 
            pair = "USDTWD=X"
        elif "Nikkei" in market_name or "JP" in market_name: 
            pair = "JPY=X"
        elif "HK" in market_name: 
            pair = "HKD=X"
        else: 
            return "USD Index (DXY)"
            
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            return f"{pair}: {rate:.2f}"
        return "N/A"
    except:
        return "N/A"

# ==========================================
# 3. ?°è??¬èŸ² (News)
# ==========================================
def get_investing_news(market_name: str) -> str:
    """
    Crawls Investing.com (Global Finance) using Selenium.
    Targets specific market sections based on the input.
    """
    news_list = []
    
    # Setup Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Define URL based on market
        # Investing.com is tough to scrape, we might fall back to a reliable aggregator like CNBC or Reuters if Investing blocks,
        # but the spec asks for Investing.com. We'll try the news section.
        # Alternatively, use Anue Global for stability if Investing.com has anti-bot cloudflare active.
        # Let's try to stick to the 'spirit' of the spec with a reliable source. 
        # For this implementation, I will use Google Finance or Yahoo Finance Global News as a stable proxy 
        # if Investing.com is inaccessible, but let's try a generic financial news aggregator.
        # User explicitly asked for "Investing.com", so we try. 
        
        url = "https://www.investing.com/news/stock-market-news"
        if "TW" in market_name:
             url = "https://tw.investing.com/news/stock-market-news" # Trad. Chinese version
        
        driver.get(url)
        time.sleep(3) # Wait for JS load
        
        # Investing.com structure varies. Look for main articles.
        # Common class for articles might change, so we search by standard tags.
        # Trying a robust selector
        articles = driver.find_elements(By.CSS_SELECTOR, "a.title")
        
        if not articles:
             # Fallback selector
             articles = driver.find_elements(By.XPATH, "//article/a[contains(@class, 'title')]")
        
        count = 0
        for art in articles:
            text = art.text.strip()
            if text and len(text) > 10:
                news_list.append(f"- {text}")
                count += 1
                if count >= 5:
                    break
                    
        if not news_list:
            return "No headlines found. Source might be protecting against bots."
            
        return "\n".join(news_list)
        
    except Exception as e:
        return f"News Crawling Error: {e}"
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# ==========================================
# 4. AI ?±å?
# ==========================================
def generate_global_report(market: str, metrics: str, macro: str, fx: str, commodities: str, news: str) -> str:
    """
    ç¶œå?å®è??¸æ??Ÿæ??±å?
    """
    model = get_model()
    prompt = prompts.get_global_report_prompt(market, metrics, macro, fx, commodities, news)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Report Failed: {e}"

def chat_with_global_analyst(user_msg: str, report_context: str, chat_history: List[Dict]) -> str:
    """
    ?‡å…¨?ƒå?è§€ç­–ç•¥å¸«é€²è?å°è?
    """
    model = get_model()
    # å»ºç?å°è?æ­·å²?¼å?ï¼Œåª?…å«?€è¿‘å¹¾æ¬¡ä»¥ç¯€??Token ä¸¦ç¶­?ä??´æ€?
    history = []
    for m in chat_history[-10:]: # ?–æ?è¿?10 ??
        history.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
    
    chat = model.start_chat(history=history)
    
    prompt = f"""
    ä½ æ˜¯?›æ??°å¯«?™ä»½?±å??„ã€Œå…¨?ƒå?è§€ç­–ç•¥å¸«ã€ã€?
    ?å??±å??§å®¹å¦‚ä?ä½œç‚º?Œæ™¯ï¼?
    {report_context}
    
    è«‹é?å°ä½¿?¨è€…ç??é??²è?å°ˆæ¥­?å®¢è§€ä¸”å…·?™æ?å¯Ÿå??„å?ç­”ã€‚è‹¥?é?æ¶‰å??¹å??„æ?è³‡å»ºè­°ï?è«‹ç¶­?ä¸­ç«‹ä¸¦å¼·èª¿é¢¨éšª??
    ä½¿ç”¨ç¹é?ä¸­æ??ç???
    
    ä½¿ç”¨?…å?é¡Œï?{user_msg}
    """
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"å°è??ºéŒ¯: {e}"
