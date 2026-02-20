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
# 設定與常數
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
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash-latest")
    return genai.GenerativeModel(model_id)

# ==========================================
# 1. 市場數據 (Market Metrics)
# ==========================================
def get_market_metrics(market_name: str) -> Tuple[str, str]:
    """
    抓取指定市場的最新指數與漲跌幅
    回傳: (metrics_str, status_str)
    """
    symbol = MARKET_MAP.get(market_name, "^TWII")
    try:
        ticker = yf.Ticker(symbol)
        # 用 5 天數據避免遇到假日或休市導致無數據
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return "No Data", "Unknown"
            
        last_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        
        metrics = f"{last_close:,.0f} ({pct_change:+.2f}%)"
        
        # 簡單趨勢判斷
        status = "Bullish" if pct_change > 0.5 else ("Bearish" if pct_change < -0.5 else "Neutral")
        
        return metrics, status
    except Exception as e:
        return f"Error: {e}", "Error"

# ==========================================
# 2. 宏觀數據 (Macro & FX)
# ==========================================
def get_short_term_data() -> Dict[str, str]:
    """
    短線攻擊力道 (Daily):
    1. 台幣匯率 (USDTWD)
    2. VIX (恐慌指數)
    3. 10年美債殖利率 (US 10Y)
    4. 黃金 (Gold)
    回傳: Dict with keys ('twd', 'vix', 'bond', 'gold')
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
    """ 回傳 list of (pub_date, title) from Google News RSS """
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
    嘗試從標題列表中用 Regex 直接找出數字。
    patterns: list of regex pattern strings
    回傳第一個找到的匹配結果，找不到回傳 None
    """
    for _, title in headlines:
        for pattern in patterns:
            m = re.search(pattern, title)
            if m:
                return m.group(0).strip()
    return None

def get_long_term_data() -> Dict[str, str]:
    """
    波段持股水位 (Monthly): US CPI, TW Export, PMI, Light Signal
    策略: Regex 實字抓取為主， AI 補充分析為辅
    """
    results = {"cpi": "需查詢", "export": "需查詢", "pmi": "需查詢", "signal": "需查詢"}

    # ---- 1. US CPI ----
    headlines = _fetch_rss_headlines("美國 CPI 年增率")
    # 充分考慮中文標題中的 "年增x%" 、"CPI年增x%" 、"CPI x%" 之類
    val = _extract_number_from_headlines(headlines, [
        r'年增[率]?\s*([\d.]+\s*%)',
        r'CPI[^\d]*([\d.]+\s*%)',
        r'([\d.]+)\s*%.*CPI',
    ])
    if val:
        results["cpi"] = val
    else:
        # AI 備案
        try:
            news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
            model = get_model()
            r = model.generate_content(f"從以下新聞找出美國最近一期CPI年增率的實際數值，直接回傳數字如 3.2%，沒有就回傳空白：\n{news_text}")
            v = r.text.strip()
            if re.search(r'[\d.]+\s*%', v):
                results["cpi"] = v
        except: pass

    # ---- 2. US PMI ----
    headlines = _fetch_rss_headlines("美國 ISM 製造業 PMI")
    val = _extract_number_from_headlines(headlines, [
        r'PMI[^\d]*([\d.]+)',
        r'([\d.]+)[^\d]*PMI',
        r'製造業[^\d]*([\d.]+)',
    ])
    if val:
        results["pmi"] = val
    else:
        try:
            news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
            model = get_model()
            r = model.generate_content(
                f"從以下新聞找出美國最近一期ISM製造業PMI的實際數字。"
                f"只請輸出純數字（例如: 48.5），不要任何文字。找不到就回傳空白：\n{news_text}"
            )
            v = r.text.strip()[:10]  # 截斷保護
            m = re.search(r'[\d.]+', v)
            if m:
                results["pmi"] = m.group(0)
        except: pass

    # ---- 3. 台灣外銷訂單 ----
    headlines = _fetch_rss_headlines("台灣 外銷訂單 年增率")
    val = _extract_number_from_headlines(headlines, [
        r'年[增减減][率]?\s*([\+\-]?[\d.]+\s*%)',
        r'([\+\-]?[\d.]+\s*%).*外銷',
        r'外銷[^\d]*([\d.]+\s*%)',
    ])
    if val:
        results["export"] = val
    else:
        try:
            news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
            model = get_model()
            r = model.generate_content(f"從以下新聞找出台灣最近一期外銷訂單年增率的實際數值，直接回傳如 +5.2% 或 -1.3%，沒有就回傳空白：\n{news_text}")
            v = r.text.strip()
            if re.search(r'[\d.]+\s*%', v):
                results["export"] = v
        except: pass

    # ---- 4. 台灣景氣燈號 ----
    headlines = _fetch_rss_headlines("台灣 景氣燈號 分數")
    # 先傳燈號名稱
    val = _extract_number_from_headlines(headlines, [
        r'((?:紅|黃紅|黃|綠|藍|低迷|熱絡)[燈])[^\d]{0,10}(\d+)[\s分]',
        r'(\d+)[\s分].*?((?:紅|黃紅|黃|綠|藍)[燈])',
    ])
    if val:
        results["signal"] = val
    else:
        # 先在標題裡找 燈+分數 組合
        for _, title in headlines[:8]:
            m = re.search(r'((?:紅|黃紅|黃|綠|藍)燈)[^0-9]{0,15}([0-9]+)[分分]', title)
            if m:
                results["signal"] = f"{m.group(1)} {m.group(2)}分"
                break
            m2 = re.search(r'([0-9]+)[分分][^0-9]{0,5}((?:紅|黃紅|黃|綠|藍)燈)', title)
            if m2:
                results["signal"] = f"{m2.group(2)} {m2.group(1)}分"
                break
        
        if results["signal"] == "需查詢":
            try:
                news_text = "\n".join(f"[{d}] {t}" for d, t in headlines[:5])
                model = get_model()
                r = model.generate_content(f"從以下新聞找出台灣最近一期景氣燈號與分數，直接回傳如 綠燈24分 或 黃紅燈35分，沒有就回傳空白：\n{news_text}")
                v = r.text.strip()
                if v:
                    results["signal"] = v
            except: pass

    print(f"[Macro Data] {results}")
    return results



def get_fx_data(market_name: str) -> str:
    """
    抓取區域報告所需的匯率 (USDTWD, JPY, HKD)
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
# 3. 新聞爬蟲 (News)
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
# 4. AI 報告
# ==========================================
def generate_global_report(market: str, metrics: str, macro: str, fx: str, commodities: str, news: str) -> str:
    """
    綜合宏觀數據生成報告
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
    與全球宏觀策略師進行對談
    """
    model = get_model()
    # 建立對談歷史格式，只包含最近幾次以節省 Token 並維持一致性
    history = []
    for m in chat_history[-10:]: # 取最近 10 則
        history.append({"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]})
    
    chat = model.start_chat(history=history)
    
    prompt = f"""
    你是剛才撰寫這份報告的「全球宏觀策略師」。
    初始報告內容如下作為背景：
    {report_context}
    
    請針對使用者的問題進行專業、客觀且具備洞察力的回答。若問題涉及特定的投資建議，請維持中立並強調風險。
    使用繁體中文回答。
    
    使用者問題：{user_msg}
    """
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"對談出錯: {e}"
