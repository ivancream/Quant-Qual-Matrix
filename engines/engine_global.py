import os
import time
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

import requests
import xml.etree.ElementTree as ET

def get_long_term_data() -> Dict[str, str]:
    """
    波段持股水位 (Monthly): US CPI, TW Export, PMI, Light Signal
    改進版 v3: 使用 Google News RSS Feed (最穩定、無需爬蟲對抗) + LLM 提取
    """
    data_defaults = {"cpi": "需查詢", "export": "需查詢", "pmi": "需查詢", "signal": "需查詢"}
    
    collected_info = ""
    
    # 定義精準的 RSS 搜尋關鍵字 (包含時間限制詞以確保新穎性)
    queries = [
        ("cpi", "美國 CPI 年增率"),
        ("pmi", "美國 ISM 製造業 PMI"),
        ("export", "台灣 外銷訂單 年增率"),
        ("signal", "台灣 景氣燈號 分數")
    ]
    
    base_url = "https://news.google.com/rss/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print("Fetching macro data via Google News RSS...")
    
    for key, q in queries:
        try:
            params = {
                "q": q,
                "hl": "zh-TW",
                "gl": "TW",
                "ceid": "TW:zh-Hant"
            }
            resp = requests.get(base_url, params=params, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                try:
                    root = ET.fromstring(resp.content)
                    items = root.findall(".//item")
                    
                    collected_info += f"\n--- {key.upper()} 相關新聞 ---\n"
                    # 取前 5 則最新新聞的標題與時間
                    for item in items[:5]:
                        title = item.find("title").text
                        pub_date = item.find("pubDate").text
                        collected_info += f"[{pub_date}] {title}\n"
                except ET.ParseError:
                    print(f"RSS Parse Error for {key}")
            else:
                print(f"RSS Request failed for {key}: {resp.status_code}")
                
        except Exception as e:
            print(f"Error fetching {key}: {e}")

    if not collected_info:
        return data_defaults

    # LLM Parsing
    try:
        model = get_model()
        prompt = f"""
        你是一個精準的財經數據提取專家。請從以下新聞標題與時間中，判斷並提取「最新公佈」的經濟數據數值。
        注意：請優先參考日期最近的新聞。
        
        請提取以下四個數值 (取最新的「實際值」Actual，不要預測值):
        1. US CPI (美國消費者物價指數年增率, 例如: 3.1%)
        2. Taiwan Export Orders (台灣外銷訂單年增率, 需找 YoY, 例如: -2.3% 或 +1.2%)
        3. US PMI (美國 ISM 製造業指數, 例如: 48.5)
        4. Taiwan Light Signal (台灣景氣燈號分數/燈號, 例如: 黃紅燈 34分)

        新聞資料來源：
        {collected_info}

        請嚴格以 JSON 格式回傳，無Markdown：
        {{"cpi": "...", "export": "...", "pmi": "...", "signal": "..."}}
        若資料不足或過期太久請填 "需查詢"。
        """
        
        response = model.generate_content(prompt)
        res_text = response.text
        
        import json, re
        match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
            for k in data_defaults:
                if extracted.get(k) and extracted.get(k) not in ["需查詢", "N/A"]:
                    data_defaults[k] = extracted[k]
                    
    except Exception as e:
        print(f"LLM Parsing failed: {e}")
        
    return data_defaults

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
