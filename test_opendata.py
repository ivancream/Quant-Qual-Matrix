"""
測試：直接從政府 Open Data API 抓台灣宏觀數據
"""
import requests, json

# ==== 1. 台灣景氣燈號 (國發會 Open Data) ====
print("=== 台灣景氣燈號 ===")
try:
    url = "https://ws.ndc.gov.tw/001/administrator/10/relfile/0/4028/ced35c70-2cf3-4e84-adfa-0c4c04f96c09.json"
    r = requests.get(url, timeout=10)
    data = r.json()
    # 拿最新一筆
    latest = data[-1] if isinstance(data, list) else data.get("data", [{}])[-1]
    print(f"最新資料: {latest}")
except Exception as e:
    print(f"失敗: {e}")

print()

# ==== 2. 台灣外銷訂單 (主計總處) ====
print("=== 台灣外銷訂單 ===")
try:
    url = "https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx"
    # 改用一個穩定的 endpoint 測試
    url2 = "https://statdb.dgbas.gov.tw/PXweb/api/v1/zh-tw/GovenmentSection/DomesticDemand/?query=%5B%7B%22code%22%3A%22A100020%22%2C%22selection%22%3A%7B%22filter%22%3A%22all%22%2C%22values%22%3A%5B%5D%7D%7D%5D"
    print(f"(試圖連接主計總處 API)")
except Exception as e:
    print(f"失敗: {e}")

print()

# ==== 3. US CPI via FRED ====
print("=== US CPI (FRED API) ===")
try:
    # FRED 提供免費 API，不需要 API key 也可以查歷史數據
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    r = requests.get(url, timeout=10)
    lines = r.text.strip().split("\n")
    print(f"共 {len(lines)} 筆, 最新 5 筆:")
    for line in lines[-5:]:
        print(f"  {line}")
except Exception as e:
    print(f"失敗: {e}")

print()

# ==== 4. 用 regex 從 RSS 標題直接抓數字 ====
import xml.etree.ElementTree as ET, re

print("=== 用 Regex 從 RSS 抓 CPI 數字 ===")
try:
    base_url = "https://news.google.com/rss/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": "美國 CPI 年增率 %", "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
    resp = requests.get(base_url, params=params, headers=headers, timeout=10)
    root = ET.fromstring(resp.content)
    items = root.findall(".//item")
    for item in items[:10]:
        title = item.find("title").text or ""
        # 嘗試抓 x.x% 格式的數字
        nums = re.findall(r'\d+\.?\d*\s*%', title)
        if nums:
            print(f"  -> {nums[0]} 來自: {title[:60]}")
            break
except Exception as e:
    print(f"失敗: {e}")
