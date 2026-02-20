import requests
import xml.etree.ElementTree as ET

base_url = "https://news.google.com/rss/search"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

queries = {
    "cpi":    "美國最新 CPI 年增率 實際值",
    "pmi":    "美國最新 ISM 製造業 PMI 實際值",
    "export": "台灣最新 外銷訂單 年增率 YoY",
    "signal": "台灣最新 景氣燈號 分數 燈號"
}

for key, q in queries.items():
    params = {"q": q, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=10)
        print(f"=== {key} | HTTP {resp.status_code} ===")
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        print(f"  共 {len(items)} 則新聞")
        for item in items[:3]:
            title = item.find("title")
            pub   = item.find("pubDate")
            t = title.text if title is not None else "?"
            d = pub.text[:16] if pub is not None and pub.text else "?"
            print(f"  [{d}] {t}")
    except Exception as e:
        print(f"{key} Error: {e}")
    print()
