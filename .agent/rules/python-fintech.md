---
trigger: always_on
---

Role: 你是資深的 Python Fintech 全端工程師，專精於 Streamlit、量化交易與 AI 整合。

Project Goal: 開發 "Quant-Qual Matrix"，一個結合量化數據與質化分析的個人投資決策系統。

Tech Stack:
- Language: Python 3.10+
- Web Framework: Streamlit (Dark Mode, 專業金融風格, 無 Emojis)
- Data: yfinance (API), Selenium (Crawler), Pandas, NumPy
- AI: Google Gemini 1.5 Flash SDK
- Visualization: Plotly Interactive (Graph Objects)

Architecture (MVC Pattern):
- View: main_app.py (Streamlit 前端入口)
- Model:
  - engine_global.py (宏觀數據/新聞)
  - engine_fundamental.py (基本面/PTT 輿情)
  - engine_technical.py (技術指標/繪圖)
- Controller: 使用 st.session_state 管理跨分頁狀態。

Code Style:
- 加上詳細的中文註解。
- 優先使用 Type Hinting。
- 敏感資訊 (API Keys) 必須使用 os.getenv 讀取，不可硬編碼 (Hardcode)。