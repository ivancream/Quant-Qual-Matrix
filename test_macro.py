import sys
sys.path.insert(0, r"c:\VS_Quant_Qual_1")

import os
from dotenv import load_dotenv
load_dotenv(r"c:\VS_Quant_Qual_1\.env")

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from engines.engine_global import get_long_term_data
result = get_long_term_data()
print("\n=== 最終結果 ===")
for k, v in result.items():
    print(f"  {k}: {v}")
