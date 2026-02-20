# engine_chips.py
import os
import google.generativeai as genai
from PIL import Image
from typing import List, Union, Dict, Any
from . import prompts

# ==========================================
# è¨­å??€
# ==========================================
# ä½¿ç”¨ os.getenv è®€?–ç’°å¢ƒè???
# ä½¿ç”¨ os.environ.get è®€?–ç’°å¢ƒè???
# GEMINI_API_KEY is configured in main_app.py

def get_vision_model() -> genai.GenerativeModel:
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-flash-latest")
    return genai.GenerativeModel(model_id)

def analyze_chips_image(uploaded_files: List[Any], stock_symbol: str, tech_data: Dict[str, Any] = None, is_short: bool = False) -> str:

    """
    ?¥æ”¶ Streamlit ä¸Šå‚³?„å??‡æ?æ¡ˆå?è¡¨ï??å‚³ AI ?†æ?çµæ?
    """
    print(f"Starting chips analysis for {stock_symbol} with {len(uploaded_files)} images...")
    model = get_vision_model()
    
    # æº–å??–ç?è³‡æ?
    image_parts = []
    for uploaded_file in uploaded_files:
        try:
            # Reset pointer to start just in case, though streamlit usually handles this
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            image_parts.append(image)
        except Exception as e:
            print(f"Error opening image: {e}")
            return f"?–ç?è®€?–å¤±?? {e}"
        
    if not image_parts:
        return "?ªæª¢æ¸¬åˆ°?–ç?ï¼Œè?ä¸Šå‚³ç±Œç¢¼?†ä??ªå???

    
    # ?¤æ–·?¯å¦?ºæ?è¡“é¢?´å?æ¨¡å?
    if tech_data:
        # å»ºæ??€è¡“é¢ Context String
        tech_context = f"""
        **å·²è?ç®—ç??€è¡“é¢?†æ??ƒè€ƒï?**
        * ?¶ç›¤?? {tech_data.get('Close', 'N/A')}
        * å¸ƒæ?å¸¶å¯¬è®Šå?: {tech_data.get('Bandwidth_Chg', 'N/A')}%
        * ä¸Šè??œç?: {tech_data.get('Upper_Slope_Pct', tech_data.get('Slope_Pct', 'N/A'))}%
        * ?ˆç??œç?: {tech_data.get('MA20_Slope_Pct', 'N/A')}%
        * ?äº¤?æ?: {tech_data.get('Vol_Ratio', 'N/A')}??
        * ä¸Šè?ä½ç½®: {tech_data.get('Pos_Upper', 'N/A')}%
        """
        # ?¹æ?æ¨¡å??¸æ??ç¤ºè©?
        if is_short:
            prompt = prompts.get_band_short_analysis_prompt(tech_context)
        else:
            prompt = prompts.get_band_long_analysis_prompt(tech_context)
    else:
        # ç´”ç?ç¢¼æ¨¡å¼?
        prompt = prompts.get_chips_analysis_prompt(stock_symbol)

    try:
        print("Sending request to Gemini Vision...")
        response = model.generate_content([prompt] + image_parts)
        print("Gemini response received.")
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"AI ?†æ?å¤±æ?: {e}"
