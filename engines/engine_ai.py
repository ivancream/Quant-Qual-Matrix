# engines/engine_ai.py
"""
çµ±ä? AI å°è©±å¼•æ? (Unified AI Brain)

?¶æ?èªªæ?ï¼?
- ç¶­è­·ä»¥ã€Œè‚¡ç¥¨ä»£?Ÿã€ç‚º?µç?è·¨æ¨¡çµ?Context å½™æ•´ (Brain)
- æ¯å€‹å??æ¨¡çµ„å??å?ï¼Œå??˜è?æ³¨å…¥ Brain
- ?€?‰å?è©±éƒ½å¸¶å???Brain Contextï¼Œè? AI ?¥é??¨æˆ¶?¨å?æ¨¡ç??”ç©¶äº†ä?éº?
- Streamlit ?é? session_state ?ä???Brain ?€??

Brain çµæ? (å­˜æ–¼ st.session_state.ai_brain)ï¼?
{
    "active_ticker": "2330",
    "contexts": {
        "2330": {
            "fundamental": "P/E 15x...",
            "technical":   "RSI=65, MAå¤šæ?...",
            "chips":       "ä¸»å?è³??3000å¼?..",
            "band_long":   "?‹å??? ä¸Šè??œç? 0.35%...",
            "band_short":  "?ˆç?ä¸‹å?, è·Œç ´?ˆç?..."
        }
    },
    "chat_history": [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from PIL import Image
from . import prompts


# ==========================================
# æ¨¡ç??ç¨±å°æ?ï¼ˆç”¨?¼é¡¯ç¤ºè? Prompt ä¸­ï?
# ==========================================
MODULE_LABELS: Dict[str, str] = {
    "fundamental": "?ºæœ¬??/ è²¡å ±?†æ?",
    "technical":   "?€è¡“æ?æ¨™å???,
    "chips":       "ç±Œç¢¼?¢å???,
    "band_long":   "æ³¢æ®µå¤šæ–¹ç­–ç•¥",
    "band_short":  "æ³¢æ®µç©ºæ–¹ç­–ç•¥",
}


def _get_model() -> genai.GenerativeModel:
    """?–å? Gemini æ¨¡å?å¯¦ä???""
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
    return genai.GenerativeModel(
        model_id,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
        )
    )


# ==========================================
# Brain ?ä??½æ•¸
# ==========================================

def get_empty_brain() -> Dict[str, Any]:
    """?å‚³ä¸€?‹å?å§‹å??„ç©º Brain çµæ???""
    return {
        "active_ticker": None,
        "contexts": {},
        "chat_history": []
    }


def update_context(brain: Dict, ticker: str, module: str, summary: str) -> Dict:
    """
    å°‡æ?æ¨¡ç??„å??æ?è¦æ³¨??Brain??
    æ¯æ¬¡?†æ?å®Œæ?å¾Œå‘¼?«ï?è®?AI ?¨å?çºŒå?è©±ä¸­?½å??¨ã€?

    Args:
        brain:   ?¾æ???Brain dict (ä¾†è‡ª session_state)
        ticker:  ?¡ç¥¨ä»?? (e.g. "2330.TW")
        module:  æ¨¡ç??ç¨± ("fundamental", "technical", etc.)
        summary: è©²æ¨¡çµ„ç??‡å??˜è? (?¯ä»¥??AI ?±å??–æ•¸?šå?ä¸?

    Returns:
        ?´æ–°å¾Œç? Brain dict
    """
    if ticker not in brain["contexts"]:
        brain["contexts"][ticker] = {}
    brain["contexts"][ticker][module] = summary
    brain["active_ticker"] = ticker
    return brain


def build_context_string(brain: Dict, ticker: str) -> str:
    """
    å°?Brain ä¸­æ??¯è‚¡ç¥¨ç??€?‰æ¨¡çµ?Context ?´å??ä?æ®?AI ?¯è??„å?ä¸²ã€?

    Args:
        brain:  Brain dict
        ticker: ?¡ç¥¨ä»??

    Returns:
        ?´å?å¾Œç? Context å­—ä¸²
    """
    ctx = brain.get("contexts", {}).get(ticker, {})
    if not ctx:
        return "ï¼ˆç›®?å??¡å…¶ä»–æ¨¡çµ„ç??†æ?è³‡æ?ï¼?

    parts = []
    for module_key, label in MODULE_LABELS.items():
        if module_key in ctx:
            parts.append(f"### [{label}]\n{ctx[module_key]}")

    return "\n\n".join(parts)


# ==========================================
# ?¸å??Ÿèƒ½?½æ•¸
# ==========================================

def initial_report(
    brain: Dict,
    ticker: str,
    module: str,
    new_data: str,
    images: Optional[List[Any]] = None,
    custom_prompt_override: Optional[str] = None
) -> str:
    """
    ?Ÿæ??å??†æ??±å?ï¼Œè‡ª?•æ•´??Brain ä¸­æ??‰å·²?¥ç?è·¨æ¨¡çµ?Context??

    Args:
        brain:                 Brain dict (ä¾†è‡ª session_state)
        ticker:                ?¡ç¥¨ä»??
        module:                ?¶å??¼å«?„æ¨¡çµ„å?ç¨?
        new_data:              ?¬æ¨¡çµ„ç??€?°æ•¸?šå?ä¸?
        images:                ?¥æ??–ç? (ç±Œç¢¼??ï¼Œå‚³??PIL Image list
        custom_prompt_override: ?¥æ?å®šï??´æ¥ä½¿ç”¨æ­?Prompt è¦†è??è¨­ Prompt

    Returns:
        AI ?è??„å ±?Šæ?å­?
    """
    model = _get_model()
    cross_context = build_context_string(brain, ticker)
    module_label = MODULE_LABELS.get(module, module)

    if custom_prompt_override:
        prompt = custom_prompt_override
    else:
        prompt = prompts.get_unified_initial_prompt(ticker, module_label, cross_context, new_data)

    try:
        if images:
            # Vision æ¨¡å?ï¼ˆå«?–ç?ï¼?
            response = model.generate_content([prompt] + images)
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI ?†æ?å¤±æ?ï¼š{e}"


def chat(
    brain: Dict,
    ticker: str,
    user_input: str,
) -> Tuple[str, Dict]:
    """
    å¸¶å??´è·¨æ¨¡ç?è¨˜æ†¶?„å?è©±å?è¦†ã€?

    Args:
        brain:      Brain dict (ä¾†è‡ª session_state)
        ticker:     ?¡ç¥¨ä»??
        user_input: ?¨æˆ¶è¼¸å…¥

    Returns:
        (AI ?è??‡å?, ?´æ–°å¾Œç? Brain dict)
    """
    model = _get_model()
    cross_context = build_context_string(brain, ticker)

    # å»ºæ?å°è©±æ­·å²å­—ä¸²
    history_text = ""
    for msg in brain.get("chat_history", []):
        role = "?¨æˆ¶" if msg["role"] == "user" else "?†æ?å¸?
        history_text += f"{role}: {msg['content']}\n\n"

    # ä½¿ç”¨ prompts.py ä¸­ç?çµ±ä??ç¤ºè©?
    prompt = prompts.get_unified_chat_prompt(ticker, cross_context, history_text, user_input)

    try:
        response = model.generate_content(prompt)
        reply = response.text
    except Exception as e:
        reply = f"?è?å¤±æ?ï¼š{e}"

    # ?´æ–°å°è©±æ­·å²
    brain["chat_history"].append({"role": "user", "content": user_input})
    brain["chat_history"].append({"role": "assistant", "content": reply})

    return reply, brain


def stream_chat(
    brain: Dict,
    ticker: str,
    user_input: str,
) -> Tuple[Any, Dict]:
    """
    Streaming ?ˆæœ¬?„å?è©±ã€‚å???(generator, brain_updater_fn)??
    ?¨æ?ï¼?
        stream, update_fn = engine_ai.stream_chat(brain, ticker, q)
        reply = st.write_stream(stream)
        st.session_state.ai_brain = update_fn(reply)
    """
    model = _get_model()
    cross_context = build_context_string(brain, ticker)

    history_text = ""
    for msg in brain.get("chat_history", []):
        role = "?¨æˆ¶" if msg["role"] == "user" else "?†æ?å¸?
        history_text += f"{role}: {msg['content']}\n\n"

    prompt = prompts.get_unified_chat_prompt(ticker, cross_context, history_text, user_input)

    def _generator():
        try:
            for chunk in model.generate_content(prompt, stream=True):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"?è?å¤±æ?ï¼š{e}"

    def _update_brain(reply: str) -> Dict:
        brain["chat_history"].append({"role": "user", "content": user_input})
        brain["chat_history"].append({"role": "assistant", "content": reply})
        return brain

    return _generator(), _update_brain


def stream_initial_report(
    brain: Dict,
    ticker: str,
    module: str,
    new_data: str,
    images: Optional[List[Any]] = None,
    custom_prompt_override: Optional[str] = None
) -> Any:
    """
    Streaming ?ˆæœ¬?„å?å§‹å ±?Šç??ã€?
    ?¨æ?ï¼šreply = st.write_stream(engine_ai.stream_initial_report(...))
    """
    model = _get_model()
    cross_context = build_context_string(brain, ticker)
    module_label = MODULE_LABELS.get(module, module)

    if custom_prompt_override:
        prompt = custom_prompt_override
    else:
        prompt = prompts.get_unified_initial_prompt(ticker, module_label, cross_context, new_data)

    def _generator():
        try:
            if images:
                # Vision æ¨¡å?ä¸æ”¯??streamingï¼Œç›´?¥å???
                response = model.generate_content([prompt] + images)
                yield response.text
            else:
                for chunk in model.generate_content(prompt, stream=True):
                    if chunk.text:
                        yield chunk.text
        except Exception as e:
            yield f"AI ?Ÿæ?å¤±æ?ï¼š{e}"

    return _generator()


def clear_chat(brain: Dict) -> Dict:
    """æ¸…é™¤å°è©±æ­·å²ï¼ˆä?ä¿ç? Contextï¼‰ã€?""
    brain["chat_history"] = []
    return brain


def clear_ticker_context(brain: Dict, ticker: str) -> Dict:
    """æ¸…é™¤?æ”¯?¡ç¥¨?„æ???Contextï¼ˆé??°ç?ç©¶æ??¨ï???""
    if ticker in brain.get("contexts", {}):
        del brain["contexts"][ticker]
    brain["chat_history"] = []
    return brain
