import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional
from PIL import Image
import re

import gradio as gr
from llama_cpp import Llama

# -------------------------
# Config - adjust if needed
# -------------------------
MODEL_FILENAME = "medgemma-4b-it_Q4_K_M.gguf"
MODEL_PATH = os.path.join(os.getcwd(), "models", MODEL_FILENAME)
LLAMA_THREADS = min(12, os.cpu_count() or 4)

MAX_NEW_TOKENS = 120
GEN_TIMEOUT_SECONDS = 25
TEMPERATURE = 0.35
TOP_P = 0.9
REPEAT_PENALTY = 1.2

SYSTEM_DISCLAIMER = (
    "This chatbot is for educational purposes only and is NOT a substitute for professional medical advice, "
    "diagnosis, or treatment. For urgent or clinical issues, consult a licensed healthcare provider."
)

# -------------------------
# Load model
# -------------------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Put the GGUF file in ./models/")

print("Loading Llama model:", MODEL_PATH)
llm = Llama(model_path=MODEL_PATH, n_threads=LLAMA_THREADS)
print("Model loaded with", LLAMA_THREADS, "threads.")

# -------------------------
# Small FAQ KB
# -------------------------
FAQ_KB = {
    "flu symptoms": "Common flu symptoms: fever, cough, sore throat, body aches, fatigue, runny nose.",
    "exercise frequency": "General adult recommendation: at least 150 minutes/week of moderate aerobic activity.",
    "hydration": "Drink enough water to avoid thirst; commonly ~2-3 L/day depending on activity and climate."
}

# Mock hospital dataset (can expand or load from CSV)
# HOSPITAL_DB = {
#     "city hospital": "Mon–Sat, 9:00 AM – 6:00 PM",
#     "general hospital": "24/7 emergency, OPD Mon–Fri 10:00 AM – 4:00 PM",
#     "healthcare clinic": "Mon–Sat, 8:00 AM – 2:00 PM",
# }
HOSPITAL_DB = {
    "city hospital": {
        "working_hours": "Mon–Sat, 9:00 AM – 6:00 PM",
        "appointments": "Call reception to book: +91-98765-43210 or book online at cityhospital.example.com",
        "contact": "+91-98765-43210"
    },
    "general hospital": {
        "working_hours": "24/7 emergency, OPD Mon–Fri 10:00 AM – 4:00 PM",
        "appointments": "Walk-in OPD for general cases, emergency always open",
        "contact": "+91-91234-56789"
    },
    "healthcare clinic": {
        "working_hours": "Mon–Sat, 8:00 AM – 2:00 PM",
        "appointments": "Call +91-99887-66554 to schedule, no online booking",
        "contact": "+91-99887-66554"
    },
    "hospital": {   # fallback if name not specified
        "working_hours": "Most hospitals operate Mon–Sat, 9:00 AM – 6:00 PM; emergencies often 24/7.",
        "appointments": "Call the hospital reception or visit the official website to book an appointment.",
        "contact": "General helpline: +91-1800-000-000"
    }
}


# def check_hospital_hours(query: str) -> Optional[str]:
#     q_lower = query.lower()
#     for name, hours in HOSPITAL_DB.items():
#         if name in q_lower:
#             return f"{name.title()} working hours: {hours}"
#     return None

def check_hospital_hours(query: str) -> Optional[str]:
    q_lower = query.lower()
    for name, info in HOSPITAL_DB.items():
        if name in q_lower:
            return (
                f"{name.title()} support information:\n"
                f"- Working hours: {info['working_hours']}\n"
                f"- Appointment: {info['appointments']}\n"
                f"- Contact: {info['contact']}\n\n"
                + SYSTEM_DISCLAIMER
            )
    # fallback if query only mentions "hospital" or "clinic"
    if "hospital" in q_lower or "clinic" in q_lower:
        info = HOSPITAL_DB["hospital"]
        return (
            f"General Hospital Support Information:\n"
            f"- Working hours: {info['working_hours']}\n"
            f"- Appointment: {info['appointments']}\n"
            f"- Contact: {info['contact']}\n\n"
            + SYSTEM_DISCLAIMER
        )
    return None


def find_faq_snippet(q: str) -> str:
    ql = (q or "").lower()
    for k, v in FAQ_KB.items():
        if k in ql or any(tok in ql for tok in k.split()):
            return v
    return ""

# -------------------------
# Extraction & cleaning
# -------------------------
def extract_text_from_response(resp) -> str:
    """
    Strict extractor for the text-completion response shape you provided:
    Uses choices[0]['text'] if available, otherwise falls back to string resp.
    """
    if resp is None:
        return ""
    # If dict: prefer choices[0]['text']
    if isinstance(resp, dict):
        try:
            choices = resp.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                c0 = choices[0]
                if isinstance(c0, dict) and "text" in c0 and isinstance(c0["text"], str):
                    return c0["text"].strip()
        except Exception:
            pass
    # If string, return stripped
    if isinstance(resp, str):
        return resp.strip()
    # Fallback
    return str(resp).strip()

def clean_output(text: str) -> str:
    """
    Clean model text:
    - Remove code fences/backticks
    - Remove any leading literal 'text' line that sometimes appears
    - Remove common code-looking lines
    - Collapse consecutive duplicate lines
    - Ensure disclaimer appended once
    """
    if not text:
        return text

    # 1) Normalize newlines and remove code fences/backticks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for token in ["```python", "```", "`python", "`", "```py"]:
        text = text.replace(token, "")

    # 2) Remove a leading literal "text" line, e.g. "text\nA fever..."
    text = re.sub(r'^\s*text\s*\n', '', text, flags=re.IGNORECASE)

    # 3) Split lines and drop code-like lines and empty lines
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        # drop if obviously code / markers
        if (low.startswith("def ")
            or low.startswith("return ")
            or low.startswith("import ")
            or low.startswith("from ")
            or low.startswith("class ")
            or low.startswith("if __")
            or low.startswith("#")
            or low.startswith(">>>")
            or low == "python"):
            continue
        # skip a line that's just the word "text" in the middle
        if s.lower() == "text":
            continue
        # collapse repeated consecutive lines
        if not lines or s != lines[-1]:
            lines.append(s)

    cleaned = "\n".join(lines).strip()

    # 4) Remove any occurrences where cleaned contains a duplicate block prefixed by "text\n"
    #    (defensive): e.g., "Answer...\n\ntext\nAnswer..."
    cleaned = re.sub(r'(?s)(.+?)\n+\s*text\s*\n+\1', r'\1', cleaned, flags=re.IGNORECASE)

    # # 5) Guarantee disclaimer present exactly once at the end
    # if SYSTEM_DISCLAIMER not in cleaned:
    #     cleaned = cleaned.rstrip() + "\n\n" + SYSTEM_DISCLAIMER
    # else:
    #     # ensure single occurrence: remove extras then append one
    #     parts = cleaned.split(SYSTEM_DISCLAIMER)
    #     cleaned_main = parts[0].strip()
    #     cleaned = cleaned_main + "\n\n" + SYSTEM_DISCLAIMER


    # 5) Guarantee disclaimer present exactly once at the end
    # Remove any model-generated "Disclaimer:" prefix to avoid duplication
    cleaned = cleaned.replace("Disclaimer:", "").strip()

    if SYSTEM_DISCLAIMER not in cleaned:
        cleaned = cleaned.rstrip() + "\n\n" + SYSTEM_DISCLAIMER
    else:
        # ensure single occurrence
        parts = cleaned.split(SYSTEM_DISCLAIMER)
        cleaned_main = parts[0].strip()
        cleaned = cleaned_main + "\n\n" + SYSTEM_DISCLAIMER
    
    return cleaned


# -------------------------
# LLM call & runner
# -------------------------
def _llm_call(prompt: str, max_tokens: int):
    return llm(
        prompt,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY
    )

def run_generation(prompt_text: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
    fallback = "⚠️ Sorry, I couldn't generate a response right now. Please try again.\n\n" + SYSTEM_DISCLAIMER
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_llm_call, prompt_text, max_tokens)
            resp = fut.result(timeout=GEN_TIMEOUT_SECONDS)
    except TimeoutError:
        return fallback
    except Exception as e:
        print("Generation error:", e)
        return fallback

    raw_text = extract_text_from_response(resp)
    cleaned = clean_output(raw_text)
    return cleaned or fallback

@lru_cache(maxsize=256)
def cached_generation(prompt_text: str):
    return run_generation(prompt_text, MAX_NEW_TOKENS)

# -------------------------
# Gradio logic
# -------------------------
def answer_query(user_question: str, uploaded_image: Optional[Image.Image], debug_prompt: bool = False):
    user_question = (user_question or "").strip()
    if not user_question:
        return "Please enter a question.", ""

    # 1) Check hospital dataset
    hospital_info = check_hospital_hours(user_question)
    if hospital_info:
        return hospital_info + "\n\n" + SYSTEM_DISCLAIMER, ""

    user_question = (user_question or "").strip()
    if not user_question:
        return "Please enter a question.", ""

    kb = find_faq_snippet(user_question) if 'find_faq_snippet' in globals() else ""
    image_note = ""
    if uploaded_image is not None:
        image_note = "User uploaded an image — describe visible findings cautiously and state uncertainty."

    prompt = f"""System: You are a helpful, cautious medical information assistant.
Always:
- Answer in plain text only (no code, no markdown fences).
- Use 3–5 sentences max.
- Do not repeat yourself.
- If unsure, say: "I may be mistaken — consult a medical professional."
- End with the disclaimer: {SYSTEM_DISCLAIMER}

Knowledge snippet: {kb}

User: {user_question}
{image_note}

Assistant:"""

    if debug_prompt:
        out = cached_generation(prompt)
        return out, prompt

    out = cached_generation(prompt)
    return out, ""

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="Healthcare Assistant — MedGemma Q4 (Final)") as demo:
    gr.Markdown("# Healthcare Assistant — Local GGUF\n> Educational only — not a substitute for medical advice.")
    with gr.Row():
        with gr.Column(scale=2):
            q = gr.Textbox(label="Ask a health question", placeholder="e.g., I have a fever, what should I do?", lines=4)
            img = gr.Image(label="Optional: upload an image (X-ray/photo)", type="pil")
            debug = gr.Checkbox(label="Show prompt (debug)", value=False)
            submit = gr.Button("Ask")
        with gr.Column(scale=2):
            out = gr.Textbox(label="Assistant response", lines=14)
            prompt_view = gr.Textbox(label="Generated prompt (debug)", lines=12, visible=False)

    def on_submit(question, uploaded_img, debug_flag):
        return answer_query(question, uploaded_img, debug_flag)

    submit.click(on_submit, inputs=[q, img, debug], outputs=[out, prompt_view])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
