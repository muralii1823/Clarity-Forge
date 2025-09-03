import os, io, time, json, requests, re
import streamlit as st
import streamlit.components.v1 as components # Import components for custom HTML buttons
import fitz # PyMuPDF
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD2cuwBME2xf8h-7v7Z6-chsQt6LrnNT5k")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
HERO_BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80"

# macOS (Homebrew) defaults; adjust for your machine if needed
try:
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
except Exception:
    pass
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
os.environ["PATH"] += os.pathsep + "/opt/homebrew/sbin"

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND: extraction (PyMuPDF → OCR fallback)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _extract_text_from_pdf(file_bytes: bytes) -> str | None:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(doc.load_page(p).get_text() for p in range(doc.page_count))
        doc.close()
        return text if len(text.strip()) > 50 else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _ocr_pdf(file_bytes: bytes, ocr_language: str) -> str | None:
    try:
        images = convert_from_bytes(file_bytes)
        return "".join(pytesseract.image_to_string(im, lang=ocr_language) for im in images)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _ocr_image(file_bytes: bytes, ocr_language: str) -> str | None:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image, lang=ocr_language)
    except Exception:
        return None

def extract_text_from_document(uploaded_file: io.BytesIO, file_type: str, ocr_language: str) -> str:
    """Combines different extraction methods with a clear fallback logic."""
    try:
        file_bytes = uploaded_file.read()
    except Exception as e:
        st.error("Failed to read uploaded file. Please ensure it is not corrupted."); return ""
    
    # Handle language for OCR
    ocr_lang_code = ocr_language if ocr_language.lower() != "auto" else "eng"

    with st.status(f"Extracting text from {file_type.split('/')[-1].upper()}…") as status_container:
        if file_type == "application/pdf":
            status_container.update(label="Attempting direct text extraction…", state="running")
            text = _extract_text_from_pdf(file_bytes)
            if text:
                status_container.update(label="Direct PDF extraction successful.", state="complete", expanded=False); return text
            
            status_container.update(label=f"Direct extraction failed. Starting OCR (lang: {ocr_lang_code})…", state="running")
            text = _ocr_pdf(file_bytes, ocr_lang_code)
            if text and text.strip():
                status_container.update(label="OCR extraction successful.", state="complete", expanded=False); return text

        elif file_type.startswith("image/"):
            status_container.update(label=f"Starting OCR for image (lang: {ocr_lang_code})…", state="running")
            text = _ocr_image(file_bytes, ocr_lang_code)
            if text and text.strip():
                status_container.update(label="OCR extraction successful.", state="complete", expanded=False); return text

        else:
            status_container.update(label="Unsupported file type.", state="error"); return ""
    
    st.error("Text extraction failed."); return ""

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND: Gemini call
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def call_gemini_api(prompt: str, response_schema: dict | None = None, max_retries: int = 4) -> str | None:
    """Makes a request to the Gemini API with retry logic and caching."""
    if not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY not set. Using mock response."); return json.dumps(
            {"document_title": "Mock Document", "author": "N/A", "date": "N/A",
             "summary": "Mock summary for offline demo.",
             "key_points": ["Works without API key", "Set GEMINI_API_KEY for live results"]},
            ensure_ascii=False, indent=2)

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if response_schema:
        payload["generationConfig"] = {"responseMimeType": "application/json", "responseSchema": response_schema}
    
    with st.status("Connecting to Gemini API...") as status_container:
        for attempt in range(max_retries):
            try:
                url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
                status_container.update(label=f"Attempt {attempt+1}/{max_retries}: Sending request…", state="running")
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
                resp.raise_for_status()
                data = resp.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if parts:
                    status_container.update(label="Gemini API call successful.", state="complete", expanded=False); return parts[0].get("text", "")
                st.error(f"Unexpected API response: {json.dumps(data)[:500]}"); return None
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt; status_container.update(label=f"API error. Retrying in {wait}s… ({e})", state="running"); time.sleep(wait)
                else:
                    st.error("Gemini API failed after retries."); return None
            except Exception as e:
                st.error("Unexpected error calling Gemini."); st.exception(e); return None
    return None

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND: JSON agent + readable summary
# ──────────────────────────────────────────────────────────────────────────────
def _coerce_json(s: str) -> dict | None:
    """Robustly extracts and parses a JSON object from a string."""
    if not s: return None
    json_match = re.search(r"\{.*\}", s.strip(), re.DOTALL)
    if json_match:
        s = json_match.group(0)
    try:
        return json.loads(s)
    except Exception as e:
        st.warning(f"Failed to parse JSON: {e}"); return None

def _truncate_text_by_lines(text: str, max_chars=8000):
    if len(text) <= max_chars: return text
    lines, out, count = text.splitlines(), [], 0
    for ln in lines:
        if count + len(ln) + 1 > max_chars: break
        out.append(ln); count += len(ln) + 1
    return "\n".join(out) + "\n\n…(truncated)"

@st.cache_data(show_spinner=False)
def document_to_json_agent(doc_text: str, doc_language: str) -> dict | None:
    if not doc_text: st.warning("No text for JSON extraction."); return None
    
    schema = {"type":"OBJECT","properties":{
                "document_title":{"type":"STRING"},
                "author":{"type":"STRING"},
                "date":{"type":"STRING"},
                "summary":{"type":"STRING"},
                "key_points":{"type":"ARRAY","items":{"type":"STRING"}}},
              "required":["document_title","summary","key_points"]}
    
    prompt_text = _truncate_text_by_lines(doc_text, 8000)
    prompt = (
        f"You are a highly efficient AI agent. Your task is to extract key information from a document and return a STRICT JSON object.\n"
        f"**Instructions:**\n"
        f"1.  Set `document_title`, `author`, `date`, and `summary`.\n"
        f"2.  For `key_points`, provide a concise list of the most important takeaways.\n"
        f"3.  If any field (like `author` or `date`) is not present in the document, use 'N/A' for its value.\n"
        f"4.  The output must be JSON ONLY, with no extra text or explanations.\n"
        f"**Document Language:** {doc_language}\n"
        f"**Input Document:**\n```\n{prompt_text}\n```\n"
        f"**Strict JSON Output:**"
    )

    raw = call_gemini_api(prompt, response_schema=schema)
    data = _coerce_json(raw)
    if data: return data
    st.error("Failed to parse model output as JSON.")
    if raw: st.code(raw, language="json")
    return None

def json_to_readable(structured: dict) -> str:
    if not structured: return "No data."
    lines = []
    if structured.get("document_title"): lines.append(f"### {structured['document_title']}")
    if structured.get("author"): lines.append(f"**Author:** {structured['author']}")
    if structured.get("date"): lines.append(f"**Date:** {structured['date']}")
    lines += ["\n#### Summary", structured.get("summary","—")]
    if structured.get("key_points"):
        lines.append("\n#### Key Points")
        lines += [f"- {p}" for p in structured["key_points"]]
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# FRONTEND: Apple-style dark UI (same visuals)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CLARITY FORGE — From Chaos to Clarity", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* Root Variables - Dark Theme with Vibrant Accents */
:root {{
    --bg-primary: linear-gradient(135deg, #0F0F23 0%, #1A1A2E 50%, #16213E 100%);
    --bg-card: #1f2937; /* Solid, slightly lighter dark background */
    --bg-card-hover: #2d3a4b; /* Solid hover background */
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --accent-blue: #00D2FF;
    --accent-purple: #8B5CF6;
    --accent-green: #10B981;
    --accent-orange: #F59E0B;
    --border: rgba(255, 255, 255, 0.1);
    --border-hover: rgba(255, 255, 255, 0.2);
    --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    --glow: 0 0 20px rgba(0, 210, 255, 0.15);
}}

/* Global Styles */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body, .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    overflow-x: hidden;
}}

.stApp {{
    background-attachment: fixed;
}}

/* Hide Streamlit Elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
.stDeployButton {{display: none;}}

/* Animated Background Particles */
.particles {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}}

.particle {{
    position: absolute;
    background: var(--accent-blue);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
    opacity: 0.1;
}}

.particle:nth-child(1) {{ left: 20%; animation-delay: 0s; width: 10px; height: 10px; }}
.particle:nth-child(2) {{ left: 60%; animation-delay: 2s; width: 6px; height: 6px; }}
.particle:nth-child(3) {{ left: 80%; animation-delay: 4s; width: 8px; height: 8px; }}

@keyframes float {{
    0%, 100% {{ transform: translateY(0px) rotate(0deg); opacity: 0.1; }}
    50% {{ transform: translateY(-100px) rotate(180deg); opacity: 0.3; }}
}}

/* Hero Section */
.hero-container {{
    position: relative;
    max-width: 1200px;
    margin: 0 auto;
    padding: 80px 20px 60px;
    text-align: center;
    background: url('{HERO_BACKGROUND_IMAGE_URL}') center/cover;
    background-blend-mode: overlay;
    border-radius: 30px;
    margin-top: 40px;
    overflow: hidden;
}}

.hero-container::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(15, 15, 35, 0.95), rgba(26, 26, 46, 0.9));
    z-index: 1;
}}

.hero-content {{
    position: relative;
    z-index: 2;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid var(--accent-purple);
    color: var(--accent-purple);
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 30px;
    animation: glow 2s ease-in-out infinite alternate;
}}

@keyframes glow {{
    from {{ box-shadow: 0 0 10px rgba(139, 92, 246, 0.3); }}
    to {{ box-shadow: 0 0 20px rgba(139, 92, 246, 0.6); }}
}}

.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(3rem, 6vw, 4.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 20px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: slideInUp 1s ease-out;
}}

@keyframes slideInUp {{
    from {{ transform: translateY(50px); opacity: 0; }}
    to {{ transform: translateY(0); opacity: 1; }}
}}

.hero-subtitle {{
    font-size: clamp(1.2rem, 2.5vw, 1.5rem);
    color: var(--text-secondary);
    margin-bottom: 40px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    animation: slideInUp 1s ease-out 0.2s both;
}}

/* Rotating Gear Icon */
.gear-icon {{
    display: inline-block;
    animation: rotate 3s linear infinite;
    margin-left: 10px;
    color: var(--accent-blue);
}}

@keyframes rotate {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}

/* Feature Cards - Extraordinary Edition */
.features-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 40px;
    max-width: 1400px;
    margin: 80px auto;
    padding: 0 20px;
    perspective: 1000px;
}}

.feature-card {{
    background: var(--bg-card); /* Use solid opaque background */
    border: 1px solid var(--border);
    border-radius: 25px;
    padding: 40px 30px;
    transition: all 0.6s cubic-bezier(0.23, 1, 0.320, 1);
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    cursor: pointer;
    transform-style: preserve-3d;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}}

/* Multi-layer animated background */
.feature-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: conic-gradient(from 0deg at 50% 50%,
        rgba(0, 210, 255, 0.1) 0deg,
        rgba(139, 92, 246, 0.1) 120deg,
        rgba(16, 185, 129, 0.1) 240deg,
        rgba(0, 210, 255, 0.1) 360deg);
    opacity: 0;
    transition: opacity 0.6s ease;
    animation: rotate 8s linear infinite;
    border-radius: 25px;
}}

.feature-card::after {{
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg,
        var(--accent-blue),
        var(--accent-purple),
        var(--accent-green),
        var(--accent-blue));
    background-size: 300% 300%;
    border-radius: 27px;
    z-index: -1;
    opacity: 0;
    animation: gradientShift 4s ease infinite;
    transition: opacity 0.6s ease;
}}

@keyframes gradientShift {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}

.feature-card:hover::before {{
    opacity: 1;
}}

.feature-card:hover::after {{
    opacity: 1;
}}

.feature-card:hover {{
    background: rgba(255, 255, 255, 0.08);
    border-color: transparent;
    transform: translateY(-15px) rotateX(5deg) rotateY(5deg);
    box-shadow:
        0 25px 80px rgba(0, 210, 255, 0.2),
        0 0 40px rgba(139, 92, 246, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}}

/* Animated Icon Container */
.feature-icon {{
    position: relative;
    font-size: 3.5rem;
    margin-bottom: 25px;
    color: var(--accent-blue);
    transition: all 0.6s cubic-bezier(0.23, 1, 0.320, 1);
    display: inline-block;
    z-index: 2;
}}

.feature-icon::before {{
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(0, 210, 255, 0.1) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(-50%, -50%) scale(0);
    transition: transform 0.6s cubic-bezier(0.23, 1, 0.320, 1);
    z-index: -1;
}}

.feature-card:hover .feature-icon {{
    transform: scale(1.2) rotateY(180deg);
    color: var(--accent-purple);
    filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.5));
}}

.feature-card:hover .feature-icon::before {{
    transform: translate(-50%, -50%) scale(1.5);
    background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%);
}}

/* Specific icon animations */
.feature-card:nth-child(1) .feature-icon {{
    animation: pulse 3s ease-in-out infinite;
}}

.feature-card:nth-child(2) .feature-icon {{
    animation: swing 2s ease-in-out infinite;
}}

.feature-card:nth-child(3) .feature-icon {{
    animation: bounce 2.5s ease-in-out infinite;
}}

.feature-card:nth-child(4) .feature-icon {{
    animation: flash 1.5s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
}}

@keyframes swing {{
    0%, 100% {{ transform: rotate(0deg); }}
    25% {{ transform: rotate(5deg); }}
    75% {{ transform: rotate(-5deg); }}
}}

@keyframes bounce {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-5px); }}
}}

@keyframes flash {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.8; }}
}}

/* Enhanced Typography */
.feature-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 20px;
    color: var(--text-primary);
    position: relative;
    transition: all 0.6s ease;
    z-index: 2;
}}

.feature-title::after {{
    content: '';
    position: absolute;
    bottom: -5px;
    left: 0;
    width: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    transition: width 0.6s cubic-bezier(0.23, 1, 0.320, 1);
}}

.feature-card:hover .feature-title::after {{
    width: 100%;
}}

.feature-description {{
    color: var(--text-secondary);
    line-height: 1.7;
    font-size: 0.95rem;
    transition: all 0.6s ease;
    z-index: 2;
    position: relative;
}}

.feature-card:hover .feature-description {{
    color: var(--text-primary);
    transform: translateY(-2px);
}}

/* Progress bar animation on hover */
.feature-card .progress-indicator {{
    position: absolute;
    bottom: 0;
    left: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    width: 0%;
    transition: width 1.2s cubic-bezier(0.23, 1, 0.320, 1);
    border-radius: 0 0 25px 25px;
}}

.feature-card:hover .progress-indicator {{
    width: 100%;
}}

/* Floating particles effect */
/* Removed .floating-particles for simplicity */

/* Staggered animation entrance */
.feature-card:nth-child(1) {{ animation: slideInUp 0.8s ease-out 0.1s both; }}
.feature-card:nth-child(2) {{ animation: slideInUp 0.8s ease-out 0.2s both; }}
.feature-card:nth-child(3) {{ animation: slideInUp 0.8s ease-out 0.3s both; }}
.feature-card:nth-child(4) {{ animation: slideInUp 0.8s ease-out 0.4s both; }}

/* Interactive number counter */
.feature-number {{
    position: absolute;
    top: 20px;
    right: 25px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    color: rgba(255, 255, 255, 0.03);
    transition: all 0.6s ease;
    z-index: 1;
}}

.feature-card:hover .feature-number {{
    color: rgba(0, 210, 255, 0.1);
    transform: scale(1.1);
}}

/* Main Content Cards */
.main-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 25px;
    padding: 40px;
    margin: 40px auto;
    max-width: 1200px;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}}

.main-card::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(0, 210, 255, 0.1), transparent);
    animation: spin 6s linear infinite;
    z-index: -1;
}}

@keyframes spin {{
    100% {{ transform: rotate(360deg); }}
}}

/* File Uploader Styling */
[data-testid="stFileUploader"] {{
    background: transparent;
}}

[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed var(--accent-blue) !important;
    background: rgba(0, 210, 255, 0.05) !important;
    border-radius: 20px !important;
    padding: 40px !important;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}

[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--accent-purple) !important;
    background: rgba(139, 92, 246, 0.1) !important;
    transform: scale(1.02);
}}

[data-testid="stFileUploaderDropzone"] > div {{
    color: var(--text-primary) !important;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
    border: none;
    border-radius: 15px;
    padding: 12px 30px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 210, 255, 0.4);
}}

/* Select Boxes */
.stSelectbox > div > div {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 15px;
    color: var(--text-primary);
}}

.stSelectbox > div > div:hover {{
    border-color: var(--accent-blue);
}}

/* Text Areas */
.stTextArea > div > div > textarea {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 15px;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}}

/* Code Blocks */
.stCodeBlock {{
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid var(--border);
    border-radius: 15px;
    backdrop-filter: blur(10px);
}}

/* Success/Warning/Error Messages */
.stSuccess {{
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid var(--accent-green);
    border-radius: 15px;
    color: var(--accent-green);
}}

.stWarning {{
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid var(--accent-orange);
    border-radius: 15px;
    color: var(--accent-orange);
}}

.stError {{
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #EF4444;
    border-radius: 15px;
    color: #EF4444;
}}

.stInfo {{
    background: rgba(0, 210, 255, 0.1);
    border: 1px solid var(--accent-blue);
    border-radius: 15px;
    color: var(--accent-blue);
}}

/* Metrics */
.metric-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    border-color: var(--accent-blue);
    transform: translateY(-2px);
}}

/* Footer */
.footer {{
    text-align: center;
    padding: 40px 20px;
    color: var(--text-muted);
    font-size: 14px;
    border-top: 1px solid var(--border);
    margin-top: 80px;
}}

/* Responsive Design */
@media (max-width: 768px) {{
    .hero-container {{
        padding: 60px 20px 40px;
        margin-top: 20px;
    }}

    .features-grid {{
        grid-template-columns: 1fr;
        gap: 20px;
        margin: 40px auto;
    }}

    .main-card {{
        margin: 20px;
        padding: 25px;
    }}
}}

/* Scroll-triggered animations */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.fade-in-up {{
    animation: fadeInUp 0.8s ease-out;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{
    width: 8px;
}}

::-webkit-scrollbar-track {{
    background: rgba(255, 255, 255, 0.05);
}}

::-webkit-scrollbar-thumb {{
    background: var(--accent-blue);
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: var(--accent-purple);
}}
</style>

<div class="particles">
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>


<div class="feature-card fade-in-up">
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-badge">
           <h1>🚀 CLARITY FORGE — From Chaos to Clarity </h1>
        </div>
        <h1 class="main-tagline">
             Unstructured → JSON + Summary
        </h1>
        <p class="hero-description">
            Transform complex documents into actionable, structured data instantly.
        </p>
    </div>
</div>
</div>

<div class="features-grid">
    <div class="feature-card fade-in-up">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI-Powered Extraction</div>
        <div class="feature-description">
            Advanced machine learning algorithms extract meaningful data from any document format with unprecedented accuracy.
        </div>
    </div>
    <div class="feature-card fade-in-up">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">OCR Technology</div>
        <div class="feature-description">
            State-of-the-art optical character recognition processes scanned documents and images with multi-language support.
        </div>
    </div>
    <div class="feature-card fade-in-up">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Structured Output</div>
        <div class="feature-description">
            Clean, organized JSON data with intelligent summarization and key point extraction for immediate use.
        </div>
    </div>
    <div class="feature-card fade-in-up">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Lightning Fast</div>
        <div class="feature-description">
            Optimized processing pipeline delivers results in seconds, not minutes. Built for enterprise-scale performance.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Application Card
st.markdown('<div class="main-card fade-in-up">', unsafe_allow_html=True)

ocr_language_options = {"English":"eng","Spanish":"spa","French":"fra","Tamil":"tam","Auto-detect":"auto"}
colA, colB = st.columns([2,1])
with colA:
    selected_language_name = st.selectbox("Select document language for OCR", list(ocr_language_options.keys()), index=0)
with colB:
    st.write(""); st.markdown('<div class="small">Max 200MB · PDF, JPG · PNG</div>', unsafe_allow_html=True)

ocr_language_code = ocr_language_options[selected_language_name]
uploaded_file = st.file_uploader("Choose a PDF or Image file", type=["pdf","jpg","jpeg","png"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    extracted_text = extract_text_from_document(uploaded_file, uploaded_file.type, ocr_language_code)

    if extracted_text:
        with st.spinner("Analyzing with LLM and generating JSON…"):
            structured_data = document_to_json_agent(extracted_text, selected_language_name)

        if structured_data:
            st.markdown("#### Extraction Stats")
            approx_tokens = max(1, len(extracted_text)//4)
            st.write({"Type": uploaded_file.type, "Characters": len(extracted_text), "ApproxTokens": approx_tokens, "Language": selected_language_name})
            st.markdown("#### Raw Extracted Text")
            st.text_area("Raw Text", extracted_text, height=260)

            st.markdown("#### Structured JSON")
            pretty = json.dumps(structured_data, indent=2, ensure_ascii=False)
            st.code(pretty, language="json")

            st.markdown("#### Summary")
            summary_md = json_to_readable(structured_data)
            st.markdown(summary_md)
        else:
            st.error("Failed to generate structured JSON.")
    else:
        st.error("Failed to extract readable text.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>🚀 Built with <strong>Streamlit</strong> • <strong>PyMuPDF</strong> • <strong>Tesseract OCR</strong> • <strong>Google Gemini AI</strong></p>
    <p style="margin-top: 10px; opacity: 0.7;">Transforming unstructured data into actionable insights with cutting-edge AI technology.</p>
</div>
""", unsafe_allow_html=True)
