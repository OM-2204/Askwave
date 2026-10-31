"""
AI Video Assistant: Streamlit front end.

Requires Streamlit >= 1.40  (pip install -U streamlit)
Optional but recommended: put config.toml in ./.streamlit/ so native widgets
(dropdown menus, toasts, chat input) match the theme.

The backend contract is unchanged:
    process_input(source) -> chunks
    transcribe_all(chunks, language) -> transcript
    generate_title / summarize / extract_* (transcript)
    build_rag_chain(transcript) -> chain
    ask_question(chain, question) -> answer
"""
from __future__ import annotations

import html
import logging
import math
import os
import re
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()
log = logging.getLogger("ai_video_assistant")

# ─── Page config (must be the first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)


def _streamlit_is_new_enough(minimum=(1, 40)) -> bool:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2]) >= minimum
    except ValueError:
        return True


if not _streamlit_is_new_enough():
    st.error(f"This interface needs Streamlit 1.40 or newer (you have {st.__version__}). "
             "Run `pip install -U streamlit` and restart the app.")
    st.stop()


# ─── Content constants ──────────────────────────────────────────────────────────
PIPELINE = [
    ("audio",      "Preparing audio",             "Fetching the recording and splitting it into segments"),
    ("transcript", "Transcribing speech",         "Turning the audio into text"),
    ("title",      "Naming the session",          "Writing a short title"),
    ("summary",    "Writing the summary",         "Condensing the discussion"),
    ("extract",    "Pulling out the follow-ups",  "Action items, decisions and open questions"),
    ("rag",        "Getting ready for questions", "Indexing the transcript so you can chat with it"),
]
STEP_NAMES = {k: n for k, n, _ in PIPELINE}

SUGGESTED_QUESTIONS = [
    "What were the key decisions?",
    "Who is responsible for what?",
    "Summarise this in three bullet points",
    "What was left unresolved?",
]

UPLOAD_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a", "aac", "ogg", "flac"]

HERO_STEPS = [
    ("Add a recording",    "Paste a YouTube link or upload a file in the sidebar."),
    ("Wait for the brief", "Speech is transcribed, then summarised and indexed for questions."),
    ("Read it or ask it",  "Skim the summary and follow-ups, or ask about anything that was said."),
]


# ─── Styles ─────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Instrument+Sans:wght@400..700&display=swap');

:root {
    --bg: #141630;
    --surface: #1B1E3F;
    --raised: #242850;
    --line: #343968;
    --line-strong: #4A5090;
    --text: #EEEDFA;
    --muted: #A3A6CC;
    --muted-2: #8D91BE;
    --accent: #6C5CF2;
    --accent-hover: #7B6CFF;
    --periwinkle: #A99DFF;
    --amber: #FFC15E;
    --mint: #6FE3B5;
    --rose: #FF9DB1;
    --font-display: 'Bricolage Grotesque', 'Instrument Sans', system-ui, sans-serif;
    --font-body: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    color-scheme: dark;
}

/* ── Base ── */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text);
}
html, body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, label, input, textarea, button,
[data-baseweb], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {
    font-family: var(--font-body) !important;
}
h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1180px !important;
    padding: 3rem 2.5rem 5rem !important;
}
hr { border-color: var(--line) !important; }
[data-testid="stCaptionContainer"] { color: var(--muted-2) !important; }
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 0.85rem !important; font-weight: 500; }
:focus-visible { outline: 2px solid var(--amber) !important; outline-offset: 2px; }
input:focus-visible, textarea:focus-visible { outline: none !important; }
mark {
    background: rgba(255,193,94,0.26); color: var(--text);
    border-radius: 4px; padding: 0 0.15em; box-shadow: inset 0 -2px 0 var(--amber);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.5rem; }
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> .stButton),
[data-testid="stSidebar"] .stButton, [data-testid="stSidebar"] .stButton > button { width: 100% !important; }

.brand { display: flex; align-items: center; gap: 0.75rem; margin: 0.25rem 0 1.6rem; }
.brand-mark {
    width: 40px; height: 40px; border-radius: 12px; background: var(--accent);
    display: flex; align-items: center; justify-content: center; gap: 2.5px; flex-shrink: 0;
}
.brand-mark i { display: block; width: 3px; border-radius: 2px; background: #fff; }
.brand-name { font-family: var(--font-display); font-weight: 700; font-size: 1.1rem; line-height: 1.1; letter-spacing: -0.01em; }
.brand-sub { font-size: 0.78rem; color: var(--muted-2); margin-top: 3px; }

.or-rule { display: flex; align-items: center; gap: 0.75rem; color: var(--muted-2); font-size: 0.8rem; margin: 0.2rem 0 0.1rem; }
.or-rule::before, .or-rule::after { content: ""; flex: 1; height: 1px; background: var(--line); }

/* ── Inputs ── */
[data-baseweb="input"], [data-baseweb="select"] > div {
    background: var(--raised) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    transition: border-color .15s, box-shadow .15s;
}
[data-baseweb="input"] > div, [data-baseweb="base-input"] { background: transparent !important; border: none !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
    border-color: var(--periwinkle) !important;
    box-shadow: 0 0 0 3px rgba(169,157,255,0.22) !important;
}
input, textarea { color: var(--text) !important; -webkit-text-fill-color: var(--text); }
input::placeholder, textarea::placeholder { color: var(--muted-2) !important; -webkit-text-fill-color: var(--muted-2); opacity: 1; }
[data-baseweb="select"] > div { color: var(--text) !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] { background: var(--raised) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: var(--line) !important; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--raised) !important;
    border: 1px dashed var(--line-strong) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; }

/* ── Buttons ── */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    transition: background .15s, border-color .15s, transform .1s;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover) !important; border-color: var(--accent-hover) !important;
}
button[kind="primary"]:active, button[data-testid="stBaseButton-primary"]:active { transform: translateY(1px); }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: 1px solid var(--line-strong) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: background .15s, border-color .15s;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--raised) !important; border-color: var(--periwinkle) !important;
}
[data-testid="stElementContainer"]:has(> .stDownloadButton), .stDownloadButton, .stDownloadButton > button { width: 100% !important; }
[class*="st-key-sugg_"], [class*="st-key-sugg_"] .stButton, [class*="st-key-sugg_"] button { width: 100% !important; }
[class*="st-key-sugg_"] button { padding: 0.7rem 0.9rem !important; }
[class*="st-key-sugg_"] button, [class*="st-key-sugg_"] button * { justify-content: flex-start !important; text-align: left !important; }

/* ── Tabs (BaseWeb in older Streamlit, React Aria in newer) ── */
[data-baseweb="tab-list"], [role="tablist"] { gap: 0.25rem; border-bottom: 1px solid var(--line) !important; }
button[data-baseweb="tab"], [data-testid="stTab"] { background: transparent !important; padding: 0.7rem 1.1rem !important; }
button[data-baseweb="tab"] p, [data-testid="stTab"] p { color: var(--muted) !important; font-weight: 500; font-size: 1rem !important; }
button[data-baseweb="tab"]:hover p, [data-testid="stTab"]:hover p { color: var(--text) !important; }
button[data-baseweb="tab"][aria-selected="true"] p, [data-testid="stTab"][aria-selected="true"] p { color: var(--text) !important; }
[data-baseweb="tab-highlight"], .react-aria-SelectionIndicator { background: var(--periwinkle) !important; height: 2px !important; }
[data-baseweb="tab-border"] { background: transparent !important; }

/* ── Alerts, expanders, spinner ── */
[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid var(--line) !important; }
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 12px !important; background: var(--surface); }
[data-testid="stExpander"] summary { color: var(--text); }
[data-testid="stSidebar"] [data-testid="stExpander"] { background: transparent; }
[data-testid="stMain"] [data-testid="stExpander"] { max-width: 680px; }
[class*="st-key-clear_chat"] { margin-left: auto; }
.stSpinner, .stSpinner * { color: var(--muted) !important; }

/* ── Hero (empty state) ── */
.hero { max-width: 880px; padding-top: 0.5rem; }
.stMarkdown h1.hero-h {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: clamp(2.4rem, 5vw, 4.1rem) !important;
    line-height: 1.02 !important;
    letter-spacing: -0.035em !important;
    margin: 0 0 1.25rem !important;
    padding: 0 !important;
    max-width: 15em;
    text-wrap: balance;
}
.stMarkdown p.hero-p { color: var(--muted) !important; font-size: 1.1rem; line-height: 1.65; max-width: 56ch; margin: 0 0 2.25rem; }

.s2t {
    display: grid; grid-template-columns: 1.15fr auto 1fr; align-items: center; gap: 1.5rem;
    max-width: 760px; padding: 1.5rem 1.75rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 20px;
}
.s2t-wave { display: flex; align-items: center; gap: 3px; height: 112px; }
.s2t-wave i {
    flex: 1; min-width: 2px; border-radius: 2px; background: var(--periwinkle);
    transform-origin: center; animation: grow .9s cubic-bezier(.2,.7,.2,1) both;
}
.s2t-arrow { width: 38px; height: 2px; background: var(--line-strong); position: relative; }
.s2t-arrow::after {
    content: ""; position: absolute; right: 0; top: -4px; width: 8px; height: 8px;
    border: solid var(--line-strong); border-width: 2px 2px 0 0; transform: rotate(45deg);
}
.s2t-text { display: flex; flex-direction: column; gap: 8px; }
.s2t-text i {
    display: block; height: 8px; border-radius: 4px; background: var(--line-strong);
    transform-origin: left; animation: wipe .55s ease-out both;
}
.s2t-text i.spk { background: var(--amber); }
.s2t-text i.spk:not(:first-child) { margin-top: 7px; }

ol.how {
    list-style: none !important; margin: 2.75rem 0 0 !important; padding: 0 !important;
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.75rem;
}
ol.how li { border-top: 1px solid var(--line-strong); padding-top: 0.9rem; margin: 0 !important; }
.how-n { display: block; font-family: var(--font-display); font-weight: 700; color: var(--amber); font-size: 0.95rem; margin-bottom: 0.35rem; }
.how-t { display: block; font-weight: 600; font-size: 1.02rem; margin-bottom: 0.25rem; }
.how-d { display: block; color: var(--muted); font-size: 0.9rem; line-height: 1.55; }
.stMarkdown p.hero-note { color: var(--muted-2) !important; font-size: 0.85rem; margin-top: 2rem; }

/* ── Live progress ── */
.rail { max-width: 680px; background: var(--surface); border: 1px solid var(--line); border-radius: 20px; padding: 1.5rem 1.75rem 0.9rem; }
.rail-top { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin-bottom: 0.9rem; }
.rail-title { font-family: var(--font-display); font-weight: 650; font-size: 1.3rem; letter-spacing: -0.01em; }
.rail-src { color: var(--muted-2); font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 50%; }
.rail-bar { height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
.rail-bar span { display: block; height: 100%; background: var(--periwinkle); border-radius: 2px; transition: width .4s ease; }
.rail-hint { font-size: 0.82rem; color: var(--muted-2); margin: 0.7rem 0 0.4rem; }
ul.rail-list { list-style: none !important; margin: 0 !important; padding: 0 !important; }
.rs {
    display: grid; grid-template-columns: 24px 1fr auto; gap: 0.85rem; align-items: start;
    padding: 0.85rem 0 !important; margin: 0 !important; border-top: 1px solid var(--line);
}
.rs:first-child { border-top: none; }
.rs-ico { width: 22px; height: 22px; box-sizing: border-box; border-radius: 50%; border: 2px solid var(--line-strong); position: relative; margin-top: 1px; }
.rs-name { display: block; font-weight: 600; font-size: 0.98rem; }
.rs-desc { display: block; font-size: 0.84rem; color: var(--muted-2); margin-top: 2px; }
.rs-time { font-size: 0.82rem; color: var(--muted); font-variant-numeric: tabular-nums; padding-top: 2px; }
.rs-pending { opacity: 0.5; }
.rs-active .rs-ico { border-top-color: var(--amber); animation: spin .8s linear infinite; }
.rs-active .rs-time { color: var(--amber); }
.rs-done .rs-ico { background: var(--mint); border-color: var(--mint); }
.rs-done .rs-ico::after {
    content: ""; position: absolute; left: 5px; top: 1px; width: 5px; height: 10px;
    border: solid var(--bg); border-width: 0 2px 2px 0; transform: rotate(45deg);
}
.rs-error .rs-ico { background: var(--rose); border-color: var(--rose); }
.rs-error .rs-ico::after { content: "!"; position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.8rem; color: var(--bg); }
.rs-error .rs-time { color: var(--rose); }
.rail-compact .rs { padding: 0.5rem 0 !important; }
.rail-compact .rs-desc { display: none; }

.notice {
    max-width: 680px; padding: 0.85rem 1.1rem; margin: 0 0 1.25rem; border-radius: 12px;
    font-size: 0.95rem; line-height: 1.55; overflow-wrap: anywhere;
}
.notice-warn { border: 1px solid rgba(255,193,94,0.45); background: rgba(255,193,94,0.09); }
.notice-error { border: 1px solid rgba(255,157,177,0.45); background: rgba(255,157,177,0.08); margin: 0.5rem 0; }
.fail {
    max-width: 680px; margin: 1rem 0 0.75rem; padding: 1rem 1.25rem; border-radius: 14px;
    border: 1px solid rgba(255,157,177,0.45); background: rgba(255,157,177,0.08);
}
.fail-title { font-weight: 600; color: var(--rose); margin-bottom: 0.3rem; }
.fail-text { color: var(--text); font-size: 0.92rem; line-height: 1.6; overflow-wrap: anywhere; }

/* ── Results ── */
.brief-head { margin: 0 0 0.5rem; }
.stMarkdown h1.brief-title {
    font-family: var(--font-display) !important; font-weight: 700 !important;
    font-size: clamp(1.9rem, 3.2vw, 2.8rem) !important; line-height: 1.08 !important;
    letter-spacing: -0.025em !important; margin: 0 0 1rem !important; padding: 0 !important;
    max-width: 30ch; text-wrap: balance;
}
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
    padding: 0.28rem 0.75rem; border-radius: 999px; background: var(--surface);
    border: 1px solid var(--line); color: var(--muted); font-size: 0.82rem; font-weight: 500;
    font-variant-numeric: tabular-nums;
}
.stMarkdown h2.read-h {
    font-family: var(--font-display) !important; font-weight: 650 !important; font-size: 1.3rem !important;
    letter-spacing: -0.01em !important; margin: 0 0 0.6rem !important; padding: 0 !important;
}
.st-key-read-summary { padding: 0.75rem 1rem 0 0; }
.st-key-read-summary p, .st-key-read-summary li { font-size: 1.04rem !important; line-height: 1.75 !important; max-width: 68ch; }
.stMarkdown li { margin: 0.3rem 0; }
.stMarkdown li::marker { color: var(--muted-2); }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { padding-top: 0.4rem; }
.stMarkdown h3 { font-size: 1.1rem; }

[class*="st-key-card-"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.4rem 1.5rem; gap: 0.5rem;
}
.sec-head { display: flex; align-items: center; gap: 0.6rem; margin: 0.35rem 0 0.5rem; }
.sec-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.sec-title { font-family: var(--font-display); font-weight: 650; font-size: 1.05rem; }
.sec-count { margin-left: auto; font-size: 0.78rem; color: var(--muted); background: var(--raised); border-radius: 999px; padding: 0.1rem 0.6rem; font-variant-numeric: tabular-nums; }
.sec-empty { color: var(--muted-2); font-size: 0.9rem; }
.sec-rule { height: 1px; background: var(--line); margin: 0.9rem 0 0.6rem; }
[class*="st-key-card-"] p, [class*="st-key-card-"] li { font-size: 0.95rem; line-height: 1.65; }

/* ── Transcript ── */
.tx-meta { font-size: 0.85rem; color: var(--muted); margin: 0.25rem 0 0.6rem; min-height: 1.3em; }
.tx {
    background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
    padding: 1.5rem 1.75rem 0.5rem; max-height: 560px; overflow-y: auto;
}
.tx-spk { color: var(--amber); font-weight: 600; }
.tx p.tx-p { max-width: 72ch; margin: 0 0 1.15rem; font-size: 0.98rem; line-height: 1.85; color: var(--text); }

/* ── Chat ── */
.chat-welcome { padding: 0.75rem 0.25rem 0.25rem; }
.chat-welcome-title { font-family: var(--font-display); font-weight: 650; font-size: 1.3rem; letter-spacing: -0.01em; margin-bottom: 0.3rem; }
.chat-welcome-text { color: var(--muted); font-size: 0.95rem; margin-bottom: 0.5rem; }
[data-testid="stChatMessage"] { background: transparent !important; padding: 0.75rem 0.5rem !important; gap: 0.9rem; border-radius: 14px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--raised) !important; }
[data-testid="stChatMessageAvatarUser"] { background: var(--accent) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--amber) !important; color: var(--bg) !important; }
[data-testid="stChatInput"] { border-radius: 14px !important; }
[data-testid="stChatInput"] > div { background: var(--raised) !important; border: 1px solid var(--line) !important; border-radius: 14px !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; }

/* ── Motion ── */
@keyframes grow { from { transform: scaleY(.15); opacity: 0; } to { transform: scaleY(1); opacity: 1; } }
@keyframes wipe { from { transform: scaleX(0); opacity: 0; } to { transform: scaleX(1); opacity: 1; } }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ── Small screens ── */
@media (max-width: 820px) {
    .block-container, [data-testid="stMainBlockContainer"] { padding: 2rem 1.25rem 4rem !important; }
    ol.how { grid-template-columns: 1fr; }
    .s2t { grid-template-columns: 1fr; gap: 1rem; }
    .s2t-arrow { transform: rotate(90deg); margin-left: 6px; }
    .rail-src { display: none; }
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--line-strong); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── Session state ──────────────────────────────────────────────────────────────
for _key, _default in {
    "result": None,
    "chat_history": [],
    "steps": {},
    "times": {},
    "just_finished": False,
    "pending_q": None,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─── Helpers ────────────────────────────────────────────────────────────────────
def H(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {v}" for v in value)
    return str(value).strip()


def md(text: str) -> None:
    """Render model-written markdown. Dollar signs are escaped so '$5M and $10M' isn't read as maths."""
    st.markdown(text.replace("$", r"\$"))


def fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"


def looks_like_url(value: str) -> bool:
    return bool(re.match(r"^(https?://|www\.)", value, re.I) or re.search(r"youtu(\.be|be\.com)", value, re.I))


def source_label(source: str) -> str:
    if looks_like_url(source):
        parsed = urlparse(source if "://" in source else f"https://{source}")
        return (parsed.netloc or source).removeprefix("www.")
    return os.path.basename(source.rstrip("/\\")) or source


def clean_title(raw) -> str:
    text = as_text(raw)
    first = text.splitlines()[0] if text else ""
    first = re.sub(r"^[#*\s\"'`“”‘’]+|[#*\s\"'`“”‘’]+$", "", first)
    return first or "Untitled session"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "meeting-notes"


_ITEM = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+\S", re.M)


def count_items(text: str) -> int:
    return len(_ITEM.findall(text or ""))


def paragraphise(text: str) -> list[str]:
    """Break a transcript into readable paragraphs, whatever shape the transcriber returns."""
    text = text.strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    if len(sentences) > 1:
        return [" ".join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]
    words = text.split()
    return [" ".join(words[i:i + 70]) for i in range(0, len(words), 70)]


def highlight(paragraph: str, query: str) -> str:
    if not query:
        return esc(paragraph)
    parts = re.split(f"({re.escape(query)})", paragraph, flags=re.I)
    return "".join(f"<mark>{esc(p)}</mark>" if i % 2 else esc(p) for i, p in enumerate(parts))


_SPEAKER = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_ .'\-]{0,28}):\s+(.+)$", re.S)


def transcript_paragraph(paragraph: str, query: str) -> str:
    """One transcript paragraph as HTML: speaker label (if any) tinted, search hits highlighted."""
    m = _SPEAKER.match(paragraph)
    if m:
        return f'<span class="tx-spk">{highlight(m.group(1), query)}:</span> {highlight(m.group(2), query)}'
    return highlight(paragraph, query)


def build_notes_markdown(r: dict) -> str:
    def block(name: str, text: str) -> str:
        return f"## {name}\n\n{text or 'Nothing recorded.'}\n"
    return "\n".join([
        f"# {r['title']}\n",
        f"Source: {r['label']}  \nLanguage: {r['language'].title()}\n",
        block("Summary", as_text(r["summary"])),
        block("Action items", as_text(r["action_items"])),
        block("Key decisions", as_text(r["key_decisions"])),
        block("Open questions", as_text(r["open_questions"])),
    ])


def save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def validate_source(text: str, uploaded) -> str | None:
    if uploaded is not None:
        return None
    value = text.strip()
    if not value:
        return "Add a YouTube link, a file path, or upload a recording first."
    if not looks_like_url(value) and not Path(value).expanduser().exists():
        return "That isn't a YouTube link, and no file exists at that path."
    return None


# ─── HTML builders ──────────────────────────────────────────────────────────────
BRAND_HTML = H("""
<div class="brand">
    <div class="brand-mark" aria-hidden="true">
        <i style="height:9px"></i><i style="height:18px"></i><i style="height:25px"></i><i style="height:14px"></i><i style="height:20px"></i>
    </div>
    <div>
        <div class="brand-name">AI Video Assistant</div>
        <div class="brand-sub">Meeting notes you can question</div>
    </div>
</div>
""")


def hero_html() -> str:
    n = 38
    bars = []
    for i in range(n):
        envelope = math.sin(math.pi * (i + 0.5) / n) ** 0.8
        detail = 0.45 + 0.55 * abs(math.sin(i * 1.9) * math.cos(i * 0.65))
        height = max(10, int(100 * envelope * detail))
        bars.append(f'<i style="height:{height}%;animation-delay:{i * 16}ms"></i>')

    shape = [("spk", 26), ("", 100), ("", 84), ("spk", 22), ("", 94), ("", 60), ("spk", 30), ("", 88)]
    lines = [
        f'<i class="{cls}" style="width:{w}%;animation-delay:{800 + k * 90}ms"></i>'
        for k, (cls, w) in enumerate(shape)
    ]
    steps = "".join(
        f'<li><span class="how-n">{i}</span><span class="how-t">{t}</span><span class="how-d">{d}</span></li>'
        for i, (t, d) in enumerate(HERO_STEPS, start=1)
    )
    return H(f"""
    <div class="hero">
        <h1 class="hero-h">Every meeting, written up and ready for your questions.</h1>
        <p class="hero-p">Add a YouTube link or a recording. You get the transcript, a summary, action items and
        decisions, plus a chat that answers from what was actually said.</p>
        <div class="s2t" aria-hidden="true">
            <div class="s2t-wave">{''.join(bars)}</div>
            <div class="s2t-arrow"></div>
            <div class="s2t-text">{''.join(lines)}</div>
        </div>
        <ol class="how">{steps}</ol>
        <p class="hero-note">Works with English and Hinglish speech.</p>
    </div>
    """)


def rail_html(steps: dict, times: dict, label: str = "", compact: bool = False) -> str:
    done = sum(1 for k, _, _ in PIPELINE if steps.get(k) == "done")
    pct = int(done / len(PIPELINE) * 100)
    failed = any(v == "error" for v in steps.values())
    heading = "Analysis stopped" if failed else "Analysing your recording"
    hint = "" if failed else '<div class="rail-hint">Keep this tab open until it finishes.</div>'  # Fixed missing closing quote here[cite: 1]
    right_text = {"active": "Working", "error": "Stopped"}
    items = []
    for key, name, desc in PIPELINE:
        state = steps.get(key, "pending")
        right = fmt_dur(times.get(key, 0)) if state == "done" else right_text.get(state, "")
        items.append(
            f'<li class="rs rs-{state}"><span class="rs-ico"></span>'
            f'<span><span class="rs-name">{name}</span><span class="rs-desc">{desc}</span></span>'
            f'<span class="rs-time">{right}</span></li>'
        )
    if compact:
        return H(f'<div class="rail-compact"><ul class="rail-list">{"".join(items)}</ul></div>')
    return H(f"""
    <div class="rail" role="status" aria-live="polite">
        <div class="rail-top">
            <div class="rail-title">{heading}</div>
            <div class="rail-src" title="{esc(label)}">{esc(label)}</div>
        </div>
        <div class="rail-bar"><span style="width:{pct}%"></span></div>
        {hint}
        <ul class="rail-list">{''.join(items)}</ul>
    </div>
    """)


def notice_html(kind: str, text: str) -> str:
    return f'<div class="notice notice-{kind}" role="alert">{esc(text)}</div>'


def chip(text: str) -> str:
    return f'<span class="chip">{esc(text)}</span>'


def section_head(color: str, title: str, count: int) -> str:
    badge = f'<span class="sec-count">{count}</span>' if count else ""
    return H(f'<div class="sec-head"><span class="sec-dot" style="background:var(--{color})"></span>'
             f'<span class="sec-title">{title}</span>{badge}</div>')


# ─── Pipeline ───────────────────────────────────────────────────────────────────
class PipelineError(Exception):
    def __init__(self, step: str, original: Exception):
        super().__init__(str(original))
        self.step = step
        self.original = original


def _extract_all(transcript):
    return (
        extract_action_items(transcript),
        extract_key_decisions(transcript),
        extract_questions(transcript),
    )


def run_pipeline(source: str, language: str, label: str, stage) -> dict:
    steps = {k: "pending" for k, _, _ in PIPELINE}
    times: dict[str, float] = {}
    started_at = time.perf_counter()

    def paint():
        st.session_state.steps = dict(steps)
        st.session_state.times = dict(times)
        stage.markdown(rail_html(steps, times, label), unsafe_allow_html=True)

    def step(key, fn, *args):
        steps[key] = "active"
        paint()
        t0 = time.perf_counter()
        try:
            out = fn(*args)
        except Exception as exc:
            steps[key] = "error"
            paint()
            raise PipelineError(key, exc) from exc
        steps[key] = "done"
        times[key] = time.perf_counter() - t0
        paint()
        return out

    paint()
    chunks = step("audio", process_input, source)
    transcript = step("transcript", transcribe_all, chunks, language)
    title = step("title", generate_title, transcript)
    summary = step("summary", summarize, transcript)
    actions, decisions, questions = step("extract", _extract_all, transcript)
    rag_chain = step("rag", build_rag_chain, transcript)

    return {
        "title": clean_title(title),
        "transcript": as_text(transcript),
        "summary": as_text(summary),
        "action_items": as_text(actions),
        "key_decisions": as_text(decisions),
        "open_questions": as_text(questions),
        "rag_chain": rag_chain,
        "label": label,
        "language": language,
        "elapsed": time.perf_counter() - started_at,
    }


def analyse(source_text: str, uploaded, language: str) -> None:
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.steps = {}
    st.session_state.times = {}

    tmp_path = None
    if uploaded is not None:
        tmp_path = save_upload(uploaded)
        source, label = tmp_path, uploaded.name
    else:
        source = source_text.strip()
        label = source_label(source)

    stage = st.empty()
    try:
        result = run_pipeline(source, language, label, stage)
    except PipelineError as err:
        log.exception("Pipeline failed at step %s", err.step)
        detail = esc(str(err.original) or err.original.__class__.__name__)[:400]
        st.markdown(H(f"""
        <div class="fail">
            <div class="fail-title">Stopped while {STEP_NAMES[err.step].lower()}</div>
            <div class="fail-text">{detail}<br>Check the link or file and run it again. The technical details below show the full error.</div>
        </div>
        """), unsafe_allow_html=True)
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(type(err.original), err.original, err.original.__traceback__)), language="text")
        st.stop()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    st.session_state.result = result
    st.session_state.just_finished = True
    st.rerun()


# ─── Chat callbacks ─────────────────────────────────────────────────────────────
def _ask_suggested(question: str) -> None:
    st.session_state.pending_q = question


def _clear_chat() -> None:
    st.session_state.chat_history = []


# ─── Views ──────────────────────────────────────────────────────────────────────
def render_results(r: dict) -> None:
    if st.session_state.pop("just_finished", False):
        st.toast("Your brief is ready.", icon=":material/check_circle:")

    transcript = r["transcript"]
    words = len(transcript.split())
    read_minutes = max(1, round(words / 200))

    left, right = st.columns([5, 2], gap="large", vertical_alignment="bottom")
    with left:
        chips = "".join([
            chip(r["label"]),
            chip(f"{words:,} words"),
            chip(f"About {read_minutes} min read"),
            chip(r["language"].title()),
            chip(f"Analysed in {fmt_dur(r['elapsed'])}"),
        ])
        st.markdown(H(f'<div class="brief-head"><h1 class="brief-title">{esc(r["title"])}</h1>'
                      f'<div class="chips">{chips}</div></div>'), unsafe_allow_html=True)
    with right:
        st.download_button(
            "Download notes", data=build_notes_markdown(r), file_name=f"{slugify(r['title'])}.md",
            mime="text/markdown", icon=":material/download:", key="dl_notes",
        )

    tab_overview, tab_transcript, tab_chat = st.tabs(["Overview", "Transcript", "Ask questions"])

    # ── Overview ────────────────────────────────────────────────────────────────
    with tab_overview:
        col_summary, col_follow = st.columns([3, 2], gap="large")
        with col_summary:
            with st.container(key="read-summary"):
                st.markdown('<h2 class="read-h">Summary</h2>', unsafe_allow_html=True)
                md(r["summary"] or "No summary was generated.")
        with col_follow:
            with st.container(key="card-extract"):
                sections = [
                    ("mint",  "Action items",   r["action_items"]),
                    ("amber", "Key decisions",  r["key_decisions"]),
                    ("rose",  "Open questions", r["open_questions"]),
                ]
                for i, (color, title, body) in enumerate(sections):
                    st.markdown(section_head(color, title, count_items(body)), unsafe_allow_html=True)
                    if body:
                        md(body)
                    else:
                        st.markdown('<div class="sec-empty">Nothing found in this recording.</div>', unsafe_allow_html=True)
                    if i < len(sections) - 1:
                        st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    # ── Transcript ──────────────────────────────────────────────────────────────
    with tab_transcript:
        search_col, dl_col = st.columns([4, 1.4], gap="medium", vertical_alignment="center")
        with search_col:
            query = st.text_input("Search the transcript", placeholder="Search for a word or phrase",
                                  label_visibility="collapsed", key="tx_query").strip()
        with dl_col:
            st.download_button("Download .txt", data=transcript, file_name=f"{slugify(r['title'])}-transcript.txt",
                               mime="text/plain", icon=":material/download:", key="dl_tx")

        paragraphs = paragraphise(transcript)
        if len(query) >= 2:
            matches = sum(len(re.findall(re.escape(query), p, flags=re.I)) for p in paragraphs)
            meta = (f"{matches} {'match' if matches == 1 else 'matches'} for “{esc(query)}”"
                    if matches else f"No matches for “{esc(query)}”")
            active_query = query
        else:
            meta, active_query = "&nbsp;", ""
        body = "".join(f'<p class="tx-p">{transcript_paragraph(p, active_query)}</p>' for p in paragraphs)
        st.markdown(H(f'<div class="tx-meta">{meta}</div>') + f'\n<div class="tx">{body}</div>',
                    unsafe_allow_html=True)

    # ── Chat ────────────────────────────────────────────────────────────────────
    with tab_chat:
        history = st.session_state.chat_history
        pending = st.session_state.pop("pending_q", None)

        header = st.container()
        box = st.container(height=440, border=False)
        typed = st.chat_input("Ask about decisions, owners, deadlines…", key="chat_input")
        question = (typed or pending or "").strip()

        with box:
            if not history and not question:
                st.markdown(H("""
                <div class="chat-welcome">
                    <div class="chat-welcome-title">Ask about this meeting</div>
                    <div class="chat-welcome-text">Answers come from the transcript. Pick a question or write your own below.</div>
                </div>
                """), unsafe_allow_html=True)
                grid = st.columns(2, gap="small")
                for i, q in enumerate(SUGGESTED_QUESTIONS):
                    grid[i % 2].button(q, key=f"sugg_{i}", on_click=_ask_suggested, args=(q,))

            for msg in history:
                with st.chat_message(msg["role"]):
                    md(msg["content"])

            if question:
                with st.chat_message("user"):
                    md(question)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Searching the transcript…"):
                            answer = as_text(ask_question(r["rag_chain"], question))
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Chat failed")
                        st.markdown(notice_html("error", f"That question couldn't be answered ({exc}). Try rephrasing it."),
                                    unsafe_allow_html=True)
                    else:
                        md(answer or "No answer was returned. Try rephrasing the question.")
                        history.append({"role": "user", "content": question})
                        history.append({"role": "assistant", "content": answer})

        with header:  # filled last, so it reflects the answer that was just added
            if history:
                note_col, clear_col = st.columns([5, 1.4], vertical_alignment="center")
                note_col.caption("Answers come from the transcript of this recording.")
                clear_col.button("Clear chat", key="clear_chat", on_click=_clear_chat, icon=":material/delete:")


def render_empty_state() -> None:
    st.markdown(hero_html(), unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(BRAND_HTML, unsafe_allow_html=True)

    source_text = st.text_input("YouTube link or file path", placeholder="https://youtube.com/watch?v=…", key="source_text")
    st.markdown('<div class="or-rule">or</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a recording", type=UPLOAD_TYPES, key="source_file",
                                help="Video or audio. Large files can take a few minutes to upload.")
    language = st.selectbox("Spoken language", ["english", "hinglish"], index=0, format_func=str.title,
                            key="language", help="Choose Hinglish for speech that mixes Hindi and English.")
    run_clicked = st.button("Analyse recording", type="primary", icon=":material/bolt:", key="run")

    if uploaded is not None and source_text.strip():
        st.caption("Both are filled in, so the uploaded file will be used.")

    if st.session_state.result:
        with st.expander("Processing details"):
            st.markdown(rail_html(st.session_state.steps, st.session_state.times, compact=True), unsafe_allow_html=True)


# ─── Main ───────────────────────────────────────────────────────────────────────
if run_clicked:
    problem = validate_source(source_text, uploaded)
    if problem:
        st.markdown(notice_html("warn", problem), unsafe_allow_html=True)
    else:
        analyse(source_text, uploaded, language)  # finishes with st.rerun() or st.stop()

if st.session_state.result:
    render_results(st.session_state.result)
else:
    render_empty_state()