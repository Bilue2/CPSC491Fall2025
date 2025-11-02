# chroma_chat_app.py
# Run: streamlit run chroma_chat_app.py

import os
import time
import datetime
import hashlib
import logging
from uuid import uuid4
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from serpapi import GoogleSearch
from chromadb import Client as ChromaClient

# -------------------
# Streamlit page config
# -------------------
st.set_page_config(page_title="Regulatory AI Assistant", page_icon="📘", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# Custom CSS for chat style & top logout
# -------------------
st.markdown("""
<style>
header {display: flex; justify-content: space-between; align-items: center;}
.block-container {padding-top: 4rem; padding-bottom: 2rem;}
.app-title {color:#002855; font-weight:700; font-size:28px; margin-bottom:4px;}
.user-bubble {background:#E6EEF7;color:#002855;padding:12px 14px;border-radius:14px;margin:6px 0;max-width:78%;font-size:15px;box-shadow:0 1px 3px rgba(0,0,0,0.06);}
.assistant-bubble {background:linear-gradient(180deg,#002855,#003D7A);color:#ffffff;padding:12px 14px;border-radius:14px;margin:6px 0;max-width:78%;font-size:15px;box-shadow:0 2px 6px rgba(0,0,0,0.12);}
.meta {font-size:12px;color:#7a869a;margin-top:6px;}
.chat-row {display:flex; flex-direction: row; align-items:flex-start;}
.chat-row.user {justify-content:flex-end;}
.chat-row.assistant {justify-content:flex-start;}
.stButton>button {background-color:#002855;color:white;border-radius:8px;padding:8px 12px;}
.stButton>button:hover {background-color:#003D99;color:white;}
.upload-section {margin-top:10px;}
</style>
""", unsafe_allow_html=True)

# -------------------
# Load secrets
# -------------------
missing = []
try:
    APP_USER = st.secrets["APP_USERNAME"]
    APP_PASS = st.secrets["APP_PASSWORD"]
except KeyError:
    missing.append("APP_USERNAME / APP_PASSWORD")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")

SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    missing.append("SERPAPI_API_KEY")

CHROMA_API_KEY = st.secrets.get("CHROMA_API_KEY")
CHROMA_TENANT = st.secrets.get("CHROMA_TENANT")
CHROMA_DATABASE = st.secrets.get("CHROMA_DATABASE")
COLLECTION_NAME = st.secrets.get("CHROMA_COLLECTION", "fcc_documents")

if missing:
    st.error("Missing required secrets: " + ", ".join(missing))
    st.stop()

# -------------------
# App constants
# -------------------
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
MIN_INGEST_LENGTH = 300

# -------------------
# Initialize OpenAI client
# -------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------
# Initialize Chroma client (Cloud)
# -------------------
try:
    client = ChromaClient()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
except Exception as e:
    st.error(f"Failed to initialize Chroma client: {e}")
    logger.exception("Chroma init error")
    st.stop()

# -------------------
# Authentication
# -------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def login_screen():
    st.title("🔐 Regulatory Assistant Login")
    st.markdown("Sign in to use the assistant.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            if username == APP_USER and password == APP_PASS:
                st.session_state.authenticated = True
                st.success("Logged in — loading assistant...")
                time.sleep(0.6)
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col2:
        if st.button("Exit"):
            st.stop()

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# -------------------
# Logout Top Right
# -------------------
col1, col2 = st.columns([9,1])
with col1:
    st.markdown('<div class="app-title">📘 Regulatory AI Assistant</div>', unsafe_allow_html=True)
with col2:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

st.write("Ask questions about emergency alerts, public safety, cybersecurity, and regulation.")

# -------------------
# Helper Functions
# -------------------
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def document_exists(content_hash: str, url: Optional[str] = "") -> bool:
    try:
        all_meta = collection.get(include=["metadatas", "ids"])
        metas = all_meta.get("metadatas", [])
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            if meta.get("hash") == content_hash or (url and meta.get("source") == url):
                return True
        return False
    except Exception as e:
        logger.exception("document_exists failed: %s", e)
        return False

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def ingest_document(title: str, url: str, content: str) -> bool:
    if not content or len(content) < MIN_INGEST_LENGTH:
        return False
    content_hash = compute_hash(content)
    if document_exists(content_hash, url):
        return False
    try:
        embedding = embed_text(content)
        collection.add(
            ids=[str(uuid4())],
            documents=[content],
            embeddings=[embedding],
            metadatas={"title": title, "source": url, "hash": content_hash, "retrieved": str(datetime.date.today())}
        )
        return True
    except Exception as e:
        logger.exception("Failed to ingest: %s", e)
        return False

def external_search(query: str, max_results: int = 5) -> List[Dict]:
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_API_KEY,
        "num": max_results,
    }
    try:
        results = GoogleSearch(params).get_dict()
        external = []
        for r in results.get("organic_results", []):
            external.append({
                "title": r.get("title","Untitled"),
                "url": r.get("link",""),
                "content": r.get("snippet","")
            })
        return external
    except Exception as e:
        logger.exception("SerpAPI search failed: %s", e)
        return []

def fetch_full_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        return "\n\n".join(paragraphs)
    except Exception:
        return ""

def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    q_emb = embed_text(query)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas"])
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"document": d, "metadata": m} for d,m in zip(docs, metas)]

def build_prompt(query: str, embedded_chunks: List[Dict], external_docs: List[Dict]) -> str:
    system_instructions = (
    "You are an expert on emergency alert systems (EAS, WEA, IPAWS), public safety communications, and regulatory frameworks. "
    "You must restrict your responses only to the information contained in the embedded data and the embeddings added through the SerpAPI search, refrain from generating answers outside this scope."
    "Provide detailed, specific answers using the context below.\n\n"
    "Guidelines:\n"
    "- Include specific details: dates, names, statistics, and technical terms like (EAS, WEA, IPAWS, CAP, FCC Part 11 and more)\n"
    '- Do not fabricate sources. Use markdown links for citations under \'Sources:\'.'
    "- Provide examples and context when helpful\n"
    "- If context is insufficient, supplement with your knowledge but indicate this clearly"
)
    parts = []
    for i, chunk in enumerate(embedded_chunks):
        title = chunk.get("metadata",{}).get("title", f"doc-{i}")
        text = chunk.get("document","")[:1500]
        parts.append(f"EMBEDDED: {title}\n{text}")
    for d in external_docs:
        parts.append(f"EXTERNAL: {d.get('title','External')} (URL: {d.get('url','')})\n{d.get('content','')[:1500]}")
    context_text = "\n\n---\n\n".join(parts) if parts else "No context documents available."
    return f"{system_instructions}\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer (with markdown citations under 'Sources:'):"

def parse_sources(answer: str) -> Tuple[str, List[Tuple[str,str]]]:
    marker = "\nSources:"
    if marker in answer:
        ans_part, src_part = answer.split(marker,1)
        sources=[]
        for line in src_part.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try:
                    t = line.split("[",1)[1].split("]")[0]
                    u = line.split("(",1)[1].split(")")[0]
                    sources.append((t,u))
                except: continue
        return ans_part.strip(), sources
    return answer.strip(), []

# -------------------
# Chat Input + Upload UI
# -------------------
st.markdown('<div class="upload-section">📄 Upload Documents (txt/pdf)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["txt","pdf"])
if uploaded:
    try:
        if uploaded.type=="application/pdf":
            from PyPDF2 import PdfReader
            import io
            reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = uploaded.getvalue().decode("utf-8",errors="ignore")
        fake_url = f"uploaded://{uploaded.name}"
        added = ingest_document(uploaded.name, fake_url, text)
        st.success("Document ingested." if added else "Skipped (duplicate/too short)")
    except Exception as e:
        st.error("Ingest failed: "+str(e))

prompt = st.text_area("Type your question here...", height=80)
if st.button("Ask") and prompt.strip():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.session_state.messages.append({"role":"user","text":prompt,"time":now})
    st.rerun()

# -------------------
# Display chat messages
# -------------------
for msg in st.session_state.messages:
    role = msg["role"]
    ts = msg.get("time","")
    bubble_class = "user-bubble" if role=="user" else "assistant-bubble"
    row_class = "chat-row user" if role=="user" else "chat-row assistant"
    st.markdown(f'<div class="{row_class}"><div class="{bubble_class}">{msg["text"]}<div class="meta">{ts}</div></div></div>', unsafe_allow_html=True)

# -------------------
# Process latest user message
# -------------------
def process_latest():
    if not st.session_state.messages: return
    last = st.session_state.messages[-1]
    if last["role"]!="user": return
    query = last["text"]

    with st.spinner("Retrieving context..."):
        embedded = retrieve_relevant_chunks(query)
        # If embeddings insufficient, use SERPAPI
        if not embedded:
            external_docs = external_search(query)
            for d in external_docs:
                full = fetch_full_text(d.get("url","")) or d.get("content","")
                d["content"] = full
                ingest_document(d.get("title","External"), d.get("url",""), full)
            embedded = retrieve_relevant_chunks(query)
        else:
            external_docs = []

        prompt_text = build_prompt(query, embedded, external_docs)

    with st.spinner("Generating answer..."):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":prompt_text}],
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.2
            )
            ans_full = resp.choices[0].message.content.strip()
        except Exception as e:
            ans_full = f"Error generating response: {e}"

    ans_text, sources = parse_sources(ans_full)
    if sources:
        ans_text += "\n\n**Sources:**\n" + "\n".join(f"- [{t}]({u})" for t,u in sources)

    st.session_state.messages.append({"role":"assistant","text":ans_text,"time":datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"]=="user":
    process_latest()
