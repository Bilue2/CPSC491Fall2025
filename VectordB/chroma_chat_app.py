# chroma_chat_app.py
# Run with: streamlit run chroma_chat_app.py

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

# -------------------
# Optional: cloud-capable Chroma SDK
# -------------------
try:
    from chromadb import HttpClient
    from chromadb import PersistentClient
except ImportError:
    st.error("ChromaDB SDK not installed. Run: pip install chromadb")
    st.stop()

from openai import OpenAI

try:
    from serpapi import GoogleSearch
except ImportError:
    try:
        from serpapi.google_search import GoogleSearch
    except ImportError:
        GoogleSearch = None
        st.warning("SerpAPI not available — external search disabled.")

# -------------------
# Streamlit page + logging
# -------------------
st.set_page_config(page_title="Regulatory AI Assistant", page_icon="📘", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# Custom CSS for chat bubbles
# -------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .app-title { color: #002855; font-weight: 700; font-size: 28px; text-align:left; margin-bottom:4px; }
    section[data-testid="stSidebar"] { background-color: #F6FBFF; color: #002855; }
    .user-bubble { background: #E6EEF7; color: #002855; padding:12px 14px; border-radius:14px; margin:6px 0; max-width:78%; font-size:15px; box-shadow:0 1px 3px rgba(0,0,0,0.06);}
    .assistant-bubble { background: linear-gradient(180deg,#002855,#003D7A); color:#ffffff; padding:12px 14px; border-radius:14px; margin:6px 0; max-width:78%; font-size:15px; box-shadow:0 2px 6px rgba(0,0,0,0.12);}
    .meta { font-size: 12px; color: #7a869a; margin-top:6px; }
    .chat-row { display:flex; flex-direction: row; align-items: flex-start; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .stButton>button { background-color:#002855; color:white; border-radius:8px; padding:8px 12px; }
    .stButton>button:hover { background-color:#003D99; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------
# Load secrets
# -------------------
missing = []
APP_USER = st.secrets.get("APP_USERNAME")
APP_PASS = st.secrets.get("APP_PASSWORD")
if not APP_USER or not APP_PASS:
    missing.append("APP_USERNAME / APP_PASSWORD")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")

SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", "")
CHROMA_CLOUD_API_KEY = st.secrets.get("CHROMA_CLOUD_API_KEY")
CHROMA_CLOUD_TENANT = st.secrets.get("CHROMA_CLOUD_TENANT")
CHROMA_CLOUD_DATABASE = st.secrets.get("CHROMA_CLOUD_DATABASE")

COLLECTION_NAME = st.secrets.get("CHROMA_COLLECTION", "fcc_documents")

if missing:
    st.error("Missing required secrets: " + ", ".join(missing))
    st.stop()

# -------------------
# OpenAI client
# -------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------
# Initialize Chroma client
# -------------------
client = None
collection = None

try:
    if CHROMA_CLOUD_API_KEY and CHROMA_CLOUD_TENANT and CHROMA_CLOUD_DATABASE:
        # Chroma Cloud setup
        client = ChromaClient(
            Settings(
                chroma_api_impl="rest",
                chroma_server_host="https://api.trychroma.com",
                chroma_server_http_headers={
                    "Authorization": f"Bearer {CHROMA_CLOUD_API_KEY}",
                    "X-Chroma-Tenant": CHROMA_CLOUD_TENANT,
                    "X-Chroma-Database": CHROMA_CLOUD_DATABASE,
                },
            )
        )
        st.sidebar.success("🟢 Using Chroma Cloud")
   
    # Get or create your collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

except Exception as e:
    st.error("Failed to initialize Chroma client: " + str(e))
    logger.exception("Chroma init error")
    st.stop()

# -------------------
# Utilities
# -------------------
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def document_exists_by_hash_or_url(content_hash: str, url: Optional[str] = "") -> bool:
    try:
        res = collection.get(where={"hash": content_hash}, include=["ids"])
        if res.get("ids"):
            return True
        if url:
            res2 = collection.get(where={"source": url}, include=["ids"])
            return bool(res2.get("ids"))
        return False
    except Exception:
        return False

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def ingest_external_document_to_chroma(title: str, url: str, content: str) -> bool:
    if not content or len(content) < 300:
        return False
    content_hash = compute_hash(content)
    if document_exists_by_hash_or_url(content_hash, url):
        return False
    try:
        embedding = embed_text(content)
        uid = str(uuid4())
        metadata = {"source": url, "title": title, "retrieved": str(datetime.date.today()), "hash": content_hash}
        collection.add(ids=[uid], documents=[content], embeddings=[embedding], metadatas=[metadata])
        return True
    except Exception as e:
        logger.exception("Failed to ingest: %s", e)
        return False

def external_search(query: str, max_results: int = 5) -> List[Dict]:
    if not SERPAPI_API_KEY or GoogleSearch is None:
        return []
    try:
        results = GoogleSearch({"q": query, "engine":"google","api_key":SERPAPI_API_KEY,"num":max_results}).get_dict()
        return [{"title": r.get("title","Untitled"), "url": r.get("link",""), "content": r.get("snippet","")} for r in results.get("organic_results",[])]
    except Exception as e:
        logger.exception("SerpAPI search failed: %s", e)
        return []

def fetch_full_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return "\n\n".join(p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip())
    except Exception:
        return ""

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Dict]:
    q_emb = embed_text(query)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas"])
    docs = results.get("documents",[[]])[0]
    metas = results.get("metadatas",[[]])[0]
    out = [{"document": d,"metadata":m} for d,m in zip(docs,metas)]
    return out

def build_prompt(query: str, embedded_chunks: List[Dict], external_docs: List[Dict]) -> str:
    system_instructions = ("You are a specialist assistant restricted to embedded documents "
                           "and explicit external sources. Answer using only this content.")
    parts = []
    for i, c in enumerate(embedded_chunks):
        md = c.get("metadata",{}) or {}
        title = md.get("title") or md.get("source") or f"doc-{i}"
        text = c.get("document","")[:1500]
        parts.append(f"EMBEDDED: {title}\n{text}")
    for d in external_docs:
        title = d.get("title","External")
        url = d.get("url","")
        content = d.get("content","")[:1500]
        parts.append(f"EXTERNAL: {title} (URL: {url})\n{content}")
    context_text = "\n\n---\n\n".join(parts) if parts else "No context available."
    return f"{system_instructions}\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer (with markdown citations under 'Sources:'):"

def parse_sources(answer: str) -> Tuple[str, List[Tuple[str,str]]]:
    marker = "\nSources:"
    if marker in answer:
        ans, src = answer.split(marker,1)
        sources=[]
        for line in src.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try:
                    title = line.split("[",1)[1].split("]")[0]
                    url = line.split("(",1)[1].split(")")[0]
                    sources.append((title,url))
                except: continue
        return ans.strip(), sources
    return answer.strip(), []

# -------------------
# Authentication
# -------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def login_screen():
    st.title("🔐 Regulatory Assistant Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == APP_USER and password == APP_PASS:
            st.session_state.authenticated = True
            st.success("Logged in! Loading assistant...")
            time.sleep(0.5)
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# -------------------
# Sidebar: upload + logout
# -------------------
with st.sidebar:
    st.header("Controls")
    st.caption("Chroma: " + ("Cloud" if CHROMA_CLOUD_API_KEY else "Local"))
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
    uploaded = st.file_uploader("Upload text/PDF to ingest", type=["txt","pdf"])
    if uploaded:
        try:
            if uploaded.type=="application/pdf":
                from PyPDF2 import PdfReader
                import io
                reader = PdfReader(io.BytesIO(uploaded.getvalue()))
                text="\n\n".join(p.extract_text() or "" for p in reader.pages)
            else:
                text = uploaded.getvalue().decode("utf-8", errors="ignore")
            ingest_external_document_to_chroma(uploaded.name,f"uploaded://{uploaded.name}",text)
            st.success("Document ingested (if not duplicate).")
        except Exception as e:
            st.error(f"Ingest failed: {e}")

# -------------------
# Main UI: chat
# -------------------
st.markdown('<div class="app-title">📘 Regulatory AI Assistant</div>', unsafe_allow_html=True)
st.write("Ask about public safety, emergency alerts, cybersecurity, and regulatory policies.")

# Chat display
for msg in st.session_state.messages:
    ts = msg.get("time","")
    if msg["role"]=="user":
        st.markdown(f'<div class="chat-row user"><div class="user-bubble">{msg["text"]}<div class="meta">{ts}</div></div></div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-row assistant"><div class="assistant-bubble">{msg["text"]}<div class="meta">{ts}</div></div></div>',unsafe_allow_html=True)

# Input
prompt = st.text_area("Your question", value="", height=100, placeholder="Type your question here...")
if st.button("Ask") and prompt.strip():
    now=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.session_state.messages.append({"role":"user","text":prompt,"time":now})
    st.experimental_rerun()

# -------------------
# Process latest user message
# -------------------
def process_latest_user_message():
    if not st.session_state.messages:
        return
    last=st.session_state.messages[-1]
    if last["role"]!="user":
        return
    user_text=last["text"]
    timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    with st.spinner("Retrieving context and external docs..."):
        embedded_chunks = retrieve_relevant_chunks(user_text)
        external_docs = external_search(user_text) if SERPAPI_API_KEY else []
        for d in external_docs:
            full_text = fetch_full_text(d.get("url","")) or d.get("content","")
            if full_text:
                d["content"]=full_text
                ingest_external_document_to_chroma(d.get("title","External"),d.get("url",""),full_text)
        final_prompt = build_prompt(user_text,embedded_chunks,external_docs)
    with st.spinner("Generating answer..."):
        try:
            response=openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":final_prompt}],
                max_tokens=500,
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer=f"Error generating response: {e}"
    ans_text, sources = parse_sources(answer)
    if sources:
        ans_text+="\n\n**Sources:**\n" + "\n".join(f"- [{t}]({u})" for t,u in sources)
    st.session_state.messages.append({"role":"assistant","text":ans_text,"time":timestamp})
    st.experimental_rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"]=="user":
    process_latest_user_message()
