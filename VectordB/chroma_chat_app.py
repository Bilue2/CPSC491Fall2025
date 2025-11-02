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


# NOTE: chromadb Cloud client API may change across releases.
# The code below tries to use a Chroma client that accepts cloud credentials.
try:
    from chromadb import Client as ChromaClient  # cloud-capable constructor in some versions
    from chromadb import PersistentClient  # fallback
except Exception:
    ChromaClient = None
    PersistentClient = None

# Third-party SDKs that must be installed:
# pip install chromadb openai serpapi beautifulsoup4 PyPDF2
from openai import OpenAI

try:
    from serpapi import GoogleSearch
except ImportError:
    try:
        from serpapi.google_search import GoogleSearch
    except ImportError:
        GoogleSearch = None
        st.error("SerpApi client import failed — check installation of `serpapi` or `google-search-results` package.")
        
# -------------------
#  USER INSTRUCTIONS
# -------------------
#
# Required secrets (set in Streamlit Cloud or local .streamlit/secrets.toml):
#
# APP_USERNAME = "youruser"
# APP_PASSWORD = "yourpass"
# OPENAI_API_KEY = "sk-..."
# SERPAPI_API_KEY = "..."                 # optional, but useful
#
# For Chroma Cloud (preferred):
# CHROMA_CLOUD_API_KEY = "chroma-cloud-key"
# CHROMA_CLOUD_TENANT  = "tenant-id"
# CHROMA_CLOUD_DATABASE= "database-name"
#
# If you don't provide Chroma Cloud credentials, the app will attempt to use a
# local PersistentClient at ./chroma_storage (not recommended for deployment).
#
# -------------------

st.set_page_config(page_title="Regulatory AI Assistant", page_icon="📘", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# Custom CSS for polished chat style (dark-blue / white)
# -------------------
st.markdown(
    """
    <style>
    /* Page */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Title */
    .app-title { color: #002855; font-weight: 700; font-size: 28px; text-align:left; margin-bottom:4px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #F6FBFF; color: #002855; }

    /* Chat bubbles */
    .user-bubble {
        background: #E6EEF7;
        color: #002855;
        padding: 12px 14px;
        border-radius: 14px;
        margin: 6px 0;
        max-width: 78%;
        font-size: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .assistant-bubble {
        background: linear-gradient(180deg,#002855,#003D7A);
        color: #ffffff;
        padding: 12px 14px;
        border-radius: 14px;
        margin: 6px 0;
        max-width: 78%;
        font-size: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    .meta { font-size: 12px; color: #7a869a; margin-top:6px; }
    .chat-row { display:flex; flex-direction: row; align-items: flex-start; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    /* Buttons */
    .stButton>button { background-color:#002855; color:white; border-radius:8px; padding:8px 12px; }
    .stButton>button:hover { background-color:#003D99; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------
# Read secrets (fail fast if critical secrets missing)
# -------------------
missing = []
try:
    APP_USER = st.secrets["APP_USERNAME"]
    APP_PASS = st.secrets["APP_PASSWORD"]
except Exception:
    missing.append("APP_USERNAME / APP_PASSWORD")

OPENAI_API_KEY = st.secrets.get(OPENAI_API_KEY)
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")

# SerpAPI optional
SERPAPI_API_KEY = st.secrets.get(SERPAPI_API_KEY, "")

# Chroma Cloud optional credentials
CHROMA_CLOUD_API_KEY = st.secrets.get("CHROMA_CLOUD_API_KEY")
CHROMA_CLOUD_TENANT = st.secrets.get("CHROMA_CLOUD_TENANT")
CHROMA_CLOUD_DATABASE = st.secrets.get("CHROMA_CLOUD_DATABASE")

if missing:
    st.error("Missing required secrets: " + ", ".join(missing) + ". Add them in Streamlit Secrets and restart.")
    st.stop()

# -------------------
# App configuration constants
# -------------------
COLLECTION_NAME = st.secrets.get("CHROMA_COLLECTION", "fcc_documents")
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
MIN_INGEST_LENGTH = 300  # characters

# -------------------
# Helper: OpenAI client
# -------------------
def get_openai_client() -> OpenAI:
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error("Failed to initialize OpenAI client. Check OPENAI_API_KEY.")
        raise

openai_client = get_openai_client()

# -------------------
# Initialize Chroma client (Cloud preferred; local fallback)
# -------------------
client = None
collection = None

def init_chroma_client():
    global client, collection
    try:
        if CHROMA_CLOUD_API_KEY and CHROMA_CLOUD_TENANT and CHROMA_CLOUD_DATABASE:
            # Many chromadb versions support constructing Client with these named args.
            # If your installed chromadb version uses a different signature, update accordingly.
            if ChromaClient is None:
                raise RuntimeError("chromadb Client class not available - install chromadb.")
            client = ChromaClient(api_key=CHROMA_CLOUD_API_KEY, tenant=CHROMA_CLOUD_TENANT, database=CHROMA_CLOUD_DATABASE)
            st.sidebar.success("🟢 Using Chroma Cloud")
        else:
            # Local persistent client fallback (useful for local testing only)
            if PersistentClient is None:
                raise RuntimeError("PersistentClient not available - install chromadb.")
            persist_path = "./chroma_storage"
            client = PersistentClient(path=persist_path)
            st.sidebar.warning("⚠️ Using local Chroma storage (not persistent across Streamlit Cloud sleeps).")
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error("Failed to initialize Chroma client: " + str(e))
        logger.exception("Chroma init error")
        st.stop()

init_chroma_client()

# -------------------
# Utility functions: hashing, existence check, embedding, ingestion
# -------------------
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def document_exists_by_hash_or_url(content_hash: str, url: Optional[str] = "") -> bool:
    """
    Try to check via Chroma collection for existing doc by hash, then by URL.
    Uses collection.get(where=...) if available; otherwise falls back to scanning metadata.
    """
    try:
        # Preferred: if collection.get supports "where" clause
        try:
            where = {"hash": content_hash}
            if url:
                # Most Chroma SDKs don't accept $or directly; try single queries
                res = collection.get(where=where, include=["ids"])
                if res and res.get("ids"):
                    return len(res["ids"]) > 0
                # Check by URL
                res2 = collection.get(where={"source": url}, include=["ids"])
                if res2 and res2.get("ids"):
                    return len(res2["ids"]) > 0
            else:
                res = collection.get(where=where, include=["ids"])
                return bool(res and res.get("ids"))
        except Exception:
            # Fallback: pull metadata (may be heavy) and scan
            all_meta = collection.get(include=["metadatas", "ids"])
            metas = all_meta.get("metadatas", [])
            ids = all_meta.get("ids", [])
            for meta in metas:
                if not isinstance(meta, dict):
                    continue
                if meta.get("hash") == content_hash:
                    return True
                if url and meta.get("source") == url:
                    return True
            return False
    except Exception as e:
        logger.exception("document_exists check failed: %s", e)
        return False  # safer to treat as not existing

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def ingest_external_document_to_chroma(title: str, url: str, content: str) -> bool:
    """
    Adds content to Chroma only if it's not a duplicate (by content hash or URL).
    Returns True if added, False if skipped or error.
    """
    if not content or len(content) < MIN_INGEST_LENGTH:
        logger.info("Skipping ingest: too short or empty.")
        return False
    content_hash = compute_hash(content)
    if document_exists_by_hash_or_url(content_hash, url):
        logger.info("Skipping ingest - already exists (hash/url).")
        return False
    try:
        embedding = embed_text(content)
        uid = str(uuid4())
        metadata = {
            "source": url,
            "title": title,
            "retrieved": str(datetime.date.today()),
            "hash": content_hash,
        }
        collection.add(ids=[uid], documents=[content], embeddings=[embedding], metadatas=[metadata])
        logger.info("Ingested document: %s", url or uid)
        return True
    except Exception as e:
        logger.exception("Failed to ingest: %s", e)
        return False

# -------------------
# External search / fetch helpers (SerpAPI + HTML fetch)
# -------------------
def external_search(query: str, max_results: int = 5) -> List[Dict]:
    if not SERPAPI_API_KEY:
        return []
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_API_KEY,
        "num": max_results,
        "hl": "en",
        "gl": "us",
    }
    try:
        results = GoogleSearch(params).get_dict()
        external = []
        for r in results.get("organic_results", []):
            external.append({
                "title": r.get("title", "Untitled"),
                "url": r.get("link", ""),
                "content": r.get("snippet", "") or ""
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
        text = "\n\n".join(paragraphs)
        return text
    except Exception as e:
        logger.debug("fetch_full_text failed for %s: %s", url, e)
        return ""

# -------------------
# Retrieval helper
# -------------------
def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    q_emb = embed_text(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        if doc is None:
            continue
        out.append({"document": doc, "metadata": meta or {}, "distance": dist})
    return out

# -------------------
# Prompt builder / source parsing
# -------------------
def build_prompt(query: str, embedded_chunks: List[Dict], external_docs: List[Dict]) -> str:
    system_instructions = (
        "You are a specialist assistant restricted to the provided embedded dataset and any "
        "explicit external documents included. Answer only using that content. "
        "Do not fabricate facts. Cite any external sources used under 'Sources:'."
    )
    parts = []
    # include the most relevant embedded snippets
    for i, chunk in enumerate(embedded_chunks):
        md = chunk.get("metadata", {}) or {}
        title = md.get("title") or md.get("source") or f"doc-{i}"
        text = chunk.get("document", "")[:1500]  # trim large docs
        parts.append(f"EMBEDDED: {title}\n{text}")

    for doc in external_docs:
        title = doc.get("title", "External")
        url = doc.get("url", "")
        content = doc.get("content", "")[:1500]
        parts.append(f"EXTERNAL: {title} (URL: {url})\n{content}")

    context_text = "\n\n---\n\n".join(parts) if parts else "No context documents available."
    prompt = f"{system_instructions}\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer (with markdown citations under 'Sources:'):"
    return prompt

def parse_sources(answer: str) -> Tuple[str, List[Tuple[str, str]]]:
    marker = "\nSources:"
    if marker in answer:
        ans_part, src_part = answer.split(marker, 1)
        sources = []
        for line in src_part.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try:
                    title = line.split("[", 1)[1].split("]")[0]
                    url = line.split("(", 1)[1].split(")")[0]
                    sources.append((title, url))
                except Exception:
                    continue
        return ans_part.strip(), sources
    return answer.strip(), []

# -------------------
# UI: auth, layout, chat loop
# -------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role":"user"/"assistant", "text":..., "time":...}

def login_screen():
    st.title("🔐 Regulatory Assistant Login")
    st.markdown("Sign in to use the Regulatory AI Assistant.")
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

# Sidebar content
with st.sidebar:
    st.image("logo.png" if os.path.exists("logo.png") else " ", width=140)
    st.header("Controls")
    st.caption("Chroma storage: " + ("Cloud" if CHROMA_CLOUD_API_KEY else "Local"))
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    st.subheader("Ingest")
    st.write("Upload a text or PDF file to ingest into the collection (deduplicated).")
    uploaded = st.file_uploader("Upload file to ingest (txt, pdf)", type=["txt", "pdf"])
    if uploaded:
        try:
            if uploaded.type == "application/pdf":
                # Try PyPDF2; fallback to raw text
                try:
                    from PyPDF2 import PdfReader
                    import io
                    reader = PdfReader(io.BytesIO(uploaded.getvalue()))
                    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception:
                    text = uploaded.getvalue().decode("utf-8", errors="ignore")
            else:
                text = uploaded.getvalue().decode("utf-8", errors="ignore")

            title = uploaded.name
            fake_url = f"uploaded://{uploaded.name}"
            added = ingest_external_document_to_chroma(title=title, url=fake_url, content=text)
            if added:
                st.success("Document ingested.")
            else:
                st.info("Document skipped (duplicate or too short).")
        except Exception as e:
            st.error("Ingest failed: " + str(e))

    st.markdown("---")
    st.subheader("Debug / Info")
    if st.button("Show collection count"):
        try:
            cnt = collection.count()
            st.write("Collection count:", cnt)
        except Exception as e:
            st.error("Error querying collection count: " + str(e))

# Main area: title + chat
st.markdown('<div class="app-title">📘 Regulatory AI Assistant</div>', unsafe_allow_html=True)
st.write("Ask about emergency alerts, public safety communications, cybersecurity policy, and regulation. Answers will be restricted to content present in the embeddings and ingested external sources.")

# Chat display container
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        ts = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-row user"><div class="user-bubble">{st.markdown(msg["text"], unsafe_allow_html=False) or msg["text"]}<div class="meta" style="text-align:right;">{ts}</div></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-row assistant"><div class="assistant-bubble">{msg["text"]}<div class="meta">{ts}</div></div></div>',
                unsafe_allow_html=True,
            )

# Input area
prompt = st.text_area("Your question", value="", height=100, placeholder="Type your question here...")
col1, col2 = st.columns([1, 0.25])
with col2:
    submit = st.button("Ask")

if submit and prompt.strip():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    # Append user message
    st.session_state.messages.append({"role": "user", "text": prompt, "time": now})
    # Immediately re-render to show user message
    st.rerun()

# Processing: detect latest user message not answered
def process_latest_user_message():
    if not st.session_state.messages:
        return
    last = st.session_state.messages[-1]
    if last["role"] != "user":
        return

    user_text = last["text"]
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    # 1) retrieve from chroma
    with st.spinner("Retrieving context and searching external sources..."):
        try:
            embedded_chunks = retrieve_relevant_chunks(user_text, top_k=SIMILARITY_TOP_K)
        except Exception as e:
            logger.exception("retrieve error: %s", e)
            embedded_chunks = []

        # 2) run SerpAPI search (if available) and ingest new results (deduped)
        external_docs = external_search(user_text, max_results=5) if SERPAPI_API_KEY else []
        ingested_external = []
        for d in external_docs:
            url = d.get("url", "")
            # fetch full text to compute hash/deduplicate
            full = fetch_full_text(url) or d.get("content", "")
            if not full:
                continue
            d["content"] = full
            added = ingest_external_document_to_chroma(title=d.get("title", "External"), url=url, content=full)
            if added:
                ingested_external.append(d)

        # Extend embedded_chunks if we ingested new external docs (optional: retrieve again or include snippets)
        # For simplicity, append the external docs as context snippets (they're already added to DB)
        # Build prompt
        final_prompt = build_prompt(user_text, embedded_chunks, external_docs)

    # 3) Call OpenAI Chat model
    with st.spinner("Generating answer..."):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": final_prompt}],
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.2,
            )
            full_answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            full_answer = f"Error generating response: {e}"

    # 4) parse sources & append assistant message
    ans_text, sources = parse_sources(full_answer)
    assistant_text = ans_text
    if sources:
        sources_md = "\n\n**Sources:**\n" + "\n".join(f"- [{t}]({u})" for t, u in sources)
        assistant_text = assistant_text + "\n\n" + sources_md

    st.session_state.messages.append({"role": "assistant", "text": assistant_text, "time": timestamp})
    # Re-render to show assistant reply
    st.rerun()

# If last message is user, process it
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    process_latest_user_message()
