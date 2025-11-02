# chroma_chat_app.py
# Run with: streamlit run chroma_chat_app.py

import os
import time
import datetime
import logging
from uuid import uuid4
from typing import List, Dict, Tuple

import streamlit as st
from chromadb import PersistentClient
from openai import OpenAI
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup

# === Config ===
PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_storage")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fcc_documents")
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500

# === User Authentication ===
VALID_USERNAME = st.secrets["APP_USERNAME"]
VALID_PASSWORD = st.secrets["APP_PASSWORD"]

st.set_page_config(page_title="Regulatory AI Assistant", page_icon="📘", layout="wide")

# === API Keys ===
openai_api_key = st.secrets["OPENAI_API_KEY"]
serpapi_api_key = st.secrets["SERPAPI_API_KEY"]


# Login state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_screen():
    """Displays the login form."""
    st.title("Regulatory Assistant Login")
    st.write("Please log in to access the Regulatory AI Assistant.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Login successful! Loading the assistant...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")


# === If not logged in, show login form ===
if not st.session_state.authenticated:
    login_screen()
    st.stop()


# === Once logged in, show main app ===
# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# Logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize clients
client = PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
openai_client = OpenAI(api_key=openai_api_key)

# === Core Functions ===
def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    q_emb = embed_text(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"document": doc, "metadata": meta} for doc, meta in zip(docs, metas)]


def external_search(query: str, max_results: int = 5) -> List[Dict]:
    params = {
        "q": query,
        "engine": "google",
        "api_key": serpapi_api_key,
        "num": max_results,
        "hl": "en",
        "gl": "us",
    }
    results = GoogleSearch(params).get_dict()
    external = []
    for r in results.get("organic_results", []):
        external.append(
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("link", ""),
                "content": r.get("snippet", ""),
            }
        )
    return external


def fetch_full_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))
    except Exception:
        return ""


def ingest_external_document_to_chroma(doc: Dict):
    content = doc.get("content", "")
    if not content or len(content) < 300:
        return
    embedding = embed_text(content)
    uid = str(uuid4())
    metadata = {
        "source": doc.get("url", ""),
        "title": doc.get("title", ""),
        "retrieved": str(datetime.date.today()),
    }
    collection.add(
        ids=[uid], documents=[content], embeddings=[embedding], metadatas=[metadata]
    )


def build_prompt(query: str, embedded_chunks: List[Dict], external_docs: List[Dict]) -> str:
    system_instructions = (
       "You are a domain-specific assistant trained solely on emergency alert systems, public safety communications,"
        "cybersecurity policy, disaster response frameworks, and regulatory principles as defined in the embedded dataset."
        "You must restrict your responses only to the information contained in the embedded data and the embeddings added through the SerpAPI search, refrain from generating answers outside this scope."
        "Do not reference general knowledge, FCC responses, or unrelated domains (e.g., cooking, entertainment, etc.)."
        "Where relevant, relate insights strictly to ideas present in the embedded documents or clearly supported by them.\n"
        "Do not fabricate sources. Use markdown links for citations under 'Sources:'."
    )

    parts = []
    for doc in external_docs:
        title = doc.get("title", "External Source")
        url = doc.get("url", "")
        parts.append(f"Title: {title} (URL: {url})\n{doc.get('content', '')}")

    context_text = "\n---\n".join(parts)

    return (
        f"{system_instructions}\n\nContext:\n{context_text}\n\n"
        f"Question: {query}\nAnswer (with markdown citations under 'Sources:'):"
    )


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


# === Streamlit Chat Interface ===
st.title("📘 Regulatory AI Assistant")
st.caption("Ask about public safety, emergency alerts, cybersecurity, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating response..."):
            try:
                embedded_chunks = retrieve_relevant_chunks(prompt)
                external_docs = external_search(prompt)
                for doc in external_docs:
                    full_text = fetch_full_text(doc["url"])
                    if full_text:
                        doc["content"] = full_text
                        ingest_external_document_to_chroma(doc)

                final_prompt = build_prompt(prompt, embedded_chunks, external_docs)
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": final_prompt}],
                    max_tokens=MAX_RESPONSE_TOKENS,
                    temperature=0.3,
                )
                full_answer = response.choices[0].message.content.strip()
                ans_text, sources = parse_sources(full_answer)

                st.markdown(ans_text)
                if sources:
                    st.markdown("**Sources:**")
                    for title, url in sources:
                        st.markdown(f"- [{title}]({url})")

                st.session_state.messages.append(
                    {"role": "assistant", "content": ans_text}
                )
            except Exception as e:
                st.error(f"Error: {e}")



