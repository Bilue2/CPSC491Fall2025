"""
Simple CLI chat that retrieves top-k documents from ChromaDB and answers with inline citations.

Usage:
  python3 VectordB.chat_with_citations.py

Features:
- Uses same embedding model as ingestion/chat (text-embedding-3-small).
- Markdown-formatted source list with titles and links.
- Graceful handling of missing context.
- Explicit guardrails to avoid hallucinating beyond sources.
- Retries on transient API errors.
"""

from config import  get_api_key, get_serpapi_key
import os
import sys
import time
from typing import List, Dict, Tuple

from chromadb import PersistentClient
from openai import OpenAI
from serpapi import GoogleSearch

# === Configuration ===

PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_storage")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "regulatory_papers")
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
FALLBACK_TEXT = "No information available in the dataset or external sources for that question."

# === Clients ===

client = PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    return OpenAI(api_key=key)

openai_client = get_openai_client()

# === Embedding & Retrieval ===

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    q_emb = embed_text(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [{"document": doc, "metadata": meta} for doc, meta in zip(docs, metas)]

# === External Search ===

def external_search(query: str, max_results: int = 3) -> List[Dict]:
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
        result = GoogleSearch(params).get_dict()
    except Exception as e:
        print(f"⚠️ SerpAPI error: {e}")
        return []

    external = []
    for r in result.get("organic_results", [])[:max_results]:
        url = r.get("link")
        title = r.get("title") or "Untitled"
        snippet = r.get("snippet") or ""
        if url and "fcc.gov" not in url.lower():
            external.append({"title": title, "url": url, "content": snippet})
    return external

def fetch_full_text(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return "\n".join(p.get_text().strip() for p in soup.find_all("p"))
    except Exception:
        return ""

# === Prompt Construction ===

def build_prompt(query: str,
                 embedded_chunks: List[Dict],
                 external_docs: List[Dict]) -> str:
    system_instructions = (
        "You are a helpful assistant that specializes in emergency alert systems, public safety communications, cybersecurity policy, "
        "disaster response frameworks, and regulatory principles. You have access to embedded documents and relevant web sources.\n\n"
        "Use the provided context to guide your answers, but you may also use your own knowledge when helpful. Be clear, accurate, and concise.\n\n"
        "When you cite a document or source explicitly, include a markdown-formatted list at the end under the heading '📚 Sources:', with the title and link.\n\n"
        "If there's no relevant context provided, still try to help the user based on what you know."
    )

    parts = []

    for chunk in embedded_chunks:
        meta = chunk["metadata"]
        title = meta.get("title", "Embedded Document")
        url = meta.get("source") or meta.get("url", "")
        parts.append(f"Title: {title}" + (f" (URL: {url})" if url else "") + f"\n{chunk['document']}")

    for d in external_docs:
        title = d.get("title", "External Source")
        url = d.get("url", "")
        parts.append(f"Title: {title}" + (f" (URL: {url})" if url else "") + f"\n{d.get('content', '')}")

    context_text = "\n---\n".join(parts)

    return (
        f"{system_instructions}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def parse_sources(answer: str) -> Tuple[str, List[Tuple[str, str]]]:
    marker = "\n📚 Sources:"
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

# === Chat Loop ===

def chat():
    print("Chat Assistant (type 'exit' or Ctrl-C to quit)")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            embedded_chunks = retrieve_relevant_chunks(user_input)
            external_docs = external_search(user_input)

            for d in external_docs:
                full = fetch_full_text(d["url"])
                if full:
                    d["content"] = full

            if not embedded_chunks and not external_docs:
                print(f"Assistant: {FALLBACK_TEXT}")
                continue

            prompt = build_prompt(user_input, embedded_chunks, external_docs)

            response = None
            for attempt in range(3):
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=MAX_RESPONSE_TOKENS,
                        temperature=0.3,
                    )
                    break
                except Exception as e:
                    print(f"API error (attempt {attempt+1}): {e}")
                    time.sleep(1)

            if not response:
                print("Assistant: Sorry, I couldn't get a response.")
                continue

            full_answer = response.choices[0].message.content.strip()
            ans_text, sources = parse_sources(full_answer)

            print(f"\nAssistant: {ans_text}\n")
            print("📚 Sources:" if sources else "📚 Sources: None cited.")
            for title, url in sources:
                print(f"- [{title}]({url})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    chat()