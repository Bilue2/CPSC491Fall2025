# chroma_chat_app_v3.py
# Run: streamlit run chroma_chat_app_v3.py

import os, time, datetime, hashlib, logging, io
from uuid import uuid4
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from serpapi import GoogleSearch
from chromadb import Client as ChromaClient
from PyPDF2 import PdfReader
import chromadb

# ------------------- Page Config -------------------
st.set_page_config(page_title="Regulatory AI Assistant", page_icon="📘", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- CSS for sticky header/footer -------------------
st.markdown("""
<style>
/* Sticky header */
.header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background:white;
    z-index:100;
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 20px;
    border-bottom:1px solid #ddd;
}
.app-title {color:#002855; font-weight:700; font-size:28px; margin:0;}
.logout-btn {background:#002855;color:white;border-radius:8px;padding:6px 12px;border:none;}
.logout-btn:hover {background:#003D99;}

/* Scrollable chat area */
.chat-container {
    position: absolute;
    top: 70px; /* height of header */
    bottom: 90px; /* height of footer */
    left: 0;
    right: 0;
    overflow-y:auto;
    padding:10px 20px;
    display:flex;
    flex-direction:column;
    gap:6px;
}

/* Chat bubbles */
.chat-row {display:flex; align-items:flex-start;}
.chat-row.user {justify-content:flex-end;}
.chat-row.assistant {justify-content:flex-start;}
.user-bubble {background:#E6EEF7;color:#002855;padding:12px 14px;border-radius:14px;margin:4px 0;max-width:78%;font-size:16px;word-break:break-word;box-shadow:0 1px 3px rgba(0,0,0,0.06);}
.assistant-bubble {background:linear-gradient(180deg,#002855,#003D7A);color:#ffffff;padding:12px 14px;border-radius:14px;margin:4px 0;max-width:78%;font-size:16px;word-break:break-word;box-shadow:0 2px 6px rgba(0,0,0,0.12);}
.meta {font-size:12px;color:#7a869a;margin-top:4px;}
.highlight {background-color: #FFF176; padding: 2px 4px; border-radius:3px;}

/* Collapsible long messages */
.collapsible {cursor:pointer;}
.collapsible-content {display:none; white-space: pre-wrap;}

/* Fixed footer input */
.footer {
    position: fixed;
    bottom:0;
    left:0;
    right:0;
    background:white;
    border-top:1px solid #ddd;
    z-index:100;
    padding:10px 20px;
    display:flex;
    flex-direction:column;
    gap:5px;
}
.input-row {display:flex; gap:5px; align-items:center;}
.input-text {flex-grow:1; padding:10px; border-radius:8px; border:1px solid #ccc; resize:none;}
.upload-btn {background:#E6EEF7;color:#002855;border:none;border-radius:8px;padding:6px 10px; font-size:16px;}
.send-btn {background:#002855;color:white;border:none;border-radius:8px;padding:8px 14px;font-size:16px;}

/* Uploaded files display */
.uploaded-files {font-size:14px;color:#555;}

/* Responsive for mobile */
@media(max-width:600px){
    .input-row {flex-direction:column;}
    .input-text {width:100%;}
    .chat-container {padding:5px 10px;}
}
</style>

<script>
function toggleMessage(id){
    var content = document.getElementById(id);
    if(content.style.display==="none"){content.style.display="block";}
    else{content.style.display="none";}
}
</script>
""", unsafe_allow_html=True)

# ------------------- Secrets -------------------
missing = []
try:
    APP_USER = st.secrets["APP_USERNAME"]
    APP_PASS = st.secrets["APP_PASSWORD"]
except KeyError:
    missing.append("APP_USERNAME / APP_PASSWORD")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY")
if not SERPAPI_API_KEY: missing.append("SERPAPI_API_KEY")
CHROMA_API_KEY = st.secrets.get("CHROMA_API_KEY")
CHROMA_TENANT = st.secrets.get("CHROMA_TENANT")
CHROMA_DATABASE = st.secrets.get("CHROMA_DATABASE")
COLLECTION_NAME = st.secrets.get("CHROMA_COLLECTION", "fcc_documents")

if missing:
    st.error("Missing required secrets: " + ", ".join(missing))
    st.stop()

# ------------------- Constants -------------------
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
MIN_INGEST_LENGTH = 300

# ------------------- Initialize clients -------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
try:
    client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
except Exception as e:
    st.error(f"Chroma init error: {e}")
    logger.exception("Chroma init error")
    st.stop()

# ------------------- Session State -------------------
if "authenticated" not in st.session_state: st.session_state.authenticated=False
if "messages" not in st.session_state: st.session_state.messages=[]
if "uploaded_files" not in st.session_state: st.session_state.uploaded_files=[]

# ------------------- Authentication -------------------
def login_screen():
    st.title("🔐 Regulatory Assistant Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            if username==APP_USER and password==APP_PASS:
                st.session_state.authenticated=True
                st.success("Logged in — loading assistant...")
                time.sleep(0.5)
                st.experimental_rerun()
            else: st.error("Invalid username or password.")
    with col2:
        if st.button("Exit"): st.stop()

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# ------------------- Sticky Header -------------------
st.markdown(f"""
<div class="header">
    <div class="app-title">📘 Regulatory AI Assistant</div>
    <form method="post" action="">
        <button class="logout-btn" name="logout">Logout</button>
    </form>
</div>
""", unsafe_allow_html=True)

if st.experimental_get_query_params().get("logout"): 
    st.session_state.authenticated=False
    st.experimental_rerun()

# ------------------- Helper Functions -------------------
def compute_hash(text:str)->str: return hashlib.sha256(text.encode("utf-8")).hexdigest()
def document_exists(content_hash:str, url:Optional[str]="")->bool:
    try:
        all_meta=collection.get(include=["metadatas","ids"])
        metas=all_meta.get("metadatas",[])
        for meta in metas:
            if not isinstance(meta, dict): continue
            if meta.get("hash")==content_hash or (url and meta.get("source")==url): return True
        return False
    except Exception as e:
        logger.exception(e)
        return False

def embed_text(text:str)->List[float]:
    resp=openai_client.embeddings.create(model=EMBED_MODEL,input=text)
    return resp.data[0].embedding

def ingest_document(title:str, url:str, content:str)->bool:
    if not content or len(content)<MIN_INGEST_LENGTH: return False
    content_hash=compute_hash(content)
    if document_exists(content_hash,url): return False
    try:
        embedding=embed_text(content)
        collection.add(ids=[str(uuid4())],documents=[content],embeddings=[embedding],
                       metadatas={"title":title,"source":url,"hash":content_hash,"retrieved":str(datetime.date.today())})
        st.session_state.uploaded_files.append(title)
        return True
    except Exception as e:
        logger.exception(e)
        return False

def external_search(query:str,max_results:int=5)->List[Dict]:
    params={"q":query,"engine":"google","api_key":SERPAPI_API_KEY,"num":max_results}
    try:
        results=GoogleSearch(params).get_dict()
        return [{"title":r.get("title","Untitled"),"url":r.get("link",""),"content":r.get("snippet","")} for r in results.get("organic_results",[])]
    except Exception as e:
        logger.exception(e)
        return []

def fetch_full_text(url:str)->str:
    try:
        resp=requests.get(url,timeout=10); resp.raise_for_status()
        soup=BeautifulSoup(resp.text,"html.parser")
        return "\n\n".join(p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip())
    except: return ""

def retrieve_relevant_chunks(query:str, top_k:int=SIMILARITY_TOP_K)->List[Dict]:
    q_emb=embed_text(query)
    results=collection.query(query_embeddings=[q_emb],n_results=top_k,include=["documents","metadatas"])
    docs=results.get("documents",[[]])[0]
    metas=results.get("metadatas",[[]])[0]
    return [{"document":d,"metadata":m} for d,m in zip(docs,metas)]

def build_prompt(query:str, embedded_chunks:List[Dict], external_docs:List[Dict])->str:
    system_instructions = (
            "You are an expert on emergency alert systems (EAS, WEA, IPAWS), public safety communications, and regulatory frameworks."
            "Provide specific, detailed answers using the context below.\n\n"
            "Guidelines:\n"
            "- Include specific details: dates, names, statistics, and technical terms like (EAS, WEA, IPAWS, CAP, FCC Part 11 and more)\n"
            '- Do not fabricate sources. Use markdown links for citations under \'Sources:\'.'
            "- Provide examples and context when helpful\n"
            "- If context is insufficient, supplement with your knowledge but indicate this clearly"
        )
    parts=[]
    for i,chunk in enumerate(embedded_chunks):
        title=chunk.get("metadata",{}).get("title",f"doc-{i}")
        text=chunk.get("document","")[:1500]
        parts.append(f"EMBEDDED: {title}\n{text}")
    for d in external_docs:
        parts.append(f"EXTERNAL: {d.get('title','External')} (URL: {d.get('url','')})\n{d.get('content','')[:1500]}")
    context_text="\n\n---\n\n".join(parts) if parts else "No context documents available."
    return f"{system_instructions}\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer (with markdown citations under 'Sources:'):"

def parse_sources(answer:str)->Tuple[str,List[Tuple[str,str]]]:
    marker="\nSources:"
    if marker in answer:
        ans_part, src_part=answer.split(marker,1)
        sources=[]
        for line in src_part.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try: t,u=line.split("[",1)[1].split("]")[0], line.split("(",1)[1].split(")")[0]; sources.append((t,u))
                except: continue
        return ans_part.strip(), sources
    return answer.strip(), []

def highlight_keywords(text:str,keywords:List[str])->str:
    for kw in keywords:
        text=text.replace(kw,f'<span class="highlight">{kw}</span>')
    return text

# ------------------- Chat Messages -------------------
chat_container = st.container()
with chat_container:
    for i,msg in enumerate(st.session_state.messages):
        role=msg["role"]; ts=msg.get("time","")
        bubble_class="user-bubble" if role=="user" else "assistant-bubble"
        row_class="chat-row user" if role=="user" else "chat-row assistant"
        text=msg["text"]
        if role=="assistant" and len(text)>600:
            msg_id=f"msg-{i}"
            text_display=f'<div class="collapsible" onclick="toggleMessage(\'{msg_id}\')">Show/Hide answer...</div><div id="{msg_id}" class="collapsible-content">{text}</div>'
        else: text_display=text
        st.markdown(f'<div class="{row_class}"><div class="{bubble_class}">{text_display}<div class="meta">{ts}</div></div></div>', unsafe_allow_html=True)

# ------------------- Fixed Footer Input -------------------
st.markdown('<div class="footer">', unsafe_allow_html=True)
col_files = st.empty()
col_input = st.container()
with col_input:
    uploaded_file = st.file_uploader("", type=["txt","pdf"], label_visibility="collapsed")
    user_prompt = st.text_area("Type your question here...", key="chat_input", height=50, label_visibility="collapsed")
    col1, col2 = st.columns([1,1])
    if col2.button("Send"):
        if user_prompt.strip():
            now=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            st.session_state.messages.append({"role":"user","text":user_prompt,"time":now})
            st.experimental_rerun()

# Display uploaded files
if st.session_state.uploaded_files:
    col_files.info("Uploaded files: " + ", ".join(st.session_state.uploaded_files))
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Process Upload -------------------
if uploaded_file:
    try:
        if uploaded_file.type=="application/pdf":
            reader=PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text="\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text=uploaded_file.getvalue().decode("utf-8", errors="ignore")
        fake_url=f"uploaded://{uploaded_file.name}"
        added=ingest_document(uploaded_file.name,fake_url,text)
        st.success("Document ingested." if added else "Skipped (duplicate/too short)")
    except Exception as e:
        st.error(f"Upload failed: {e}")

# ------------------- Process Latest User Message -------------------
def process_latest():
    if not st.session_state.messages: return
    last=st.session_state.messages[-1]
    if last["role"]!="user": return
    query=last["text"]
    with st.spinner("Retrieving context..."):
        embedded=retrieve_relevant_chunks(query)
        external_docs=[]
        if not embedded:
            external_docs=external_search(query)
            for d in external_docs:
                full=fetch_full_text(d.get("url","")) or d.get("content","")
                d["content"]=full
                ingest_document(d.get("title","External"), d.get("url",""), full)
            embedded=retrieve_relevant_chunks(query)
        prompt_text=build_prompt(query,embedded,external_docs)
    with st.spinner("Generating answer..."):
        try:
            resp=openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":prompt_text}],
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.2
            )
            ans_full=resp.choices[0].message.content.strip()
        except Exception as e: ans_full=f"Error generating response: {e}"
    ans_text, sources=parse_sources(ans_full)
    if sources: ans_text+="\n\n**Sources:**\n" + "\n".join(f"- [{t}]({u})" for t,u in sources)
    keywords=query.split()
    ans_text=highlight_keywords(ans_text,keywords)
    st.session_state.messages.append({"role":"assistant","text":ans_text,"time":datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")})
    st.experimental_rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"]=="user":
    process_latest()
