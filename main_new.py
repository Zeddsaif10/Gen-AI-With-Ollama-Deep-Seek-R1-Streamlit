import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS, Qdrant
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
from langchain_experimental.tools import PythonREPLTool
import streamlit_authenticator as stauth
import io
import contextlib
import os
import datetime

# ----------------- AUTH -------------------
names = ["Alice", "Bob"]
usernames = ["alice123", "bob456"]

# Current correct way to hash passwords:
try:
    # Try the newest method first
    hasher = stauth.Hasher(['pass123', 'pass456'])
    passwords = hasher.generate()
except AttributeError:
    # Fallback to manual hashing if needed
    passwords = [stauth.Hasher._hash_password(p) for p in ['pass123', 'pass456']]

# --- Authenticator Setup ---
authenticator = stauth.Authenticate(
    credentials={
        "usernames": {
            "alice123": {
                "name": "Alice",
                "password": passwords[0]
            },
            "bob456": {
                "name": "Bob",
                "password": passwords[1]
            }
        }
    },
    cookie_name="code_app",
    key="abcdef",
    cookie_expiry_days=30
)

# --- Login Widget ---
name, auth_status, username = authenticator.login("Login", "main")
if not auth_status:
    st.stop()
st.success(f"Welcome {name} 👋")

# ----------------- SIDEBAR EXTENSIONS -------------------
with st.sidebar:
    if st.button("📁 Download Chat Log"):
        if 'message_log' in st.session_state:
            log = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.message_log])
            fname = f"chat_{username}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button("Download", log, file_name=fname)


# ----------------- DEBUGGING -------------------
if st.checkbox("🛠 Debug Mode"):
    user_bug_code = st.text_area("Paste buggy code here")
    if user_bug_code:
        debug_prompt = f"""Find bugs in this Python code and suggest corrected version:
```python
{user_bug_code}
```"""
        debugged = ChatOllama(
            model="deepseek-r1:3b",
            base_url="http://localhost:11434",
            temperature=0.3
        ).invoke(debug_prompt)
        st.code(debugged, language="python")

# ----------------- DOCUMENTATION GENERATOR -------------------
def generate_docstring(code_snippet):
    doc_prompt = f"""Analyze the following Python code and generate detailed docstrings for all functions and classes:
```python
{code_snippet}
```"""
    return ChatOllama(
        model="deepseek-r1:3b",
        base_url="http://localhost:11434",
        temperature=0.3
    ).invoke(doc_prompt)

if st.checkbox("🧾 Auto Doc Generator"):
    user_code = st.text_area("Paste your Python code here")
    if user_code:
        docs = generate_docstring(user_code)
        st.code(docs, language="markdown")

# ----------------- AGENT TOOL -------------------
tools = [
    Tool(
        name="python_repl",
        func=PythonREPLTool().run,
        description="Executes Python code"
    )
]

agent = initialize_agent(
    tools,
    ChatOllama(
        model="deepseek-r1:3b",
        base_url="http://localhost:11434",
        temperature=0.3
    ),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if st.checkbox("🤖 Use AI Agent"):
    agent_query = st.text_input("Ask agent to run a task")
    if agent_query:
        result = agent.run(agent_query)
        st.markdown(result)

# ----------------- MULTI-PDF RAG -------------------
st.markdown("### 📄 Upload and Chat with PDFs")
rag_docs = st.file_uploader("Upload multiple PDFs", type=["pdf"], accept_multiple_files=True)
if rag_docs:
    all_chunks = []
    for doc in rag_docs:
        all_chunks.extend(PyPDFLoader(doc).load_and_split())
    vectorstore_pdf = Qdrant.from_documents(
        all_chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        location=":memory:",
        collection_name="rag_pdfs"
    )
    qa_chain_pdf = RetrievalQA.from_chain_type(llm=ChatOllama(
        model="deepseek-r1:3b",
        base_url="http://localhost:11434",
        temperature=0.3
    ), retriever=vectorstore_pdf.as_retriever())
    pdf_query = st.text_input("Ask a question about uploaded PDFs")
    if pdf_query:
        response = qa_chain_pdf.run(pdf_query)
        st.markdown(response)
