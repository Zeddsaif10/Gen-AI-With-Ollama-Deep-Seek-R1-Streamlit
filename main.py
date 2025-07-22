import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
# Custom CSS styling
st.markdown("""
<style>
/* Global font and background */
html, body, .main {
    background: linear-gradient(145deg, #121212, #1e1e1e);
    font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
    color: #f1f1f1;
    font-size: 16px;
}

/* Smooth transitions */
* {
    transition: all 0.3s ease-in-out;
}

/* Sidebar styling with glass effect */
.css-1d391kg .sidebar-content {
    background: rgba(32, 32, 32, 0.7);
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.05);
}

/* Titles and headings */
h1, h2, h3 {
    color: #00f7ff;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.2rem;
    background: rgba(255, 255, 255, 0.05);
    border-left: 4px solid #00adb5;
}

.stChatMessage[data-testid="chat-message-user"] {
    border-left: 4px solid #22c1c3;
    background: rgba(34, 193, 195, 0.05);
}

.stChatMessage[data-testid="chat-message-ai"] {
    border-left: 4px solid #8fd3f4;
    background: rgba(143, 211, 244, 0.05);
}

/* Input box */
.stTextInput textarea, .stTextArea textarea {
    background-color: #2a2a2a !important;
    color: #ffffff !important;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 0.75rem;
}

/* Chat input box */
div[data-testid="stChatInput"] {
    background-color: #1c1c1c;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #333;
    box-shadow: 0 0 5px rgba(0, 255, 255, 0.2);
}

/* Select box */
.stSelectbox div[data-baseweb="select"] {
    background-color: #2b2b2b !important;
    color: #fff !important;
    border-radius: 8px;
    border: 1px solid #444 !important;
    padding: 6px;
}

.stSelectbox svg {
    fill: white !important;
}

div[role="listbox"] > div {
    background-color: #1e1e1e !important;
    color: white !important;
    padding: 8px;
    border-radius: 4px;
}

/* Spinner */
.css-1cpxqw2 {
    color: #00ffc8 !important;
}

/* Markdown links */
a {
    color: #00e0ff;
    text-decoration: none;
}
a:hover {
    color: #38f8ff;
    text-decoration: underline;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 1.5rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background-color: #555;
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover {
    background-color: #888;
}
</style>
""", unsafe_allow_html=True)
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# initiate the chat engine

llm_engine=ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",

    temperature=0.3

)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()