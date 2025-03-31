import os
import sys
import streamlit as st

# Setup Python path to import pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pipeline.pipeline import RAGPipeline

# --- Config ---
st.set_page_config(page_title="HR Resume Bot", layout="wide")
st.title("💼 HR Assistant RAG Bot")
st.caption("Ask questions about candidate resumes. Responses are generated using retrieved resume chunks.")

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

# --- Sidebar: Model Selector ---
st.sidebar.header("🔧 Configuration")
model_choice = st.sidebar.radio("Choose LLM", ["Gemini", "OpenAI", "OptimizeRag LLM"])

use_openai = model_choice == "OpenAI"
use_optimize_llm = model_choice == "OptimizeRag LLM"

top_k = st.sidebar.slider("Top K Chunks", min_value=3, max_value=600, value=60)

if st.sidebar.button("🔄 Reload Pipeline"):
    st.session_state.rag_pipeline = RAGPipeline(
        use_openai=use_openai,
        use_optimize_llm=use_optimize_llm
    )
    st.success("Pipeline reloaded!")

# Load pipeline if not already loaded
if st.session_state.rag_pipeline is None:
    st.session_state.rag_pipeline = RAGPipeline(
        use_openai=use_openai,
        use_optimize_llm=use_optimize_llm
    )

# --- Chat UI ---
with st.form("query_form"):
    user_query = st.text_input("💬 Ask a question:")
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Retrieving & Generating..."):
        chunks = st.session_state.rag_pipeline.retriever.query(user_query, top_k=top_k)
        answer, response = st.session_state.rag_pipeline.generator.generate_response(user_query, chunks)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Display Chat Messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑‍💼 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg['content']}")
