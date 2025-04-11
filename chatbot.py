import streamlit as st
import requests


def log_interaction(question, answer, file_path="./naive_qnalogs.txt"):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {answer}\n\n")


API_URL = "http://localhost:8010/query"  # Local NaiveRAG endpoint

st.set_page_config(page_title="NaiveRAG - HR Assistant", page_icon="🤖", layout="centered")

# --- Custom CSS for better UI ---
st.markdown("""
    <style>
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
            padding-right: 8px;
            margin-bottom: 1rem;
        }

        .stChatMessage {
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            line-height: 1.6;
            word-wrap: break-word;
            border-radius: 12px;
        }

        .user-msg-wrapper {
            display: flex;
            justify-content: flex-end;
        }

        .user-msg {
            background-color: #d6f5ff;
            color: #00334d;
            max-width: 60%;
            text-align: right;
        }

        .bot-msg-wrapper {
            display: flex;
            justify-content: flex-start;
        }

        .bot-msg {
            background-color: #f4f4f4;
            color: #1a1a1a;
            width: 100%;
        }

        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 0.75rem;
            border: 1px solid #888;
        }

        .stTextInput>div>div>input:focus {
            border-color: #555;
            outline: none;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 💼 HR Assistant - NaiveRAG")
st.markdown("Ask questions related to candidate resumes. This version uses a local RAG pipeline.")

# --- Chat Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

query_input = st.chat_input("Ask something about the candidate...", key="naive_query_input")

# Step 1: Handle new query
if query_input:
    st.session_state.chat_history.append(("user", query_input))
    st.session_state.chat_history.append(("bot", "🤖 Thinking..."))
    st.session_state.pending_query = query_input
    st.rerun()

# Step 2: Send query to local RAG API
if st.session_state.pending_query:
    payload = {
        "query": st.session_state.pending_query,
        "use_openai": False,
        "use_optimize_llm": True,
        "top_k": 100
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result["response"]
        log_interaction(st.session_state.pending_query, reply)
    except Exception as e:
        reply = f"❌ Error: {e}"

    # Replace the last "Thinking..." with the actual reply
    st.session_state.chat_history[-1] = ("bot", reply)
    st.session_state.pending_query = None
    st.rerun()

# --- Chat UI ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"""
        <div class="user-msg-wrapper">
            <div class="stChatMessage user-msg">
                <div style="display: flex; align-items: flex-start; justify-content: flex-end;">
                    <div style="margin-left: 8px;">👤</div>
                    <div style="max-width: 100%;">{message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-msg-wrapper">
            <div class="stChatMessage bot-msg">
                <div style="display: flex; align-items: flex-start;">
                    <div style="margin-right: 8px;">🤖</div>
                    <div style="max-width: 100%;">{message}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
