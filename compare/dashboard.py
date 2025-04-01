import streamlit as st
import os
import sys
import json

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compare.compare_runner import run_single_query_comparison, run_batch_query_comparison
from compare.utils import load_queries_from_json

st.set_page_config(page_title="RAG Comparison Dashboard", layout="wide")
st.title("📊 RAG Comparison Dashboard")
st.markdown("Compare responses between **OptimizeRag** and your **Local RAG** side by side.")

st.divider()

# --- Button to Run All Queries ---
if st.button("🚀 Start Comparison"):
    st.success("Running queries in background...")
    run_batch_query_comparison(load_queries_from_json())
    st.success("✅ All queries processed. Check result.json or result.xlsx.")

st.divider()

# --- Custom Query Section ---
st.subheader("🔎 Test a Custom Query")
custom_query = st.text_input("Enter your query and compare both RAGs")

if st.button("Run Custom Comparison") and custom_query.strip():
    result = run_single_query_comparison(custom_query, store=False)

    st.markdown(f"#### 🔍 Query: {custom_query}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🤖 OptimizeRAG Response")
        st.markdown(result["optimize_rag_response"], unsafe_allow_html=True)

    with col2:
        st.markdown("##### 🧪 Local RAG Response")
        st.markdown(result["local_rag_response"], unsafe_allow_html=True)

st.divider()

# --- Load and Display Past Results ---
st.subheader("📜 Past Comparison Logs")

result_path = os.path.join("compare", "result.json")
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for result in results[::-1]:
        with st.expander(f"🔍 {result['query'][:100]}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 🤖 OptimizeRAG Response")
                st.markdown(result["optimize_rag_response"], unsafe_allow_html=True)

            with col2:
                st.markdown("##### 🧪 Local RAG Response")
                st.markdown(result["local_rag_response"], unsafe_allow_html=True)
else:
    st.info("No past comparison results found.")
