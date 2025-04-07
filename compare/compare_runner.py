import requests
from compare.utils import append_to_json_file, append_to_excel

OPTIMIZE_RAG_API = "http://localhost:8000/api/v1/query"
LOCAL_RAG_API = "http://localhost:8010/query"

def query_optimize_rag(query):
    payload = {
        "query": query,
        "llm_provider": "azure_openai",
        "llm_model_name": "gpt-4o-mini",
        "filters": {
            "category": "general",
            "mode": "hybrid",
            "prompt_type": "hr_assistance",
            "graph_algorithm": ""
        },
        "vector_storage": "NanoVectorDBStorage",
        "graph_storage": "NetworkXStorage"
    }
    try:
        res = requests.post(OPTIMIZE_RAG_API, json=payload, timeout=120)
        return res.json()["data"]["response"]
    except Exception as e:
        return f"❌ Error from OptimizeRag: {e}"

def query_local_rag(query):
    payload = {
        "query": query,
        "top_k": 100,
        "use_openai": False,
        "use_optimize_llm": True
    }
    try:
        res = requests.post(LOCAL_RAG_API, json=payload, timeout=120)
        return res.json()["response"]
    except Exception as e:
        return f"❌ Error from Local RAG: {e}"

def run_single_query_comparison(query, store=True):
    print(f"\n🔍 Query: {query}")
    print("⏳ Querying OptimizeRag...")
    optimize_rag_response = query_optimize_rag(query)
    print("✅ Got response from OptimizeRag.")

    print("⏳ Querying Local RAG...")
    local_rag_response = query_local_rag(query)
    print("✅ Got response from Local RAG.")

    result = {
        "query": query,
        "optimize_rag_response": optimize_rag_response,
        "local_rag_response": local_rag_response
    }

    if store:
        append_to_json_file(result)
        append_to_excel(result)

    return result

def run_batch_query_comparison(queries):
    for query in queries:
        run_single_query_comparison(query)
