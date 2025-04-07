import json
import pandas as pd
from pathlib import Path

# Paths to input files
base_path = Path("compare/evaluated_logs")

opt_path = base_path / "optimize_evaluated.json"
naive_path = base_path / "naive_evaluated.json"
compare_path = base_path / "head_to_head_comparison.json"

# Load JSONs
with open(opt_path, "r", encoding="utf-8") as f:
    opt_data = {e["query"]: e for e in json.load(f)}
with open(naive_path, "r", encoding="utf-8") as f:
    naive_data = {e["query"]: e for e in json.load(f)}
with open(compare_path, "r", encoding="utf-8") as f:
    compare_data = {e["query"]: e for e in json.load(f)}

# Merge by query
all_queries = set(opt_data) | set(naive_data) | set(compare_data)
rows = []
for query in all_queries:
    o = opt_data.get(query, {})
    n = naive_data.get(query, {})
    c = compare_data.get(query, {})

    rows.append({
        "Query": query,
        "OptimizeRAG_Response": o.get("response", ""),
        "OptimizeRAG_Score": o.get("faithfulness_score", ""),
        "OptimizeRAG_Explanation": o.get("llm_explanation", ""),
        "NaiveRAG_Response": n.get("response", ""),
        "NaiveRAG_Score": n.get("faithfulness_score", ""),
        "NaiveRAG_Explanation": n.get("llm_explanation", ""),
        "LLM_Comparison_Reasoning": c.get("llm_reasoning", "")
    })

# Save to Excel
df = pd.DataFrame(rows)
df.to_excel("compare/evaluated_logs/rag_evaluation_combined.xlsx", index=False)
print("✅ Excel file saved: rag_evaluation_combined.xlsx")
