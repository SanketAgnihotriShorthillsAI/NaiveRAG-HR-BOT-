import json

# === FILE PATHS ===
naive_file = "data/eval_results/naive_eval_v2.json"
optimize_file = "data/eval_results/optimize_eval_v2.json"
null_queries_file = "data/null_queries.json"
output_txt_file = "results_summary.txt"

# === INPUT TOP_K VALUES FROM USER ===
top_k_naive = input("Enter top_k value used for NaiveRAG: ")
top_k_optimize = input("Enter top_k value used for OptimizeRAG: ")

# === LOAD FILES ===
with open(naive_file, encoding="utf-8") as f:
    naive_data = json.load(f)

with open(optimize_file, encoding="utf-8") as f:
    optimize_data = json.load(f)

with open(null_queries_file, encoding="utf-8") as f:
    null_queries = json.load(f)

# === FILTER OUT NULL QUERIES ===
null_query_texts = {item["query_text"] for item in null_queries}

def filter_valid(data):
    return [entry for entry in data if entry["query"] not in null_query_texts]

naive_filtered = filter_valid(naive_data)
optimize_filtered = filter_valid(optimize_data)

# === AVERAGE CALCULATOR ===
def calculate_avg(filtered, key):
    scores = [
        entry[key]["score"]
        for entry in filtered
        if key in entry and isinstance(entry[key], dict)
    ]
    return round(sum(scores) / len(scores), 2) if scores else 0.0

# === CALCULATE METRICS ===
naive_recall = calculate_avg(naive_filtered, 'context_recall')
naive_precision = calculate_avg(naive_filtered, 'context_precision')
optimize_recall = calculate_avg(optimize_filtered, 'context_recall')
optimize_precision = calculate_avg(optimize_filtered, 'context_precision')
evaluated_count = len(naive_filtered)

# === FORMAT OUTPUT ===
summary = f"""
=== Evaluation Summary ===
NaiveRAG:     top_k = {top_k_naive}
OptimizeRAG:  top_k = {top_k_optimize}

NaiveRAG Recall:     {naive_recall}
NaiveRAG Precision:  {naive_precision}
OptimizeRAG Recall:     {optimize_recall}
OptimizeRAG Precision:  {optimize_precision}

Total Queries Evaluated: {evaluated_count} (after excluding null queries)
"""

# === PRINT TO TERMINAL AND FILE ===
print(summary)

with open(output_txt_file, "a", encoding="utf-8") as f:
    f.write(summary.strip())
    f.write("\n")

print(f"📄 Summary written to: {output_txt_file}")
