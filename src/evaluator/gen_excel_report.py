import os
import json
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
# Toggle flags for which columns to include in the Excel output
INCLUDE_QUERY = True
INCLUDE_GENERATED = True            # from query log file ("response")
INCLUDE_CONTEXT_RECALL = True       # score, reason (raw_response will be omitted)
INCLUDE_CONTEXT_PRECISION = True    # score, reason (raw_response will be omitted)
INCLUDE_FAITHFULNESS = True         # score, reason (raw_response will be omitted)
INCLUDE_LOG_CONTEXT = False         # if you need the log context

# File paths (adjust these as needed)
NAIVE_EVAL_FILE = Path("data/eval_results/naive_eval_v2.json")
OPTIMIZE_EVAL_FILE = Path("data/eval_results/optimize_eval_v2.json")
QUERY_LOG_FILE = Path("compare/query_logs/naiveRag_query_logs.json")  # adjust if needed
NULL_QUERIES_FILE = Path("data/null_queries.json")  # file containing queries to exclude

# === FUNCTIONS ===
def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def flatten_evaluation(evals: list, prefix: str):
    records = []
    for item in evals:
        record = {"query": item.get("query", "")}
        if "response" in item:
            record[f"generated_answer_{prefix}"] = item.get("response", "")
        cr = item.get("context_recall", {})
        record[f"context_recall_score_{prefix}"] = cr.get("score", "")
        record[f"context_recall_reason_{prefix}"] = cr.get("reason", "")
        cp = item.get("context_precision", {})
        record[f"context_precision_score_{prefix}"] = cp.get("score", "")
        record[f"context_precision_reason_{prefix}"] = cp.get("reason", "")
        fth = item.get("faithfulness", {})
        record[f"faithfulness_score_{prefix}"] = fth.get("score", "")
        record[f"faithfulness_reason_{prefix}"] = fth.get("reason", "")
        records.append(record)
    return records

def flatten_query_log(logs: list):
    records = []
    for item in logs:
        record = {
            "query": item.get("query", ""),
            "generated_answer_log": item.get("response", ""),
            "log_context": item.get("context", "")
        }
        records.append(record)
    return records

def merge_evaluation_data(naive_df: pd.DataFrame, optimize_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(naive_df, optimize_df, on="query", how="outer", suffixes=("_naive", "_optimize"))

def merge_with_query_log(eval_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(eval_df, log_df, on="query", how="outer")

def select_columns(df: pd.DataFrame, include: dict) -> pd.DataFrame:
    cols = []
    if include.get("query", False):
        cols.append("query")
    if include.get("generated_answer", False):
        for prefix in ["naive", "optimize"]:
            col = f"generated_answer_{prefix}"
            if col in df.columns:
                cols.append(col)
    if include.get("context_recall", False):
        for prefix in ["naive", "optimize"]:
            cols.extend([f"context_recall_score_{prefix}", f"context_recall_reason_{prefix}"])
    if include.get("context_precision", False):
        for prefix in ["naive", "optimize"]:
            cols.extend([f"context_precision_score_{prefix}", f"context_precision_reason_{prefix}"])
    if include.get("faithfulness", False):
        for prefix in ["naive", "optimize"]:
            cols.extend([f"faithfulness_score_{prefix}", f"faithfulness_reason_{prefix}"])
    if include.get("log_context", False) and "log_context" in df.columns:
        cols.append("log_context")
    return df[cols]

def load_null_queries(path: Path):
    null_data = load_json(path)
    return {entry["query_text"] for entry in null_data}

def export_to_excel(merged_df: pd.DataFrame, output_path: str):
    merged_df.to_excel(output_path, index=False)
    print(f"✅ Excel file generated: {output_path}")

# === MAIN PROCESSING ===
def main():
    naive_eval_data = load_json(NAIVE_EVAL_FILE)
    optimize_eval_data = load_json(OPTIMIZE_EVAL_FILE)
    query_log_data = load_json(QUERY_LOG_FILE)
    null_query_texts = load_null_queries(NULL_QUERIES_FILE)

    naive_records = flatten_evaluation(naive_eval_data, "naive")
    optimize_records = flatten_evaluation(optimize_eval_data, "optimize")
    naive_df = pd.DataFrame(naive_records)
    optimize_df = pd.DataFrame(optimize_records)

    eval_merged_df = merge_evaluation_data(naive_df, optimize_df)

    log_records = flatten_query_log(query_log_data)
    log_df = pd.DataFrame(log_records)
    final_merged_df = merge_with_query_log(eval_merged_df, log_df)

    # 🚫 Remove rows where query is in the null queries list
    final_merged_df = final_merged_df[~final_merged_df["query"].isin(null_query_texts)]

    include_flags = {
        "query": True,
        "generated_answer": INCLUDE_GENERATED,
        "context_recall": INCLUDE_CONTEXT_RECALL,
        "context_precision": INCLUDE_CONTEXT_PRECISION,
        "faithfulness": INCLUDE_FAITHFULNESS,
        "log_context": INCLUDE_LOG_CONTEXT,
    }

    final_df = select_columns(final_merged_df, include_flags)

    output_path = "evaluation_results.xlsx"
    export_to_excel(final_df, output_path)

if __name__ == "__main__":
    main()
