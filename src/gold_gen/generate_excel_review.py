import json
import pandas as pd
from pathlib import Path

# Input and output paths
INPUT_JSON = Path("data/gold_query_matrix.json")
OUTPUT_EXCEL = Path("data/golden_answers_review.xlsx")
NULL_QUERIES_LOG = Path("data/null_queries.json")

def load_gold_matrix(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def is_valid_answer(answer):
    return answer not in (None, "null", "Null", "NULL", {}, [])

def compile_golden_answers(matrix):
    """
    Returns two things:
    1. compiled_answers: {query_text: list of {filename, answer}} for non-null answers
    2. null_queries: list of queries with no valid answers
    """
    compiled = {}
    null_queries = []

    for query_entry in matrix:
        query_text = query_entry["query_text"]
        query_id = query_entry["query_id"]
        results = query_entry["results"]

        valid_entries = []
        for filename, answer in results.items():
            if is_valid_answer(answer):
                valid_entries.append({
                    "resume_file": filename,
                    "answer": answer
                })

        if valid_entries:
            compiled[query_text] = valid_entries
        else:
            null_queries.append({
                "query_id": query_id,
                "query_text": query_text
            })

    return compiled, null_queries

def save_to_excel(compiled_answers, output_path):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for i, (query_text, entries) in enumerate(compiled_answers.items()):
            # Shorten query text for sheet name
            sheet_name = f"Query_{i+1}"
            df = pd.DataFrame(entries)

            # Convert dict-type answers to JSON strings
            df["answer"] = df["answer"].apply(
                lambda x: json.dumps(x, ensure_ascii=False, indent=2) if isinstance(x, (dict, list)) else str(x)
            )

            df.to_excel(writer, index=False, sheet_name=sheet_name)

            # Optional: Format columns
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 45)  # Resume file column
            worksheet.set_column(1, 1, 80)  # Answer column

    print(f"✅ Excel file saved to: {output_path}")

def save_json(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    print("📥 Loading matrix...")
    matrix = load_gold_matrix(INPUT_JSON)
    compiled, null_queries = compile_golden_answers(matrix)

    print(f"📊 Compiled {len(compiled)} query sheets.")
    save_to_excel(compiled, OUTPUT_EXCEL)

    save_json(null_queries, NULL_QUERIES_LOG)
    print(f"📄 Null-only queries logged to: {NULL_QUERIES_LOG}")

if __name__ == "__main__":
    main()
