import json
from pathlib import Path

# Input and output paths
INPUT_JSON = Path("data/gold_query_matrix.json")
OUTPUT_JSON = Path("data/golden_answers_compiled.json")

def load_gold_matrix(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_answers_with_source(matrix):
    query_to_answers = {}

    for entry in matrix:
        query_text = entry["query_text"]
        results = entry["results"]

        collected = []
        for resume_file, answer in results.items():
            if answer is not None and answer != "null":
                collected.append({
                    "resume": resume_file,
                    "answer": answer
                })

        query_to_answers[query_text] = collected

    return query_to_answers

def save_aggregated_answers(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved aggregated answers (with source) to: {output_path}")

def main():
    print("📥 Loading gold query matrix...")
    matrix = load_gold_matrix(INPUT_JSON)

    print(f"🔍 Processing {len(matrix)} queries...")
    aggregated = extract_answers_with_source(matrix)

    save_aggregated_answers(aggregated, OUTPUT_JSON)

if __name__ == "__main__":
    main()
