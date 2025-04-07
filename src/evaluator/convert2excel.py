import json
import pandas as pd
from pathlib import Path

# Input files
BASE_DIR = Path("data/eval_results/")
OPTIMIZE_PATH = BASE_DIR / "optimize_evaluations.json"
NAIVE_PATH = BASE_DIR / "naive_evaluations.json"
OUTPUT_EXCEL = BASE_DIR / "eval_comparison.xlsx"

# Load JSON data
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Flatten results
def flatten_results(data: dict, pipeline: str):
    flat = []
    for eval_type, entries in data.items():
        for entry in entries:
            flat.append({
                "query": entry.get("query", "").strip(),
                "evaluation_type": eval_type,
                f"{pipeline}_score": entry.get("score"),
                f"{pipeline}_missing_info": entry.get("missing_info"),
                f"{pipeline}_noise": entry.get("noise"),
                f"{pipeline}_percent_matched": entry.get("percent_matched"),
            })
    return flat

# Merge both pipelines side-by-side
def merge_results(optimize_data, naive_data):
    df_opt = pd.DataFrame(flatten_results(optimize_data, "optimize"))
    df_naive = pd.DataFrame(flatten_results(naive_data, "naive"))

    merged = pd.merge(df_opt, df_naive, on=["query", "evaluation_type"], how="outer")
    return merged

# Save to Excel
def save_excel(df, output_path):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Evaluation Comparison")

        # Formatting
        worksheet = writer.sheets["Evaluation Comparison"]
        worksheet.set_column("A:A", 60)  # query
        worksheet.set_column("B:B", 18)  # eval type
        worksheet.set_column("C:J", 30)  # metrics

    print(f"✅ Excel file saved to: {output_path}")

def main():
    print("📥 Loading evaluation results...")
    optimize_data = load_json(OPTIMIZE_PATH)
    naive_data = load_json(NAIVE_PATH)

    print("🔄 Merging results...")
    df = merge_results(optimize_data, naive_data)

    print("📤 Writing to Excel...")
    save_excel(df, OUTPUT_EXCEL)

if __name__ == "__main__":
    main()
