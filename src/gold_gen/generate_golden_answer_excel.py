import json
import pandas as pd
from pathlib import Path

# Input and output paths
INPUT_JSON = Path("data/gold_query_matrix.json")
OUTPUT_EXCEL = Path("data/gold_query_matrix.xlsx")

def load_gold_matrix(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_to_dataframe(matrix):
    # Step 1: Collect all unique resume filenames
    all_resume_files = set()
    for entry in matrix:
        all_resume_files.update(entry["results"].keys())

    sorted_resumes = sorted(all_resume_files)

    # Step 2: Create table rows
    table = []
    for entry in matrix:
        row = {
            "query_id": entry["query_id"],
            "query_text": entry["query_text"],
        }
        for resume in sorted_resumes:
            row[resume] = entry["results"].get(resume, None)
        table.append(row)

    df = pd.DataFrame(table)
    return df, sorted_resumes

def save_to_excel(df, sorted_resumes, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="GoldMatrix")
        workbook = writer.book
        worksheet = writer.sheets["GoldMatrix"]

        # Define styles
        null_format = workbook.add_format({"bg_color": "#F0F0F0", "font_color": "#888888"})
        match_format = workbook.add_format({"bg_color": "#E2F7E2"})
        header_format = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        query_text_format = workbook.add_format({"text_wrap": True, "valign": "top"})

        # Apply header formatting
        for col_idx, value in enumerate(df.columns):
            worksheet.write(0, col_idx, value, header_format)
            worksheet.set_column(col_idx, col_idx, 25)

        # Set wider column for query_text
        worksheet.set_column(1, 1, 60, query_text_format)

        # Apply cell formatting
        for row_idx in range(1, len(df) + 1):
            for col_idx in range(2, len(df.columns)):  # Resume columns only
                cell_value = df.iloc[row_idx - 1, col_idx]
                if cell_value is None or (isinstance(cell_value, float) and pd.isna(cell_value)):
                    worksheet.write(row_idx, col_idx, None, null_format)
                elif isinstance(cell_value, (dict, list)):
                    worksheet.write(row_idx, col_idx, json.dumps(cell_value, ensure_ascii=False), match_format)
                else:
                    worksheet.write(row_idx, col_idx, str(cell_value), match_format)

    print(f"✅ Excel file with styles saved to: {output_path}")

def main():
    print("📥 Loading gold matrix JSON...")
    matrix = load_gold_matrix(INPUT_JSON)
    print(f"🧮 Loaded {len(matrix)} queries.")

    df, sorted_resumes = convert_to_dataframe(matrix)
    print(f"📊 Generated DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")

    save_to_excel(df, sorted_resumes, OUTPUT_EXCEL)

if __name__ == "__main__":
    main()
