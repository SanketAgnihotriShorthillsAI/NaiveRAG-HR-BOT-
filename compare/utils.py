import json
import openpyxl
from datetime import datetime

def load_queries_from_json(file_path="compare/queries.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_to_json_file(data, file_path="compare/result.json"):
    existing = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        pass
    existing.append(data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

def append_to_excel(data, file_path="compare/result.xlsx"):
    try:
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Query", "OptimizeRag Response", "Local RAG Response", "Timestamp"])

    ws.append([
        data["query"],
        data["optimize_rag_response"],
        data["local_rag_response"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])
    wb.save(file_path)
