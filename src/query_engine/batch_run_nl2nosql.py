import os
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from nl2noSql_query import extract_keywords, build_mongo_query
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
import time
# === Setup ===
load_dotenv()
mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]

def load_test_queries(path="data/test_queries.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def process_query(query: str, use_gemini: bool):
    result_log = {
        "query": query,
        "keywords": [],
        "matched_resumes": []
    }

    # 1. Extract Keywords
    keywords = await extract_keywords(query, use_gemini=use_gemini)
    result_log["keywords"] = keywords
    if not keywords:
        return result_log  # Skip empty extraction

    # 2. Build and Run Mongo Query
    mongo_query = build_mongo_query(keywords)
    matched_docs = list(collection.find(mongo_query))

    # 3. Store name/email only
    for doc in matched_docs:
        result_log["matched_resumes"].append({
            "name": doc.get("name", "Not Provided"),
            "email": doc.get("email", "Not Provided")
        })

    return result_log

async def main(use_gemini: bool):
    test_queries = load_test_queries()
    print(f"📄 Loaded {len(test_queries)} queries.")

    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n⚙️ [{i}/{len(test_queries)}] Processing: {query}")
        if use_gemini:
            time.sleep(2)  # Avoid Gemini rate limits
        result = await process_query(query, use_gemini=use_gemini)
        results.append(result)

    output_path = "data/nosql_retrieval_logs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Finished. Log saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini for keyword extraction")
    args = parser.parse_args()

    asyncio.run(main(use_gemini=args.use_gemini))
