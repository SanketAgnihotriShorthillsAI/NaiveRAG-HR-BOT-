import os
import json
import argparse
import asyncio
import httpx
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
from dotenv import load_dotenv
import time 
load_dotenv()
print("📦 cwd =", os.getcwd())
print("📄 .env found?", os.path.exists(".env"))
print("🔐 Loaded deployment =", os.getenv("AZURE_OPENAI_DEPLOYMENT"))

# === Load Azure OpenAI Environment ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# === MongoDB Setup ===
mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]

# === Azure OpenAI call (async) ===
async def call_azure_llm(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts keywords from natural language queries for robust MongoDB document search."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            print("🌐 Sending request to Azure...")
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Azure API call failed: {e}")
        raise e

# === Gemini fallback (async) ===
async def call_gemini_llm(prompt: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not gemini_key:
        raise ValueError("❌ GEMINI_API_KEY not set in .env")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            print("🌐 Sending request to Gemini...")
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        raise e

# === Clean LLM Response ===
def clean_llm_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[3:-3].strip()
    return cleaned

# === Extract Keywords ===
async def extract_keywords(query: str, use_gemini=False) -> list[str]:
    print("\n🧠 Calling LLM to extract keywords...")

    prompt = f"""
You are an intelligent and helpful keyword extraction engine for a resume retrieval system powered by a NoSQL database (MongoDB). 

Given a natural language query, extract a flat list of **search-relevant keywords or phrases**. These keywords will be used to construct MongoDB $regex OR queries over structured JSON resumes to retrieve relevant candidates.

📌 DO extract:
- Skills, tools, technologies (e.g., "Python", "Postman", "GraphQL")
- Job titles or roles (e.g., "data engineer", "QA tester")
- Domains, frameworks, and platforms
- Experience levels or durations (e.g., "5 years experience", "mid-level")
- Certifications, degrees, or keywords from academic background

🚫 DO NOT include:
- Instructions or intents like "extract details", "provide an example", "show resumes with..."
- Question words or vague phrases (e.g., "how", "what", "example of")
- Generic terms like "project", "details", "describe", or filler text

🎯 Objective:
Identify ONLY the useful keywords that would help retrieve relevant resumes when matched against fields like skills, experience, summary, education, and certifications.

📝 Output Format:
Return a **valid JSON array** of strings only. Example:
["api testing", "postman", "rest", "soap", "graphql"]

Query: "{query}"
"""
    try:
        raw = await (call_gemini_llm(prompt) if use_gemini else call_azure_llm(prompt))
        print("📝 Raw LLM Response:", raw)
        cleaned = clean_llm_response(raw)
        keywords = json.loads(cleaned) if cleaned.startswith("[") else []
        if not isinstance(keywords, list):
            print("⚠️ Response is not a valid JSON list.")
            return []
        print("✅ Parsed Keywords:", keywords)
        return keywords
    except Exception as e:
        print(f"❌ Keyword extraction failed: {e}")
        return []

# === Build MongoDB Query ===
def build_mongo_query(keywords: list[str]) -> dict:
    print("\n🔧 Building MongoDB OR query...")
    or_conditions = []
    fields = [
        "summary", "skills", "projects.description", "projects.title",
        "experience.description", "experience.title", "experience.company",
        "education.institution", "certifications.title", "certifications.issuer",
    ]

    for kw in keywords:
        for field in fields:
            or_conditions.append({field: {"$regex": kw, "$options": "i"}})

    query = {"$or": or_conditions} if or_conditions else {}
    print("📄 Final Mongo Query:\n", json.dumps(query, indent=2))
    return query

# === Search Execution ===
async def search_resumes(nl_query: str, use_gemini=False):
    print(f"\n🔍 Original NL Query: \"{nl_query}\"")
    keywords = await extract_keywords(nl_query, use_gemini)
    if not keywords:
        print("⚠️ No keywords extracted. Exiting.")
        return {
            "query": nl_query,
            "keywords": [],
            "matched": [],
            "error": "Keyword extraction failed"
        }

    mongo_query = build_mongo_query(keywords)
    print("\n🔎 Searching in MongoDB...")
    results = list(collection.find(mongo_query))

    print(f"\n📊 Found {len(results)} matching resumes:\n")
    for res in results:
        print(f" - {res.get('name')} | {res.get('email')}")

    return {
        "query": nl_query,
        "keywords": keywords,
        "matched": [
            {
                "name": res.get("name", ""),
                "email": res.get("email", "")
            }
            for res in results
        ]
    }

# === CLI Runner ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language resume query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    args = parser.parse_args()

    result = asyncio.run(search_resumes(args.query, use_gemini=args.use_gemini))
