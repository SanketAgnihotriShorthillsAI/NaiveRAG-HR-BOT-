import os
import json
import argparse
import asyncio
import httpx
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

class ResumeRetriever:
    def __init__(self):
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        self.collection = MongoClient(MONGO_URI)[DB_NAME][COLLECTION_NAME]

    async def call_azure_llm(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_key,
        }
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts keywords from natural language queries for robust MongoDB document search."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 6000,
        }
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version={self.azure_version}"

        async with httpx.AsyncClient(timeout=60) as client:
            print("🌐 Sending request to Azure...")
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def call_gemini_llm(self, prompt: str) -> str:
        if not self.gemini_key:
            raise ValueError("❌ GEMINI_API_KEY not set in .env")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60) as client:
            print("🌐 Sending request to Gemini...")
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    def clean_llm_response(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            return cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            return cleaned[3:-3].strip()
        return cleaned

    async def extract_keywords(self, query: str, use_gemini=False) -> list[str]:
        print("\n🧠 Calling LLM to extract keywords...")

        prompt = f"""
You are an intelligent and helpful keyword extraction engine for a resume retrieval system powered by a NoSQL database (MongoDB). 

Given a natural language query, extract a flat list of **search-relevant keywords or phrases**. These keywords will be used to construct MongoDB $regex OR queries over structured JSON resumes to retrieve relevant candidates.

📌 DO extract:
- Skills, tools, technologies (e.g., \"Python\", \"Postman\", \"GraphQL\")
- Job titles or roles (e.g., \"data engineer\", \"QA tester\")
- Domains, frameworks, and platforms
- Experience levels or durations (e.g., \"5 years experience\", \"mid-level\")
- Certifications, degrees, or keywords from academic background

🚫 DO NOT include:
- Instructions or intents like \"extract details\", \"provide an example\", \"show resumes with...\"
- Question words or vague phrases (e.g., \"how\", \"what\", \"example of\")
- Generic terms like \"project\", \"details\", \"describe\", or filler text

🎯 Objective:
Identify ONLY the useful keywords that would help retrieve relevant resumes when matched against fields like skills, experience, summary, education, and certifications.

📝 Output Format:
Return a **valid JSON array** of strings only. Example:
["api testing", "postman", "rest", "soap", "graphql"]

Query: "{query}"
"""
        try:
            raw = await (self.call_gemini_llm(prompt) if use_gemini else self.call_azure_llm(prompt))
            print("📝 Raw LLM Response:", raw)
            cleaned = self.clean_llm_response(raw)
            keywords = json.loads(cleaned) if cleaned.startswith("[") else []
            if not isinstance(keywords, list):
                print("⚠️ Response is not a valid JSON list.")
                return []
            print("✅ Parsed Keywords:", keywords)
            return keywords
        except Exception as e:
            print(f"❌ Keyword extraction failed: {e}")
            return []

    def build_mongo_query(self, keywords: list[str]) -> dict:
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

    async def search(self, query: str, use_gemini: bool = False) -> dict:
        print(f"\n🔍 Original NL Query: \"{query}\"")
        keywords = await self.extract_keywords(query, use_gemini)
        if not keywords:
            return {
                "query": query,
                "keywords": [],
                "matched": [],
                "error": "Keyword extraction failed"
            }

        mongo_query = self.build_mongo_query(keywords)
        print("\n🔎 Searching in MongoDB...")
        results = list(self.collection.find(mongo_query))

        print(f"\n📊 Found {len(results)} matching resumes:\n")
        for res in results:
            print(f" - {res.get('name')} | {res.get('email')}")

        return {
            "query": query,
            "keywords": keywords,
            "matched": [
                {k: v for k, v in res.items() if k != "_id"}
                for res in results
            ]
        }
    def log_query_result(self, query: str, keywords: list[str], matched: list[dict], log_path: str):
        log_data = {
            "query": query,
            "keywords": keywords,
            "matched_names": [res.get("name", "Not Provided") for res in matched],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language resume query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    args = parser.parse_args()

    retriever = ResumeRetriever()
    result = asyncio.run(retriever.search(args.query, use_gemini=args.use_gemini))

    import pprint
    pprint.pprint(result)

    log_path = "data/nosql_retrieval_logs.json"
    retriever.log_query_result(
        query=result["query"],
        keywords=result["keywords"],
        matched=result["matched"],
        log_path=log_path
    )

