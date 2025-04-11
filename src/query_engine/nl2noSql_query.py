import os
import json
import argparse
import asyncio
import httpx
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "false").lower() == "true"
LOG_FILE = "logs/nosql_pipeline.log" if PIPELINE_MODE else "logs/nosql_retriever.log"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

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
        print("\n🧠 Stage 1: Extracting primary keywords...")

        # ----------- PROMPT 1: Primary Extraction -----------
        prompt1 = f"""
You are an intelligent and helpful keyword extraction engine for a resume retrieval system powered by a NoSQL database (MongoDB).

Your task is to extract a flat list of highly relevant keywords or phrases from a natural language HR query. These keywords will be used to perform $regex OR matches against structured resume documents to retrieve suitable candidates.

-------------------------------
WHAT TO EXTRACT (Keyword Types):
-------------------------------
✓ Technical skills, tools, and technologies (e.g., "Python", "Postman", "GraphQL")
✓ Roles and job titles (e.g., "QA tester", "team lead", "recruitment specialist")
✓ HR/management actions and responsibilities (e.g., "led teams", "conducted training", "hiring pipeline", "onboarding process")
✓ Domains, frameworks, and platforms (e.g., "e-commerce", "SAP", "MERN stack")
✓ Educational qualifications and certifications (e.g., "MBA", "B.Tech", "PMP certified")
✓ Experience levels or durations (e.g., "5 years experience", "entry-level", "mid-senior")
✓ Names of candidates (e.g., "rahul sharma") if explicitly mentioned in the query

-------------------------------
VERB + PHRASE CLARIFICATION:
-------------------------------
✓ Keep compound action phrases that describe responsibilities or accomplishments
    → e.g., "led test automation", "conducted workshops", "managed payroll"
✗ Do NOT include action verbs as standalone keywords unless the full phrase is meaningful
    → ❌ "implemented" (bad), ✅ "implemented REST APIs" (good)

-------------------------------
DO NOT INCLUDE:
-------------------------------
✗ Generic verbs on their own ("implemented", "developed", "built", "created")
✗ Filler terms that add no value unless the query demands them
    → ("tools", "types", "project", "experience", "etc.")
✗ Vague instructional words from the question format ("how", "what", "describe", "give an example")

-------------------------------
OUTPUT FORMAT:
-------------------------------
Return only a valid JSON array of strings. No explanation. Example:
["api testing", "graphql", "payroll management", "rahul", "rahul sharma"]

-------------------------------
Query:
-------------------------------
{query}
"""


        try:
            raw1 = await (self.call_gemini_llm(prompt1) if use_gemini else self.call_azure_llm(prompt1))
            print("📝 Stage 1 Raw Response:", raw1)
            clean1 = self.clean_llm_response(raw1)
            print("✅ Stage 1 Cleaned:", clean1)
            primary_keywords = json.loads(clean1)
            if not isinstance(primary_keywords, list):
                print("❌ Stage 1 output is not a list.")
                return []
        except Exception as e:
            print(f"❌ Stage 1 keyword extraction failed: {e}")
            return []

        # ----------- PROMPT 2: Keyword Expansion and Decomposition -----------
        print("\n🔁 Stage 2: Refining keywords (expansion and splitting)...")

        prompt2 = f"""
    You are an assistant improving keyword coverage for document retrieval.

    Given a list of primary keywords extracted from an HR query, enhance them by:
    1. Adding related/enriched terms that would logically appear in resumes (e.g., "REST" → "rest api", "restful api").
    2. Breaking down multi-word phrases into additional meaningful terms (e.g., "rahul sharma" → "rahul", "sharma").
    3. Retaining all original keywords.

    Avoid adding:
    - Generic verbs unless part of a compound.
    - Overly vague terms like "experience", "tools", "project", unless critical to the query.

    Return only a flat valid JSON list of keywords. Example:
    ["api testing", "rest", "rest api", "rahul sharma", "rahul", "sharma"]

    Original Keywords:
    {json.dumps(primary_keywords, indent=2)}
    """

        try:
            raw2 = await (self.call_gemini_llm(prompt2) if use_gemini else self.call_azure_llm(prompt2))
            print("📝 Stage 2 Raw Response:", raw2)
            clean2 = self.clean_llm_response(raw2)
            print("✅ Stage 2 Cleaned:", clean2)
            final_keywords = json.loads(clean2)
            if not isinstance(final_keywords, list):
                print("❌ Stage 2 output is not a list.")
                return primary_keywords
            print("🎯 Final Keywords:", final_keywords)
            return final_keywords
        except Exception as e:
            print(f"❌ Stage 2 enrichment failed: {e}")
            return primary_keywords


    def build_mongo_query(self, keywords: list[str]) -> dict:
        print("\n🔧 Building MongoDB OR query...")
        or_conditions = []
        fields = [
            "name","summary", "skills", "projects.description", "projects.title",
            "experience.description", "experience.title", "experience.company",
            "education.institution", "certifications.title", "certifications.issuer",
        ]

        for kw in keywords:
            for field in fields:
                or_conditions.append({field: {"$regex": kw, "$options": "i"}})

        query = {"$or": or_conditions} if or_conditions else {}
        # print("📄 Final Mongo Query:\n", json.dumps(query, indent=2))
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
               {**res, "_id": str(res["_id"])} for res in results
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

    # import pprint
    # pprint.pprint(result)

    log_path = "data/nosql_retrieval_logs.jsonl"
    retriever.log_query_result(
        query=result["query"],
        keywords=result["keywords"],
        matched=result["matched"],
        log_path=log_path
    )

