import os
import json
import argparse
import httpx
import asyncio
from dotenv import load_dotenv
from .nl2noSql_query import ResumeRetriever

load_dotenv()

LOG_FILE = "logs/answer_generator.log"
os.makedirs("logs", exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

class AnswerGenerator:
    def __init__(self, use_gemini: bool = False):
        self.use_gemini = use_gemini

        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    async def _call_azure(self, messages):
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version={self.azure_version}"
        headers = {"Content-Type": "application/json", "api-key": self.azure_key}
        body = {"messages": messages, "temperature": 0.3, "max_tokens": 1000}

        async with httpx.AsyncClient(timeout=60) as client:
            log("🌐 Sending to Azure...")
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt):
        if not self.gemini_key:
            raise ValueError("❌ GEMINI_API_KEY not set in .env")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60) as client:
            log("🌐 Sending to Gemini...")
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

    async def rerank_resumes(self, query: str, resumes: list[dict]) -> list[dict]:
        input_text = "\n\n".join(
            [f"Resume {i+1}:\n" + json.dumps(res, indent=2) for i, res in enumerate(resumes)]
        )
        prompt = f"""
You are an intelligent assistant designed to help HR professionals make sense of multiple candidate resumes based on a specific hiring query.

You are provided with:
1. A **natural language query** that an HR professional might ask (e.g., "List candidates who have worked with Python and C++", or "Find people with managerial experience in the education sector").
2. A list of **resumes (in JSON format)** that were already filtered and retrieved from MongoDB based on key terms extracted from the query. These resumes are the most likely matches to the HR query.

<query>
{query}
</query>

<resumes>
{input_text}
</resumes>

Your task is to:
- Rigorously evaluate each resume to determine if it directly or strongly supports answering the query.
- Perform intelligent filtering and re-ranking of resumes based on their relevance and closeness to the query intent.
- Avoid superficial matches: for example, if the query is "Python developers", a resume that just mentions Python once without any related role or experience might be less relevant than one showing active development using Python.
- Prioritize **precision over blind matching** but be cautious in rejecting. You should only classify a resume as "irrelevant" when you're confident that it does **not** meaningfully contribute toward answering the query.
- Avoid hallucinations: base your decisions **strictly on the resume content**.

Guidelines:
- If the query is simple and broad (e.g., "List all Python developers"), then **most resumes might be relevant** and re-ranking isn't necessary. In such cases, list them all under "relevant_names" with minimal or no "irrelevant_names".
- If the query includes multiple filters (e.g., "Python devs with knowledge of C++ and ML frameworks"), you must find resumes that satisfy **all conditions**, and consider others as less relevant or irrelevant.
- For ambiguous or subjective queries (e.g., "candidates showing a clear career shift"), use your best analytical reasoning from the resume timeline, titles, and industries.
- If you're unsure about a resume, **default to keeping it relevant** rather than wrongly discarding it.

Return your answer in the following JSON format ONLY:
{{
  "relevant_names": ["Full Name 1", "Full Name 2"],
  "irrelevant_names": ["Full Name 3", "Full Name 4"]
}}

Remember:
- Do not explain.
- Do not return anything outside of the JSON block.
- The two lists must be mutually exclusive and only include candidate "name" fields.

"""

        try:
            response_raw = await (
                self._call_gemini(prompt) if self.use_gemini else self._call_azure([{"role": "user", "content": prompt}])
            )
            log("🧾 Raw reranker response: " + response_raw)
            response_clean = self.clean_llm_response(response_raw)
            log("🧹 Cleaned reranker response: " + response_clean)

            parsed = json.loads(response_clean)
            relevant_names = parsed.get("relevant_names", [])
            irrelevant_names = parsed.get("irrelevant_names", [])

            relevant = [r for r in resumes if r["name"] in relevant_names]

            log("🔍 Resumes sent to reranker: " + str([r["name"] for r in resumes]))
            log("✅ Relevant resumes after rerank: " + str(relevant_names))
            log("❌ Irrelevant resumes after rerank: " + str(irrelevant_names))

            return relevant
        except Exception as e:
            log(f"⚠️ Reranking failed: {e}")
            return resumes

    async def generate_answer(self, query: str, reranked_resumes: list[dict]) -> str:
        content_summary = "\n\n".join([
            f"Resume {i+1}:\n" + json.dumps(res, indent=2)
            for i, res in enumerate(reranked_resumes)
        ])

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes relevant candidate information based on a natural language query."},
            {"role": "user", "content": f"""
Query: {query}

Here are the top-ranked resumes relevant to this query:
{content_summary}

Based on this, generate a clear, direct summary answer to the query.
"""}
        ]

        try:
            response = await (
                self._call_gemini(messages[-1]["content"]) if self.use_gemini else self._call_azure(messages)
            )
            log("📝 Final generated answer:\n" + response)
            return response
        except Exception as e:
            log(f"⚠️ Generation failed: {e}")
            return f"⚠️ Generation failed: {e}"


# === CLI TESTING ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    args = parser.parse_args()

    async def run():
        retriever = ResumeRetriever()
        retrieval_result = await retriever.search(args.query, use_gemini=args.use_gemini)
        resumes = retrieval_result.get("matched", [])

        if not resumes:
            print("❌ No resumes retrieved. Aborting generation.")
            log("❌ No resumes retrieved. Aborting generation.")
            return

        generator = AnswerGenerator(use_gemini=args.use_gemini)
        reranked = await generator.rerank_resumes(args.query, resumes)
        summary = await generator.generate_answer(args.query, reranked)

        print("\n📝 FINAL GENERATED ANSWER:\n")
        print(summary)

    asyncio.run(run())