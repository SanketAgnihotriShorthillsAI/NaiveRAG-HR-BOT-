# Save this as evaluate.py
import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx
from typing import List, Dict

# Load env
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Input files
GOLDEN_ANSWER_FILE = Path("data/golden_answers_compiled.json")
NAIVE_LOGS_FILE = Path("compare/query_logs/naiveRag_query_logs.json")
OPTIMIZE_LOGS_FILE = Path("compare/query_logs/optimizeRag_query_logs.json")

# Output
OUTPUT_DIR = Path("data/eval_results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load logs
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Azure LLM call
async def call_azure_llm(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful evaluation assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# Prompt templates
def make_faithfulness_prompt(golden_entries, answer_text):
    golden_bullets = "\n".join([f"{i+1}. {entry['answer']}" for i, entry in enumerate(golden_entries)])
    return f"""
You're evaluating how faithful a generated answer is to the golden answers from resumes.

Golden Answers:
{golden_bullets}

Generated Answer:
\"\"\"{answer_text}\"\"\"

Instructions:
- Score from 0–10 for how well this answer reflects the **relevant people** mentioned in golden answers.
- Ignore how much content is matched — focus on the **correct people being included or excluded**.
- Do NOT penalize for missing job details or structure mismatches.
- Return response in **this JSON format only**:

{{
  "score": <0–10>,
  "missing_info": "...",
  "noise": "..."
}}
"""

def make_context_recall_prompt(golden_entries, context):
    golden_bullets = "\n".join([f"{i+1}. {entry['answer']}" for i, entry in enumerate(golden_entries)])
    return f"""
You're evaluating if the golden answers from resumes are findable in the retrieved context chunks.

Golden Answers:
{golden_bullets}

Retrieved Context:
\"\"\"{context}\"\"\"

Instructions:
- Score from 0–10 based on whether the **right names/resumes** mentioned in golden answers are supported by context.
- We do NOT care about job details — only presence/absence of names.
- Focus only on whether someone should have been found in context.
- Return response in **this JSON format only**:

{{
  "score": <0–10>,
  "missing_info": "...",
  "noise": "N/A"
}}
"""

def make_resume_mention_prompt(golden_entries, answer_text):
    expected_names = [entry["resume"] for entry in golden_entries]
    names_list = "\n".join(f"- {name}" for name in expected_names)
    return f"""
Check if the following resume owners were mentioned in the generated answer.

Expected Resumes:
{names_list}

Generated Answer:
\"\"\"{answer_text}\"\"\"

Instructions:
- Score from 0–10 based on how well these resume names are covered.
- Do not penalize if detailed content is missing — care only about **name mentions**.
- Return response in **this JSON format only**:

{{
  "score": <0–10>,
  "missing_info": "Names not mentioned...",
  "noise": "Extra names/hallucinations"
}}
"""

# Per-query all evals in parallel
async def run_all_evals(query, golden_entry, log_entry):
    answer = log_entry["response"]
    context = log_entry["context"]

    print(f"🔎 Running 3 evaluations for query: {query[:60]}...")

    async def faithfulness():
        prompt = make_faithfulness_prompt(golden_entry, answer)
        try:
            print("🔍 [FAITHFULNESS]")
            return {
                "query": query,
                "llm_response": await call_azure_llm(prompt),
            }
        except Exception as e:
            return {"query": query, "llm_response": f"Error: {str(e)}"}

    async def context_recall():
        prompt = make_context_recall_prompt(golden_entry, context)
        try:
            print("🔍 [CONTEXT]")
            return {
                "query": query,
                "llm_response": await call_azure_llm(prompt),
            }
        except Exception as e:
            return {"query": query, "llm_response": f"Error: {str(e)}"}

    async def resume_mention():
        prompt = make_resume_mention_prompt(golden_entry, answer)
        try:
            print("🔍 [RESUME NAMES]")
            return {
                "query": query,
                "llm_response": await call_azure_llm(prompt),
            }
        except Exception as e:
            return {"query": query, "llm_response": f"Error: {str(e)}"}

    return await asyncio.gather(faithfulness(), context_recall(), resume_mention())

# Run pipeline
async def evaluate_pipeline(name: str, logs: List[Dict], golden_answers: Dict):
    print(f"\n🚀 Starting evaluation for: {name.upper()} | Total queries: {len(logs)}\n")
    all_results = {"faithfulness": [], "context": [], "resume": []}

    for idx, log_entry in enumerate(logs, 1):
        query = log_entry["query"]
        if query not in golden_answers:
            print(f"⚠️  Skipping query not in golden set: '{query[:60]}...'")
            continue

        print(f"\n🔹 [{idx}/{len(logs)}] Query:\n{query[:80]}...\n")
        golden_entry = golden_answers[query]

        faith, ctx, resume = await run_all_evals(query, golden_entry, log_entry)
        all_results["faithfulness"].append(faith)
        all_results["context"].append(ctx)
        all_results["resume"].append(resume)

    output_file = OUTPUT_DIR / f"{name}_evaluations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved evaluation results for {name.upper()} to: {output_file}\n")

# MAIN
async def main():
    print("📥 Loading input files...")
    golden = load_json(GOLDEN_ANSWER_FILE)
    golden_map = {k: v for k, v in golden.items()}
    naive_logs = load_json(NAIVE_LOGS_FILE)
    optimize_logs = load_json(OPTIMIZE_LOGS_FILE)

    await evaluate_pipeline("naive", naive_logs, golden_map)
    await evaluate_pipeline("optimize", optimize_logs, golden_map)

if __name__ == "__main__":
    asyncio.run(main())
