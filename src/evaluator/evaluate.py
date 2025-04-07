import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx
from typing import List, Dict
import argparse
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
def make_faithfulness_prompt(golden: str, generated: str) -> str:
    return f"""
You are an AI evaluator checking the faithfulness of a generated RAG answer against a golden reference answer.

The goal is to verify whether the generated answer correctly includes the people listed in the golden answer. You do **not** need to verify the details *about* the people — only whether the **correct names** are included or omitted.

Golden Answer (reference list of people):
\"\"\"{golden}\"\"\"

Generated Answer:
\"\"\"{generated}\"\"\"

Instructions:
- Judge whether the expected people are mentioned.
- Do NOT penalize the answer for summarizing or changing the format of the information.
- If some expected names are missing, note them.
- If extra names appear (not in the golden answer), list them as hallucinations.
- Give a score between 0 and 10 based on how many correct names were retrieved.
- Special Case: If the list of expected resumes is empty, it means no resumes should be mentioned in the answer.
If any names appear in the answer, they must be treated as hallucinations and the score should be 0.

💡 **Scoring must be aligned with match coverage**:
- 0 = 0% of expected names mentioned
- 5 = ~50% of expected names mentioned
- 10 = All or nearly all expected names correctly mentioned

Return only a valid JSON with this format:

{{
  "score": <integer between 0-10>,
  "missing_info": "<comma-separated missing names or 'N/A'>",
  "noise": "<hallucinated names or 'N/A'>",
  "percent_matched": "<X% of expected names mentioned>"
}}
"""


def make_context_recall_prompt(golden: str, context: str) -> str:
    return f"""
You are evaluating if the golden answer is properly supported by the retrieved context chunks.

The focus is on whether the context contains enough evidence to support the people listed in the golden answer — even if the exact sentence or structure differs.

Golden Answer (expected people):
\"\"\"{golden}\"\"\"

Retrieved Context:
\"\"\"{context}\"\"\"

Instructions:
- Your job is NOT to judge the final answer — only whether this context could have helped an LLM get the correct people.
- Identify any expected people missing from the context.
- Ignore formatting or paraphrasing.
- Give a score between 0 and 10 based on how well the context supports the golden answer.

💡 **Scoring must be aligned with match coverage**:
- 0 = 0% of expected names found in context
- 5 = ~50% of expected names found
- 10 = All or nearly all expected names clearly supported

Return only a valid JSON with this format:

{{
  "score": <integer between 0-10>,
  "missing_info": "<comma-separated missing names or 'N/A'>",
  "percent_matched": "<X% of golden names found in context>"
}}
"""



def make_resume_mention_prompt(golden_entries, answer_text):
    expected_names = [entry["resume"] for entry in golden_entries]
    names_list = "\n".join(f"- {name}" for name in expected_names)
    return f"""
You are verifying whether the correct resume names were mentioned in a generated answer.

Expected Resume Names (from golden answer):
\"\"\"{names_list}\"\"\"

Generated Answer:
\"\"\"{answer_text}\"\"\"

Instructions:
- Check if the expected names or people are mentioned in the answer.
- Do NOT be strict on name formatting (e.g., "Neha Resume.pdf" → "Neha" is fine).
- If some are missing, list them.
- If the answer mentions people who were not in the golden list, flag them as hallucinated.
- Do NOT be strict on name formatting (e.g., "Neha Resume.pdf" → "Neha" is fine).
- Score from 0 to 10 based on how many expected names were retrieved.
- Special Case: If the list of expected resumes is empty, it means no resumes should be mentioned in the answer.
If any names appear in the answer, they must be treated as hallucinations and the score should be 0.

💡 **Scoring must be aligned with match coverage**:
- 0 = None of the expected resumes are mentioned
- 5 = ~50% of expected resumes are correctly included
- 10 = All or nearly all resume names are correctly mentioned

Return only a valid JSON with this format:

{{
  "score": <integer between 0-10>,
  "missing_info": "<comma-separated missing names or 'N/A'>",
  "noise": "<hallucinated names or 'N/A'>",
  "percent_matched": "<X% of expected resumes mentioned>"
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
            llm_raw = await call_azure_llm(prompt)
            try:
                parsed = json.loads(llm_raw)
            except Exception as e:
                print(f"⚠️ Error parsing faithfulness response for '{query[:40]}': {e}")
                parsed = {}

            return {
                "query": query,
                "llm_response": llm_raw,
                "score": parsed.get("score"),
                "missing_info": parsed.get("missing_info"),
                "noise": parsed.get("noise"),
                "percent_matched": parsed.get("percent_matched"),
            }
        except Exception as e:
            return {"query": query, "llm_response": f"Error: {str(e)}"}

    async def context_recall():
        prompt = make_context_recall_prompt(golden_entry, context)
        try:
            print("🔍 [CONTEXT]")
            llm_raw = await call_azure_llm(prompt)
            try:
                parsed = json.loads(llm_raw)
            except Exception as e:
                print(f"⚠️ Error parsing context recall response for '{query[:40]}': {e}")
                parsed = {}

            return {
                "query": query,
                "llm_response": llm_raw,
                "score": parsed.get("score"),
                "missing_info": parsed.get("missing_info"),
                "percent_matched": parsed.get("percent_matched"),
            }
        except Exception as e:
            return {"query": query, "llm_response": f"Error: {str(e)}"}

    async def resume_mention():
        prompt = make_resume_mention_prompt(golden_entry, answer)
        try:
            print("🔍 [RESUME NAMES]")
            llm_raw = await call_azure_llm(prompt)
            try:
                parsed = json.loads(llm_raw)
            except Exception as e:
                print(f"⚠️ Error parsing resume mention response for '{query[:40]}': {e}")
                parsed = {}

            return {
                "query": query,
                "llm_response": llm_raw,
                "score": parsed.get("score"),
                "missing_info": parsed.get("missing_info"),
                "noise": parsed.get("noise"),
                "percent_matched": parsed.get("percent_matched"),
            }
        except Exception as e:
            return {"query": query, "llm_response": f"Error: {str(e)}"}

    return await asyncio.gather(faithfulness(), context_recall(), resume_mention())

# Run pipeline
async def evaluate_pipeline(name: str, logs: List[Dict], golden_answers: Dict):
    print(f"\n🚀 Starting evaluation for: {name.upper()} | Total queries: {len(logs)}\n")

    output_file = OUTPUT_DIR / f"{name}_evaluations.json"

    # Initialize maps to avoid duplicates and allow overriding
    faithfulness_map = {}
    context_map = {}
    resume_map = {}

    already_evaluated = set()

    # Load existing results if available
    if output_file.exists():
        print(f"🔁 Loading previously saved results from {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            previous = json.load(f)

        faithfulness_map = {entry["query"]: entry for entry in previous.get("faithfulness", [])}
        context_map = {entry["query"]: entry for entry in previous.get("context", [])}
        resume_map = {entry["query"]: entry for entry in previous.get("resume", [])}

        # Mark queries as fully evaluated only if all 3 are valid
        for q in faithfulness_map:
            f = faithfulness_map.get(q)
            c = context_map.get(q)
            r = resume_map.get(q)
            if (
                f and isinstance(f.get("llm_response"), str) and not f["llm_response"].startswith("Error:")
                and c and isinstance(c.get("llm_response"), str) and not c["llm_response"].startswith("Error:")
                and r and isinstance(r.get("llm_response"), str) and not r["llm_response"].startswith("Error:")
            ):
                already_evaluated.add(q)

    for idx, log_entry in enumerate(logs, 1):
        query = log_entry["query"]
        if query not in golden_answers:
            print(f"⚠️  Skipping query not in golden set: '{query[:60]}...'")
            continue

        if query in already_evaluated:
            print(f"⏩ Skipping previously evaluated query: '{query[:60]}...'")
            continue

        print(f"\n🔹 [{idx}/{len(logs)}] Query:\n{query[:80]}...\n")
        golden_entry = golden_answers[query]

        faith, ctx, resume = await run_all_evals(query, golden_entry, log_entry)

        # Overwrite or add new entries
        faithfulness_map[query] = faith
        context_map[query] = ctx
        resume_map[query] = resume

        # Save incrementally
        all_results = {
            "faithfulness": list(faithfulness_map.values()),
            "context": list(context_map.values()),
            "resume": list(resume_map.values()),
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved evaluation results for {name.upper()} to: {output_file}\n")

# MAIN
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=["naive", "optimize"], help="Pipeline to evaluate")
    args = parser.parse_args()

    print("📥 Loading input files...")
    golden = load_json(GOLDEN_ANSWER_FILE)
    golden_map = {k: v for k, v in golden.items()}
    naive_logs = load_json(NAIVE_LOGS_FILE)
    optimize_logs = load_json(OPTIMIZE_LOGS_FILE)

    if args.pipeline == "naive":
        await evaluate_pipeline("naive", naive_logs, golden_map)
    elif args.pipeline == "optimize":
        await evaluate_pipeline("optimize", optimize_logs, golden_map)

if __name__ == "__main__":
    asyncio.run(main())
