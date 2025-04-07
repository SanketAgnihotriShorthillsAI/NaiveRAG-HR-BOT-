import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx
import argparse

# Load environment variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Input files
GOLDEN_ANSWER_FILE = Path("data/golden_answers_compiled.json")
NAIVE_LOGS_FILE = Path("compare/query_logs/naiveRag_query_logs.json")
OPTIMIZE_LOGS_FILE = Path("compare/query_logs/optimizeRag_query_logs.json")

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load json from file
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Azure LLM call (async)
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

# Revised prompt for Context Recall
def make_context_recall_prompt(golden: str, context: str) -> str:
    return f"""
You are evaluating whether the retrieved context provides all the necessary supporting evidence for the expected candidate names as specified in the golden answer.

Golden Answer (expected supporting evidence for candidates):
\"\"\"{golden}\"\"\"

Retrieved Context:
\"\"\"{context}\"\"\"

Instructions:
- First, check if every expected candidate name is mentioned in the context.
- Then, for each name that is mentioned, verify that there is sufficient supporting detail (e.g., descriptions of roles, achievements, metrics) that clearly justifies the inclusion of that name.
- In your explanation (under the key "reason"), state clearly if:
   1. Some expected names are completely missing.
   2. Some names are mentioned, but the supporting details are inadequate.
   3. Both issues exist.
- Provide a fractional score between 0 and 1 (e.g., 0.8 means 80% of required evidence is present).
- Finally, also comment on whether the context is properly grounded (i.e., whether it clearly links the evidence to the expected names).

Return only a valid JSON object with exactly these keys:
{{
  "score": <fraction between 0 and 1>,
  "reason": "<a single, concise sentence explaining your evaluation>"
}}
"""

def make_context_precision_prompt(golden: str, context: str) -> str:
    return f"""
You are evaluating the relevance of the retrieved context with respect to the supporting evidence expected for the candidate names in the golden answer.

Golden Answer (expected supporting evidence for candidates):
\"\"\"{golden}\"\"\"

Retrieved Context:
\"\"\"{context}\"\"\"

Instructions:
- Evaluate how focused the context is on the necessary supporting evidence.
- Identify if the context includes extraneous or irrelevant information.
- In your explanation (under the key "reason"), state whether:
   1. The context is tightly focused on the required evidence.
   2. The context includes irrelevant details that reduce precision.
   3. Both issues (lack of focus and extraneous content) are present.
- Provide a fractional score between 0 and 1 (e.g., 0.7 means 70% of the context is relevant).
- Also, comment briefly if the context seems properly grounded.

Return only a valid JSON object with exactly these keys:
{{
  "score": <fraction between 0 and 1>,
  "reason": "<a single, concise sentence explaining your evaluation>"
}}
"""


# Evaluate a single query (sequential LLM calls)
async def evaluate_query(query: str, golden_entry: str, log_entry: dict) -> dict:
    # Extract the retrieved context from log entry
    context = log_entry["context"]

    # Evaluate context recall
    recall_prompt = make_context_recall_prompt(golden_entry, context)
    try:
        print(f"🔍 [Context Recall] Evaluating query: {query[:60]}...")
        recall_llm_raw = await call_azure_llm(recall_prompt)
        recall_parsed = json.loads(recall_llm_raw)
    except Exception as e:
        recall_parsed = {"score": 0, "reason": f"Error: {str(e)}"}
    
    # Evaluate context precision
    precision_prompt = make_context_precision_prompt(golden_entry, context)
    try:
        print(f"🔍 [Context Precision] Evaluating query: {query[:60]}...")
        precision_llm_raw = await call_azure_llm(precision_prompt)
        precision_parsed = json.loads(precision_llm_raw)
    except Exception as e:
        precision_parsed = {"score": 0, "reason": f"Error: {str(e)}"}
    
    return {
        "query": query,
        "context_recall": recall_parsed,
        "context_precision": precision_parsed,
    }

# Run evaluation pipeline sequentially for all queries
async def evaluate_pipeline(name: str, logs: list, golden_answers: dict):
    print(f"\n🚀 Starting evaluation for: {name.upper()} | Total queries: {len(logs)}\n")
    output_file = OUTPUT_DIR / f"{name}_eval_v2.json"

    results = []

    # Load existing results if available (optional)
    if output_file.exists():
        print(f"🔁 Loading previously saved results from {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    # Process each log entry sequentially
    for idx, log_entry in enumerate(logs, 1):
        query = log_entry["query"]
        if query not in golden_answers:
            print(f"⚠️  Skipping query not in golden set: '{query[:60]}...'")
            continue

        print(f"\n🔹 [{idx}/{len(logs)}] Query:\n{query[:80]}...\n")
        golden_entry = golden_answers[query]  # the golden evidence (broad context)

        evaluation = await evaluate_query(query, golden_entry, log_entry)
        results.append(evaluation)

        # Save incrementally
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved evaluation results for {name.upper()} to: {output_file}\n")

# MAIN
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=["naive", "optimize"], help="Pipeline to evaluate")
    args = parser.parse_args()

    print("📥 Loading input files...")
    golden = load_json(GOLDEN_ANSWER_FILE)
    # Assuming golden is a dict with query text as keys and broad evidence as value
    golden_map = {k: v for k, v in golden.items()}
    naive_logs = load_json(NAIVE_LOGS_FILE)
    optimize_logs = load_json(OPTIMIZE_LOGS_FILE)

    if args.pipeline == "naive":
        await evaluate_pipeline("naive", naive_logs, golden_map)
    elif args.pipeline == "optimize":
        await evaluate_pipeline("optimize", optimize_logs, golden_map)

if __name__ == "__main__":
    asyncio.run(main())
