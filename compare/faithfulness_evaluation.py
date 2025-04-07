import os
import json
import logging
from dotenv import load_dotenv
import asyncio
import httpx

load_dotenv()
logging.basicConfig(level=logging.INFO)

# === Gemini setup ===
import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# === Azure setup ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# === Log paths ===
INPUT_FILES = {
    "optimize": "query_logs/optimizeRag_query_logs.json",
    "naive": "query_logs/naiveRag_query_logs.json"
}
OUTPUT_FOLDER = "evaluated_logs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def make_evaluation_prompt(query, context, response):
    return f"""
### Task
Evaluate how faithfully the following **response** is grounded in the provided **context** for a user **query**. Give a faithfulness score (0 to 10) and briefly explain if any hallucinations are present.

### Query:
{query}

### Context:
{context}

### Response:
{response}

### Evaluation Format:
Faithfulness Score: X  (numeric between 0-10)
Hallucinations: (explain what was added or inferred without evidence)
"""


async def evaluate_with_azure(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a judge evaluating faithfulness of RAG system outputs."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, headers=headers, json=body)
        res.raise_for_status()
        result = res.json()
        return result["choices"][0]["message"]["content"]


async def evaluate_entry(entry, use_azure=False):
    prompt = make_evaluation_prompt(entry["query"], entry["context"], entry["response"])
    try:
        if use_azure:
            text = await evaluate_with_azure(prompt)
        else:
            response = gemini_model.generate_content(prompt)
            text = response.text.strip()

        score_line = next((line for line in text.splitlines() if "score" in line.lower()), "")
        score = int("".join(filter(str.isdigit, score_line))) if score_line else None

        return {
            **entry,
            "faithfulness_score": score,
            "llm_explanation": text
        }

    except Exception as e:
        logging.warning(f"Evaluation failed: {e}")
        return {
            **entry,
            "faithfulness_score": None,
            "llm_explanation": f"Evaluation failed: {str(e)}"
        }


async def run_evaluation(source: str, use_azure: bool = False):
    input_path = INPUT_FILES[source]
    output_path = os.path.join(OUTPUT_FOLDER, f"{source}_evaluated.json")

    with open(input_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    # Load existing evaluations if any
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                evaluated_entries = json.load(f)
            except json.JSONDecodeError:
                evaluated_entries = []
    else:
        evaluated_entries = []

    # Avoid re-evaluation of already processed queries
    evaluated_set = {
        (entry["query"], entry["response"]) for entry in evaluated_entries
    }

    new_results = []
    for entry in all_entries:
        identifier = (entry["query"], entry["response"])
        if identifier in evaluated_set:
            continue  # Already evaluated
        result = await evaluate_entry(entry, use_azure)
        new_results.append(result)

    full_results = evaluated_entries + new_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluated {len(new_results)} new entries. Total: {len(full_results)} saved to {output_path}")

async def head_to_head_evaluate(opt_entry, naive_entry, use_azure=False):
    query = opt_entry["query"]
    opt_context = opt_entry["context"]
    naive_context = naive_entry["context"]
    opt_response = opt_entry["response"]
    naive_response = naive_entry["response"]

    prompt = f"""
### Task
Compare the two RAG systems (OptimizeRAG and NaiveRAG) based on their responses and context for the user query. Determine which response is more faithful, helpful, and complete **given its own context**.

Be fair to both systems: judge each response **only against its own context**. If both are equally good or equally bad, mark it as "equal".

Then, analyze how the two responses differ in actual content. Mention differences in:
- Information included or omitted
- Specific claims made
- Level of detail
- Any assumptions or inferences
Do **not** mention formatting, language style, or tone differences.

---

### Query:
{query}

---

### OptimizeRAG
Context:
{opt_context}

Response:
{opt_response}

---

### NaiveRAG
Context:
{naive_context}

Response:
{naive_response}

---

### Evaluation Format:
Preferred System: optimize | naive | equal  
Reason: (brief explanation why one was preferred or if both were equal)  
Differences: (how the responses differ in content — include examples if helpful)
"""
    try:
        if use_azure:
            result_text = await evaluate_with_azure(prompt)
        else:
            result_text = gemini_model.generate_content(prompt).text.strip()

        # Naive extraction of preferred system
        if "optimize" in result_text.lower():
            preferred = "optimize"
        elif "naive" in result_text.lower():
            preferred = "naive"
        else:
            preferred = "equal"

        return {
            "query": query,
            "preferred_system": preferred,
            "llm_reasoning": result_text
        }

    except Exception as e:
        return {
            "query": query,
            "preferred_system": None,
            "llm_reasoning": f"Evaluation error: {e}"
        }

async def run_head_to_head(use_azure=False):
    path_opt = os.path.join(OUTPUT_FOLDER, "optimize_evaluated.json")
    path_naive = os.path.join(OUTPUT_FOLDER, "naive_evaluated.json")
    output_path = os.path.join(OUTPUT_FOLDER, "head_to_head_comparison.json")

    with open(path_opt, "r", encoding="utf-8") as f:
        opt_entries = json.load(f)
    with open(path_naive, "r", encoding="utf-8") as f:
        naive_entries = json.load(f)

    # Match queries
    naive_lookup = {e["query"]: e for e in naive_entries}
    comparisons = []

    print(f"🔄 Starting head-to-head evaluation... Total queries: {len(opt_entries)}")


    for idx, opt_entry in enumerate(opt_entries, start=1):
        query = opt_entry["query"]
        if query not in naive_lookup:
            print(f"⚠️  Skipping: No matching NaiveRAG response for query [{query}]")
            continue
        naive_entry = naive_lookup[query]
        result = await head_to_head_evaluate(opt_entry, naive_entry, use_azure)
        comparisons.append(result)

        print(f"[{idx}/{len(opt_entries)}] ✅ Query: {query[:60]}...")
        print(f"   ↪ Preferred: {result['preferred_system']}")
        print(f"   🧠 Reason: {result['llm_reasoning'].splitlines()[0]}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False)

    print(f"✅ Head-to-head comparison done. Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["optimize", "naive"], help="Evaluate one RAG system.")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure OpenAI instead of Gemini.")
    parser.add_argument("--compare", action="store_true", help="Run head-to-head comparison between systems.")
    args = parser.parse_args()

    if args.compare:
        asyncio.run(run_head_to_head(use_azure=args.use_azure))
    elif args.source:
        asyncio.run(run_evaluation(args.source, use_azure=args.use_azure))
    else:
        parser.print_help()
