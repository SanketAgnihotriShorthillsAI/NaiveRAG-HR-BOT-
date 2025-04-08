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

# Revised prompt for Context Recall with additional instructions on grounding
def make_context_recall_prompt(golden: str, context: str) -> str:
    return f"""
You are evaluating whether the retrieved context provides all the necessary supporting evidence for the expected candidate names as specified in the golden answer.

Golden Answer (expected supporting evidence for candidates):
\"\"\"{golden}\"\"\"

Retrieved Context:
\"\"\"{context}\"\"\"

Instructions:
-Do NOT be strict on name formatting:
For example, if the candidate’s name appears as "Neha Resume.pdf" or "Neha-Resume.pdf", you should interpret it simply as "Neha" without penalizing the response for including file extensions or minor formatting differences.
- First, check if every expected candidate name is mentioned in the context.
- Then, for each mentioned name, verify that there is sufficient supporting detail (e.g., descriptions of roles, achievements, metrics) that clearly justifies its inclusion.
- Explicitly state if the evidence is missing entirely, present but insufficient, or if both issues exist.
- Also, comment on whether the context is properly grounded (i.e., the supporting evidence is clearly connected to the candidate names).
- Provide a fractional score between 0 and 1 (e.g., 0.8 means 80% of required evidence is present).
- Provide a single concise sentence as the explanation for your score.

Return only a valid JSON object with exactly these keys:
{{
  "score": <fraction between 0 and 1>,
  "reason": "<a single, concise sentence explaining your evaluation>"
}}
"""

# Revised prompt for Context Precision (Updated to evaluate the same evidence as Recall, but with precision criteria)
def make_context_precision_prompt(golden: str, context: str) -> str:
    return f"""
You are evaluating the relevance and focus of the retrieved context with respect to the supporting evidence expected for the candidate names as specified in the golden answer.

Golden Answer (expected supporting evidence for candidates):
\"\"\"{golden}\"\"\"

Retrieved Context:
\"\"\"{context}\"\"\"

Instructions:
- Check if every expected candidate name is mentioned and, for each name, verify that there is sufficient supporting detail (e.g., role descriptions, achievements, metrics) that justifies its inclusion.
-Do NOT be strict on name formatting: 
    1.For example, if the candidate’s name appears as "Neha Resume.pdf" or "Neha-Resume.pdf", you should interpret it simply as "Neha" without penalizing the response for including file extensions or minor formatting differences.
    2.In some cases, the candidate's name may appear in a cluttered or non-standard format and might be positioned above the relevant details. Keep this in mind when interpreting responses, especially for name-based queries.
- In addition, assess whether the context includes extraneous or irrelevant details that do not contribute to this evidence.
- Explicitly state if:
   1. All expected candidate names and evidence are present without extraneous content.
   2. Some expected names/evidence are present but are mixed with a lot of irrelevant details.
   3. Some expected names/evidence are missing or the supporting details are insufficient.
- Provide a fractional score between 0 and 1 where a higher score means the context is highly focused and contains little irrelevant information, while a lower score indicates that much of the context is extraneous relative to the expected evidence.
- Provide a single concise sentence as your explanation.

Return only a valid JSON object with exactly these keys:
{{
  "score": <fraction between 0 and 1>,
  "reason": "<a single, concise sentence explaining your evaluation>"
}}
"""


# New prompt for Faithfulness Evaluation
def make_faithfulness_prompt(golden: str, generated: str) -> str:
    return f"""
You are evaluating the faithfulness of a generated answer against a golden reference answer that lists expected candidate names and supporting evidence.

Golden Answer (expected candidate names and supporting evidence):
\"\"\"{golden}\"\"\"

Generated Answer:
\"\"\"{generated}\"\"\"

Instructions:
- Check if each expected candidate name is present in the generated answer.
- Determine if the generated answer includes sufficient supporting details (e.g., role descriptions, achievements, metrics) for each name.
- Clearly state in your explanation if:
    1. Some expected names are completely missing.
    2. Some names are mentioned but without adequate supporting details.
    3. Both issues exist.
- Do not penalize for minor formatting differences or extra non-critical text.
- Provide a fractional score between 0 and 1 (e.g., 0.8 means 80% fidelity).
- Provide a single concise sentence explaining your evaluation, noting any missing or insufficient evidence and whether the answer is properly grounded.

Return only a valid JSON object with exactly these keys:
{{
  "score": <fraction between 0 and 1>,
  "reason": "<a single, concise sentence explaining your evaluation>"
}}
"""

def extract_json_from_text(text: str) -> str:
    """
    Extracts the first JSON object from the provided text by finding the first '{'
    and then finding the corresponding closing '}'.
    Returns the extracted JSON string or an empty string if not found.
    """
    start = text.find("{")
    if start == -1:
        return ""
    count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            count += 1
        elif text[i] == "}":
            count -= 1
            if count == 0:
                return text[start:i+1]
    return ""

# Evaluate a single query (sequential LLM calls) based on evaluation mode
async def evaluate_query(query: str, golden_entry: str, log_entry: dict, mode: str) -> dict:
    context = log_entry["context"]
    result = {"query": query}

    if mode == "faithfulness":
        # Faithfulness evaluation uses the generated answer from the log_entry
        generated = log_entry["response"]
        faith_prompt = make_faithfulness_prompt(golden_entry, generated)
        try:
            print(f"🔍 [Faithfulness] Evaluating query: {query[:60]}...")
            faith_llm_raw = await call_azure_llm(faith_prompt)
            json_str_faith = extract_json_from_text(faith_llm_raw)
            if not json_str_faith:
                faith_parsed = {"score": 0, "reason": "Empty response"}
            else:
                faith_parsed = json.loads(json_str_faith)
            result["faithfulness"] = {
                "score": faith_parsed.get("score", 0),
                "reason": faith_parsed.get("reason", ""),
                "raw_response": faith_llm_raw
            }
        except Exception as e:
            result["faithfulness"] = {"score": 0, "reason": f"Error: {str(e)}", "raw_response": ""}
            print(f"⚠️ [Faithfulness] Error for query '{query[:40]}': {str(e)}")
    else:
        # Retrieval evaluation (Context Recall & Context Precision)
        # Evaluate context recall
        recall_prompt = make_context_recall_prompt(golden_entry, context)
        try:
            print(f"🔍 [Context Recall] Evaluating query: {query[:60]}...")
            recall_llm_raw = await call_azure_llm(recall_prompt)
            json_str_recall = extract_json_from_text(recall_llm_raw)
            if not json_str_recall:
                recall_parsed = {"score": 0, "reason": "Empty response"}
            else:
                recall_parsed = json.loads(json_str_recall)
        except Exception as e:
            recall_llm_raw = ""
            recall_parsed = {"score": 0, "reason": f"Error: {str(e)}"}
            print(f"⚠️ [Context Recall] Error for query '{query[:40]}': {str(e)}")
    
        # Evaluate context precision
        precision_prompt = make_context_precision_prompt(golden_entry, context)
        try:
            print(f"🔍 [Context Precision] Evaluating query: {query[:60]}...")
            precision_llm_raw = await call_azure_llm(precision_prompt)
            json_str_precision = extract_json_from_text(precision_llm_raw)
            if not json_str_precision:
                precision_parsed = {"score": 0, "reason": "Empty response"}
            else:
                precision_parsed = json.loads(json_str_precision)
        except Exception as e:
            precision_llm_raw = ""
            precision_parsed = {"score": 0, "reason": f"Error: {str(e)}"}
            print(f"⚠️ [Context Precision] Error for query '{query[:40]}': {str(e)}")
    
        result["context_recall"] = {
            "score": recall_parsed.get("score", 0),
            "reason": recall_parsed.get("reason", ""),
            "raw_response": recall_llm_raw
        }
        result["context_precision"] = {
            "score": precision_parsed.get("score", 0),
            "reason": precision_parsed.get("reason", ""),
            "raw_response": precision_llm_raw
        }
    return result

# Run evaluation pipeline sequentially for all queries with a unified result per query
async def evaluate_pipeline(name: str, logs: list, golden_answers: dict, mode: str):
    print(f"\n🚀 Starting evaluation for: {name.upper()} | Total queries: {len(logs)} | Mode: {mode}\n")
    output_file = OUTPUT_DIR / f"{name}_eval_v2.json"

    results_dict = {}
    # Load existing results if available (optional)
    if output_file.exists():
        print(f"🔁 Loading previously saved results from {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        for entry in existing_results:
            results_dict[entry["query"]] = entry

    # Process each log entry sequentially
    for idx, log_entry in enumerate(logs, 1):
        query = log_entry["query"]
        if query not in golden_answers:
            print(f"⚠️  Skipping query not in golden set: '{query[:60]}...'")
            continue

        # ⏭️ Skip logic: check if already evaluated and not a 429 error
        if query in results_dict:
            existing_entry = results_dict[query]
            reasons = []

            if mode == "faithfulness":
                reasons.append(existing_entry.get("faithfulness", {}).get("reason", ""))
            elif mode == "retrieval":
                reasons.append(existing_entry.get("context_recall", {}).get("reason", ""))
                reasons.append(existing_entry.get("context_precision", {}).get("reason", ""))

            if all("429 Too Many Requests" not in r for r in reasons if r):
                print(f"⏭️  Skipping already evaluated query [{idx}]: '{query[:60]}...'")
                continue
            else:
                print(f"🔁 Re-evaluating due to 429 error: '{query[:60]}...'")


        print(f"\n🔹 [{idx}/{len(logs)}] Query:\n{query[:80]}...\n")
        golden_entry = golden_answers[query]  # the golden evidence

        evaluation = await evaluate_query(query, golden_entry, log_entry, mode)
        if query in results_dict:
            results_dict[query].update(evaluation)
        else:
            results_dict[query] = evaluation

        # Save incrementally (convert dict values to list)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(list(results_dict.values()), f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved evaluation results for {name.upper()} to: {output_file}\n")

# MAIN
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=["naive", "optimize"], help="Pipeline to evaluate")
    parser.add_argument("--mode", required=True, choices=["retrieval", "faithfulness"], help="Evaluation mode: retrieval (context recall and precision) or faithfulness")
    args = parser.parse_args()

    print("📥 Loading input files...")
    golden = load_json(GOLDEN_ANSWER_FILE)
    # Assuming golden is a dict with query text as keys and broad evidence as value
    golden_map = {k: v for k, v in golden.items()}
    naive_logs = load_json(NAIVE_LOGS_FILE)
    optimize_logs = load_json(OPTIMIZE_LOGS_FILE)

    if args.pipeline == "naive":
        await evaluate_pipeline("naive", naive_logs, golden_map, args.mode)
    elif args.pipeline == "optimize":
        await evaluate_pipeline("optimize", optimize_logs, golden_map, args.mode)

if __name__ == "__main__":
    asyncio.run(main())
