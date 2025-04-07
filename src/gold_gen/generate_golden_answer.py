import os
import json
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# --- Configuration ---
RESUMES_DIR = Path("data/processed_resumes")
QUERIES_FILE = Path("data/test_queries.json")
OUTPUT_MATRIX_FILE = Path("data/gold_query_matrix.json")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


# --- Loaders ---
def load_test_queries():
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)  # List of query strings


def load_existing_matrix():
    if OUTPUT_MATRIX_FILE.exists():
        with open(OUTPUT_MATRIX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None  # Important for initializing correctly later


def save_matrix(matrix):
    OUTPUT_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MATRIX_FILE, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)



# --- Prompt builder ---
def build_prompt(resume_content, queries):
    prompt = (
        "You are an AI system helping an HR pipeline analyze resumes. "
        "Your task is to evaluate how well a given resume answers a set of HR queries.\n\n"

        "====================\n"
        "🔒 STRICT INSTRUCTIONS:\n"
        "- Match ONLY when the resume **clearly and fully** satisfies the query.\n"
        "- If a query has multiple parts (e.g., DevOps + Kubernetes), ALL parts must be present.\n"
        "- Do NOT make assumptions, guesses, or inferences beyond what's explicitly written.\n"
        "- Do NOT hallucinate structured answers like job transitions or timelines unless they are clearly stated.\n"
        "- Do NOT fabricate summaries or extract vague interpretations.\n"
        "- If nothing clearly matches, return null.\n"
        "- Never reuse answers across unrelated queries.\n\n"

        "====================\n"
        "🔎 CONTEXT:\n"
        "You will receive the full content of one resume followed by a list of queries.\n"
        "Respond to each query independently, based ONLY on the information inside the resume.\n\n"

        "Resume:\n"
        f"{resume_content}\n\n"

        "Queries:\n"
    )
    for q in queries:
        prompt += f"- {q}\n"

    prompt += (
        "\n====================\n"
        "📤 OUTPUT FORMAT:\n"
        "Return a JSON object with keys as the FULL query texts. Do not assume any structures unless clearly obvious.\n"
        "Each value must be:\n"
        "- A directly quoted or accurately paraphrased snippet from the resume (if relevant), OR\n"
        "- null (if not clearly present or incomplete).\n\n"

        "✅ Example:\n"
        "{\n"
        "  \"List candidates with DevOps and Kubernetes experience\": \"Worked as a DevOps Engineer using Docker and Kubernetes in CI/CD\",\n"
        "  \"Who has experience with Java and React.js?\": null\n"
        "}\n\n"

        "✅ Final Check: Ensure every answer is specific to the exact query. "
        "DO NOT fabricate structured objects. "
        "Only return what is clearly evident in the resume."
    )
    return prompt


# --- LLM API ---
def call_llm(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping HR evaluate resumes."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content.strip().strip("```json").strip("```"))
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None


# --- Resume evaluator ---
def is_resume_processed(matrix, resume_filename):
    return all(resume_filename in entry.get("results", {}) for entry in matrix)


def initialize_matrix(queries):
    return [
        {
            "query_id": f"query_{i}",
            "query_text": q,
            "results": {}
        }
        for i, q in enumerate(queries)
    ]


def process_resume(resume_file, queries, matrix):
    with open(resume_file, "r", encoding="utf-8") as f:
        resume_data = json.load(f)
    resume_filename = resume_data.get("file", resume_file.name)
    resume_content = resume_data.get("content", "")

    if not resume_content.strip():
        print(f"⚠️ Skipping empty resume: {resume_filename}")
        return

    prompt = build_prompt(resume_content, queries)
    print(f"🔍 Calling LLM for {resume_filename}...")

    llm_response = call_llm(prompt)
    if not llm_response:
        print(f"⚠️ Skipping {resume_filename} due to failed LLM call.")
        return

    query_text_to_id = {q: f"query_{i}" for i, q in enumerate(queries)}

    for query_entry in matrix:
        qid = query_entry["query_id"]
        qtext = query_entry["query_text"]
        answer = llm_response.get(qtext, None)
        query_entry["results"][resume_filename] = answer

    print(f"✅ Processed {resume_filename}")


# --- Main ---
def main():
    queries = load_test_queries()
    matrix = load_existing_matrix()
    if matrix is None:
        matrix = initialize_matrix(queries)

    resume_files = list(RESUMES_DIR.glob("*.json"))
    print(f"📄 {len(resume_files)} resumes found.\n🧪 Starting evaluation...\n")

    for resume_file in resume_files:
        with open(resume_file, "r", encoding="utf-8") as f:
            resume_data = json.load(f)
        resume_filename = resume_data.get("file", resume_file.name)

        if is_resume_processed(matrix, resume_filename):
            print(f"⏩ Skipping {resume_filename} (already processed).")
            continue

        process_resume(resume_file, queries, matrix)
        save_matrix(matrix)

    print("\n✅ All resumes processed. Output saved to:", OUTPUT_MATRIX_FILE)


if __name__ == "__main__":
    main()
