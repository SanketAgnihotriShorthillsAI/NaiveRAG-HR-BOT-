import os
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
import re

def clean_json_output(text):
    # Remove code block markers (```json ... ```)
    cleaned = re.sub(r"^```json", "", text.strip())
    cleaned = re.sub(r"^```", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()
# Load environment variables (make sure .env exists at project root)
load_dotenv()

# --- Configuration ---
RESUMES_DIR = Path("data/processed_resumes")
QUERIES_FILE = Path("data/test_queries.json")
OUTPUT_MATRIX_FILE = Path("data/gold_query_matrix.json")

# --- LLM API configuration via Azure OpenAI ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# --- Helper Functions ---

def load_test_queries():
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)  # Expecting a list of queries
    return queries

def load_existing_matrix():
    if OUTPUT_MATRIX_FILE.exists():
        with open(OUTPUT_MATRIX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Initialize matrix: each key is "query_0", "query_1", ... for each query in the list
        queries = load_test_queries()
        return {f"query_{i}": {} for i in range(len(queries))}

def save_matrix(matrix):
    OUTPUT_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MATRIX_FILE, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

def is_resume_processed(matrix, resume_filename):
    # Check if this resume already appears in any query entry
    for mapping in matrix.values():
        if resume_filename in mapping:
            return True
    return False

def build_prompt(resume_content, queries):
    prompt = (
        "You are an assistant helping an HR system identify relevant parts of a resume that answer specific HR queries.\n\n"
        "Your task is to read the full resume content and a list of queries. For each query, check whether this resume contains clear and direct information that would help answer the query.\n\n"
        "⚠️ Be extremely strict and precise when interpreting the query:\n"
        "- The resume must match **all parts** of the query conditions.\n"
        "- For example, if the query asks for 'Python developers who also know C++', only respond if the resume shows **both** Python and C++ clearly.\n"
        "- Do NOT respond if the resume only partially matches or includes just one skill.\n"
        "- Do NOT assume or infer skills or roles that are not explicitly stated.\n"
        "- Avoid vague or partial matches. Only include information that clearly supports the full query intent.\n\n"
        "If strictly relevant according to the query, extract and return the exact content (or paraphrased summary without missing any relevant content information) from the resume that answers the query prepend this with the name of the person whose resume it is.\n"
        "If the resume does not help answer a particular query, return null for that query.\n\n"
        "Resume:\n"
        "--------\n"
        f"{resume_content}\n\n"
        "Queries:\n"
    )
    for i, query in enumerate(queries):
        prompt += f"{i+1}. {query}\n"

    prompt += (
        "\nOutput Format:\n"
        "Return a JSON object with keys: \"query_0\", \"query_1\", ..., one for each query.\n"
        "Each value must either be:\n"
        "- A short string that quotes or summarizes the relevant part of the resume, OR\n"
        "- null (if no relevant info exists or the resume does not fully satisfy the query)\n\n"
        "Example:\n"
        "{\n"
        "  \"query_0\": \"Worked as a Python + C++ Developer at ABC Corp, 2021–2023\",\n"
        "  \"query_1\": null,\n"
        "  ...\n"
        "}"
    )

    return prompt


def call_llm(prompt):
    # Call Azure OpenAI API with the prompt and return the parsed JSON response.
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are an HR assistant evaluating resumes."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        cleaned = clean_json_output(content)

        # 🧾 Debug print
        print("\n🧾 Cleaned LLM Response (first 300 chars):")
        print(cleaned[:300])

        return json.loads(cleaned)
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None

def process_resume(resume_file, queries, matrix):
    # Load the processed resume JSON (expects keys "file" and "content")
    with open(resume_file, "r", encoding="utf-8") as f:
        resume_data = json.load(f)
    resume_filename = resume_data.get("file", resume_file.name)
    resume_content = resume_data.get("content", "")
    
    # Build the prompt for this resume
    prompt = build_prompt(resume_content, queries)
    print(f"🔍 Calling LLM for {resume_filename}...")
    
    llm_response = call_llm(prompt)
    if not llm_response:
        print(f"⚠️  No response for {resume_filename}. Skipping.")
        return
    
    # Expect llm_response to be a dict with keys "query_0", "query_1", ... up to len(queries)
    for i in range(len(queries)):
        key = f"query_{i}"
        evidence = llm_response.get(key)
        matrix.setdefault(key, {})[resume_filename] = evidence
    print(f"✅ Processed {resume_filename}.")

def main():
    queries = load_test_queries()
    matrix = load_existing_matrix()
    resume_files = list(RESUMES_DIR.glob("*"))
    print(f"🔍 Found {len(resume_files)} processed resume files to evaluate.\n")
    
    for resume_file in resume_files:
        if is_resume_processed(matrix, resume_file.name):
            print(f"⏩ Skipping {resume_file.name} (already processed).")
            continue
        process_resume(resume_file, queries, matrix)
        save_matrix(matrix)
    
    print("✅ All resumes processed. Output saved to:", OUTPUT_MATRIX_FILE)

if __name__ == "__main__":
    main()
