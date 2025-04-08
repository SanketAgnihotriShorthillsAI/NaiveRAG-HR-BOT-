import os
import json
from pathlib import Path
from dotenv import load_dotenv
import httpx
import asyncio

# Load environment variables
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise ValueError("❌ Missing Azure OpenAI environment variables in .env")

# Paths
INPUT_DIR = Path("data/llama_parse_resumes")
OUTPUT_DIR = Path("data/standardized_resumes")
RAW_LOG_DIR = Path("data/standardized_raw_responses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Prompt Template
def make_standardizer_prompt(content: str, links: list) -> str:
    return f"""
You are an intelligent and robust context-aware resume standardizer. Your task is to convert resume content into a clean, structured, normalized/standardized JSON format suitable for both semantic retrieval and relational database storage.

--- PARSED RESUME CONTENT ---
The content has been parsed using *LlamaParse*, so please note:
- It is formatted in markdown-style text.
- Section names like Education, Experience, Projects, Skills, etc., are likely to be marked by headings or bullet points.
- The resume templates can vary significantly. For example, "Professional Experience", "Work History", or "Employment" may all refer to the same section. We need you for Normalizing such variants into the defined structure mentioned below.
- Some artifacts or repetition may occur due to Llama Parse not being able to understand the text is embedded link.

--- EXTRACTED HYPERLINKS ---
The hyperlinks have been extracted using *Fitz*. Please note:
- The links are mostly accurate, but the anchor texts may be confusing and should not be blindly trusted.
- Only use a link where the surrounding content makes the intent clear (e.g., GitHub → social, certificate → certifications).

--- RESUME CONTENT ---
\"\"\"{content}\"\"\"

--- HYPERLINKS (Extracted from PDF) ---
{json.dumps(links, indent=2)}

--- STANDARDIZED STRUCTURE ---
Convert the resume to a JSON object strictly following this structure:

{{
  "name": str,
  "email": str,
  "phone": str,
  "location": str,
  "summary": str,
  "education": [
    {{
      "degree": str,
      "institution": str,
      "year": int or str
    }}
  ],
  "experience": [
    {{
      "title": str,
      "company": str,
      "duration": str,
      "location": str (optional),
      "description": str
    }}
  ],
  "skills": [str],
  "projects": [
    {{
      "title": str,
      "description": str,
      "link": str (optional)
    }}
  ],
  "certifications": [
    {{
      "title": str,
      "issuer": str (optional),
      "year": str or int (optional),
      "link": str (only if it can be reliably mapped from extracted hyperlinks)
    }}
  ],
  "languages": [str],
  "social_profiles": [
    {{
      "platform": str,
      "link": str
    }}
  ]
}}

--- GUIDELINES ---
- Strictly follow the above structure. Do not introduce or remove fields arbitrarily.
- Do not change the structure based on data availability. Keep all keys present, with empty arrays if needed.
- Avoid assumptions. Only use data present in the parsed content or reliably inferred via links.
- Ensure all meaningful content is preserved — even if duplicated or malformed in parsing.
- Output only a valid JSON object. Do not wrap in markdown or include any extra commentary.

The quality of your standardization directly impacts downstream processing. Ensure robustness and consistency in formatting across all resumes.
"""


# Extract raw JSON from response even if wrapped in ```json
def clean_llm_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[3:-3].strip()
    return cleaned

# Azure OpenAI call
async def call_azure_llm(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that formats resumes into structured JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# Parse + standardize one file
async def standardize_resume(file_path: Path):
    output_path = OUTPUT_DIR / file_path.name
    raw_log_path = RAW_LOG_DIR / file_path.name.replace(".json", ".md")

    if output_path.exists():
        print(f"⏩ Skipping {file_path.name} (already standardized)")
        return

    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)

    content = raw.get("content", "")
    links = raw.get("links", [])

    if not content.strip():
        print(f"⚠️ Empty content in {file_path.name}, skipping.")
        return

    prompt = make_standardizer_prompt(content, links)

    try:
        print(f"🔍 Standardizing: {file_path.name}")
        raw_response = await call_azure_llm(prompt)

        # Save raw LLM response
        with open(raw_log_path, "w", encoding="utf-8") as f:
            f.write(raw_response)

        cleaned_json = clean_llm_response(raw_response)
        parsed_json = json.loads(cleaned_json)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved standardized resume: {output_path.name}")

    except Exception as e:
        print(f"❌ Failed to standardize {file_path.name}: {e}")

# Main runner
async def main():
    files = list(INPUT_DIR.glob("*.json"))
    print(f"📂 Found {len(files)} resumes to standardize.\n")
    for file in files:
        await standardize_resume(file)

if __name__ == "__main__":
    asyncio.run(main())
