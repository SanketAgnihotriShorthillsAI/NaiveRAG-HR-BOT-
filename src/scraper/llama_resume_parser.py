import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not api_key:
    raise ValueError("❌ LLAMA_CLOUD_API_KEY is missing from .env")

# Config paths
RESUME_DIR = Path("data/resumes")
OUTPUT_DIR = Path("data/processed_resumes")
SUPPORTED_EXTENSIONS = [".pdf", ".docx"]

# LlamaParse setup
parser = LlamaParse(
    api_key=api_key,
    result_type="markdown",
    do_not_unroll_columns=True
)

def parse_resume(file_path):
    try:
        documents = parser.load_data(file_path)
        combined_text = "\n".join([doc.text for doc in documents])
        return {
            "file": file_path.name,
            "content": combined_text
        }
    except Exception as e:
        print(f"❌ Failed to parse {file_path.name}: {e}")
        return None

def save_to_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resume_files = [f for f in RESUME_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    print(f"🔍 Found {len(resume_files)} resume files to process.\n")

    for resume in resume_files:
        output_path = OUTPUT_DIR / f"{resume.stem}.json"
        if output_path.exists():
            print(f"⏩ Skipping {resume.name} (already processed)")
            continue

        print(f"📄 Processing: {resume.name}")
        parsed_data = parse_resume(resume)
        if parsed_data:
            save_to_json(parsed_data, output_path)
            print(f"✅ Saved: {output_path.name}")
        print("-" * 40)

if __name__ == "__main__":
    main()
