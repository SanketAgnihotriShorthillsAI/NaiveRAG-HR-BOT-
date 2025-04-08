import os
import json
import uuid
from pathlib import Path
from tqdm import tqdm
import logging

# === CONFIGURATION ===
INPUT_DIR = Path("data/standardized_resumes")
OUTPUT_DIR = Path("data/chunked_resumes_v2")
LOG_FILE = "logs/chunker_v2.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === CHUNKING STRATEGY ===
SECTION_KEYS = [
    "summary", "education", "experience", "skills", "projects",
    "certifications", "languages", "social_profiles"
]

def create_chunk(name, section, content, filename):
    """Create a standardized chunk with UUID and metadata."""
    return {
        "chunk_id": str(uuid.uuid4()),
        "source": filename,
        "text": f"{name}\n\nSection: {section.capitalize()}\n\n{content}"
    }

def convert_section_to_text(section_key, section_value):
    """Convert structured section to human-readable chunk text."""
    if isinstance(section_value, str):
        return section_value

    if isinstance(section_value, list):
        lines = []
        for item in section_value:
            if isinstance(item, str):
                lines.append(f"- {item}")
            elif isinstance(item, dict):
                line = " | ".join([f"{k.capitalize()}: {v}" for k, v in item.items() if v])
                if line:
                    lines.append(f"- {line}")
        return "\n".join(lines)

    return ""

def chunk_resume(file_path: Path):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", "Unknown Candidate")
    filename = file_path.name
    chunks = []

    for key in SECTION_KEYS:
        content = data.get(key)
        if not content:
            continue

        text = convert_section_to_text(key, content)
        if text.strip():
            chunk = create_chunk(name, key, text, filename)
            chunks.append(chunk)

    return chunks

def process_all():
    files = list(INPUT_DIR.glob("*.json"))
    print(f"📁 Found {len(files)} standardized resumes to chunk...\n")

    for file in tqdm(files, desc="Chunking resumes"):
        output_path = OUTPUT_DIR / f"{file.stem}.jsonl"
        if output_path.exists():
            logging.info(f"⏩ Skipped {file.name} (already chunked)")
            continue

        try:
            chunks = chunk_resume(file)
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")
            logging.info(f"✅ Chunked {file.name} into {len(chunks)} chunks.")
        except Exception as e:
            logging.error(f"❌ Failed {file.name} | {str(e)}")

    print("\n✅ Chunking complete.")

if __name__ == "__main__":
    process_all()
