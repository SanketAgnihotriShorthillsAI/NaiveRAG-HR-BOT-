import os
import re
import json
import uuid
import logging
from tqdm import tqdm

# === CONFIG ===
INPUT_DIR = "data/parsed_resumes"
OUTPUT_DIR = "data/chunked_resumes"
LOG_FILE = "logs/chunking.logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === SECTION HEADERS ===
SECTION_KEYWORDS = [
    "SUMMARY", "OBJECTIVE", "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
    "INTERNSHIP", "PROJECTS", "EDUCATION", "SKILLS", "CERTIFICATIONS", "ACHIEVEMENTS",
    "POSITIONS OF RESPONSIBILITY", "EXTRACURRICULAR ACTIVITIES", "PUBLICATIONS",
    "PERSONAL DETAILS", "LANGUAGES", "HOBBIES", "DECLARATION", "RESPONSIBILITIES",
    "ADDITIONAL DETAILS", "SOFT SKILLS"
]
HEADER_PATTERN = re.compile(rf"^({'|'.join(map(re.escape, SECTION_KEYWORDS))})\s*$", re.IGNORECASE | re.MULTILINE)

# === CHUNKING ===
def chunk_raw_text(filename, raw_text):
    chunks = []
    file_prefix = os.path.splitext(filename)[0].replace(" ", "_")

    raw_text = raw_text.strip()
    sections = []
    matches = list(HEADER_PATTERN.finditer(raw_text))

    if not matches:
        # No headers found, fallback to whole resume
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "source": filename,
            "text": f"{file_prefix}\n\n{raw_text}"
        })
        return chunks

    # Add last boundary
    positions = [m.start() for m in matches] + [len(raw_text)]

    for i in range(len(matches)):
        start = positions[i]
        end = positions[i + 1]
        section_text = raw_text[start:end].strip()

        # Filter out meaningless or tiny sections
        if len(section_text.split()) < 5:
            continue

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "source": filename,
            "text": f"{file_prefix}\n\n{section_text}"
        }
        chunks.append(chunk)

    return chunks

# === PROCESSING ===
def process_file(input_path, output_path, filename):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_text = data.get("raw_text", "").strip()
        if not raw_text:
            raise ValueError("No raw_text found.")

        chunks = chunk_raw_text(filename, raw_text)

        with open(output_path, "w", encoding="utf-8") as out:
            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")

        logging.info(f"✅ Chunked: {filename} into {len(chunks)} chunks.")
    except Exception as e:
        logging.error(f"❌ Failed: {filename} | {str(e)}")

# === MAIN ===
def run():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    logging.info(f"Found {len(files)} parsed resumes.")

    for file in tqdm(files, desc="Chunking resumes"):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file.replace(".json", ".jsonl"))

        if os.path.exists(output_path):
            logging.info(f"⏩ Skipped (already processed): {file}")
            continue

        process_file(input_path, output_path, file)

    logging.info("🎉 Chunking complete.")

if __name__ == "__main__":
    run()
