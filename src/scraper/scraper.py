import os
import re
import json
import unicodedata
import logging
from tqdm import tqdm
from datetime import datetime
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scraping.logs",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Preprocessing Helpers ---
def clean_special_chars(text):
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')

def normalize_whitespace(text):
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return "\n".join(line.strip() for line in text.splitlines())

def space_out_section_headers(text):
    section_keywords = [
        "EDUCATION", "WORK EXPERIENCE", "EXPERIENCE", "PROJECTS", "SKILLS",
        "CERTIFICATIONS", "SUMMARY", "ADDITIONAL DETAILS", "POSITION OF RESPONSIBILITY",
        "EXTRACURRICULAR ACTIVITIES", "ACHIEVEMENTS", "MEMBERSHIP", "HOBBIES",
        "RESPONSIBILITIES", "INTERNSHIPS", "TECHNICAL SKILLS"
    ]
    for keyword in section_keywords:
        text = re.sub(rf"(?<!\n)({keyword})(?!\n)", r"\n\n\1\n", text, flags=re.IGNORECASE)
    return text

# --- Text Extraction ---
def extract_text_blocks(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return partition_pdf(file_path, strategy="hi_res")
    elif ext == ".docx":
        return partition_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

# --- Main Processing ---
def process_resume(file_path, output_path):
    try:
        logging.info(f"Processing: {file_path}")
        elements = extract_text_blocks(file_path)
        text_blocks = [el.text for el in elements if el.text]
        full_text = "\n".join(text_blocks)

        # Preprocess text
        full_text = clean_special_chars(full_text)
        full_text = normalize_whitespace(full_text)
        full_text = space_out_section_headers(full_text)

        # Output JSON with just raw_text
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"raw_text": full_text}, f, indent=2)

        logging.info(f"✅ Saved to: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to process {file_path}: {str(e)}")

# --- Batch Runner ---
def batch_process():
    input_dir = "data/resumes"
    output_dir = "data/parsed_resumes"
    os.makedirs(output_dir, exist_ok=True)

    resume_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".pdf", ".docx"))
    ]

    logging.info(f"Started batch processing at {datetime.now().isoformat()}")
    logging.info(f"Found {len(resume_files)} resumes to check.")

    for filename in tqdm(resume_files, desc="Processing resumes"):
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        if os.path.exists(output_path):
            logging.info(f"Skipped (already processed): {filename}")
            continue

        process_resume(input_path, output_path)

    logging.info("🎉 Batch processing complete.\n")

# --- Entry Point ---
if __name__ == "__main__":
    batch_process()
