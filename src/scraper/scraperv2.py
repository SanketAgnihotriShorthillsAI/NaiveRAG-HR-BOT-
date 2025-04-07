import os
import re
import json
import logging
import unicodedata
import fitz  # PyMuPDF
from tqdm import tqdm
from docx import Document

# ========== SETUP ==========
INPUT_DIR = "data/resumes"
OUTPUT_DIR = "data/parsed_resumes"
LOG_FILE = "logs/scraping.logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ========== LOGGING ==========
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== SECTION HEADERS ==========
SECTION_KEYWORDS = [
    "SUMMARY", "OBJECTIVE", "CAREER OBJECTIVE", "PROFESSIONAL SUMMARY", "PROFILE", "ABOUT ME",
    "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT HISTORY",
    "INDUSTRIAL EXPERIENCE", "INTERNSHIP", "INTERNSHIPS", "INTERNSHIP/S", "PROJECT EXPERIENCE",
    "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS",
    "EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATIONS", "ACADEMIC QUALIFICATIONS",
    "SKILLS", "KEY SKILLS", "TECHNICAL SKILLS", "PROGRAMMING SKILLS", "TOOLS AND TECHNOLOGIES",
    "EXPERTISE", "CORE COMPETENCIES", "SOFTWARE PROFICIENCY",
    "CERTIFICATIONS", "CERTIFICATION", "LICENSES", "COURSES", "TRAINING",
    "ACHIEVEMENTS", "AWARDS", "RECOGNITION", "HONORS", "ACCOMPLISHMENTS", "ACADEMIC ACHIEVEMENTS",
    "POSITIONS OF RESPONSIBILITY", "LEADERSHIP", "EXTRA RESPONSIBILITIES", "ROLES AND RESPONSIBILITIES",
    "EXTRACURRICULAR ACTIVITIES", "CO-CURRICULAR ACTIVITIES", "HOBBIES", "INTERESTS",
    "PUBLICATIONS", "RESEARCH WORK", "RESEARCH PAPERS", "CONFERENCES", "THESIS",
    "PERSONAL DETAILS", "PERSONAL INFORMATION", "CONTACT INFORMATION", "ADDRESS", "DECLARATION",
    "LANGUAGES", "LANGUAGE PROFICIENCY", "SOFT SKILLS",
    "REFERENCES", "REFERENCE"
]

# ========== CLEANING & FORMATTING ==========

def clean_special_chars(text):
    return ''.join(
        ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t'
    )

def normalize_whitespace(text):
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return "\n".join(line.strip() for line in text.splitlines())

def space_out_section_headers(text):
    for keyword in SECTION_KEYWORDS:
        text = re.sub(rf"(?<!\n)({keyword})(?!\n)", r"\n\n\1\n", text, flags=re.IGNORECASE)
    return text

# ========== EXTRACTORS ==========

def extract_from_pdf(path):
    doc = fitz.open(path)
    blocks = []
    for page in doc:
        blocks.extend(page.get_text("blocks"))
    blocks.sort(key=lambda b: (b[1], b[0]))  # y, then x
    return "\n".join([b[4].strip() for b in blocks if b[4].strip()])

def extract_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(path)
    elif ext == ".docx":
        return extract_from_docx(path)
    else:
        raise ValueError("Unsupported file format: " + ext)

# ========== MAIN ==========

def process_resume(file_path, output_path):
    try:
        raw_text = extract_text(file_path)
        raw_text = clean_special_chars(raw_text)
        raw_text = normalize_whitespace(raw_text)
        raw_text = space_out_section_headers(raw_text)

        with open(output_path, "w", encoding="utf-8", ) as f:
            json.dump({"raw_text": raw_text}, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Processed: {file_path}")
    except Exception as e:
        logging.error(f"❌ Error in {file_path}: {str(e)}")

def batch_process():
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".pdf", ".docx"))
    ]
    logging.info(f"Resumes found: {len(files)}")

    for file in tqdm(files, desc="Processing resumes"):
        input_path = os.path.join(INPUT_DIR, file)
        name = os.path.splitext(file)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")

        if os.path.exists(output_path):
            logging.info(f"⏩ Skipped (already processed): {file}")
            continue

        process_resume(input_path, output_path)

    logging.info("🎉 Batch processing complete.\n")

if __name__ == "__main__":
    batch_process()
