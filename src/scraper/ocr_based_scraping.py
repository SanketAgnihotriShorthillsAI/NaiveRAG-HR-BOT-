import os
import sys
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import json

# === CONFIG ===
INPUT_DIR = Path("data/resumes")
OUTPUT_DIR = Path("data/ocr_parsed_resumes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        images = convert_from_path(str(pdf_path), dpi=300)
        full_text = ""
        for img in images:
            text = pytesseract.image_to_string(img)
            full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        return ""

def process_single_resume(file_path: Path):
    output_path = OUTPUT_DIR / f"{file_path.stem}.json"
    if output_path.exists():
        print(f"⏩ Skipping {file_path.name} (already parsed)")
        return

    print(f"📄 Processing: {file_path.name}")
    text = extract_text_from_pdf(file_path)

    if text:
        data = {
            "file": file_path.name,
            "content": text
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {output_path.name}")
    else:
        print(f"⚠️ No text extracted for {file_path.name}")
    print("-" * 40)

def main():
    if len(sys.argv) > 1:
        # Specific file mode
        filename = sys.argv[1]
        file_path = INPUT_DIR / filename
        if file_path.exists() and file_path.suffix.lower() == ".pdf":
            process_single_resume(file_path)
        else:
            print(f"❌ File '{filename}' not found in {INPUT_DIR}")
    else:
        # Bulk processing mode
        pdf_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() == ".pdf"]
        print(f"🔍 Found {len(pdf_files)} PDF resumes to OCR.\n")
        for pdf in pdf_files:
            process_single_resume(pdf)

if __name__ == "__main__":
    main()
