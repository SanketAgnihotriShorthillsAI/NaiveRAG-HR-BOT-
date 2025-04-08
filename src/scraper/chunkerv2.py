import os
import json
import uuid
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
INPUT_DIR = Path("data/standardized_resumes")
OUTPUT_DIR = Path("data/chunked_resumes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === CHUNKING LOGIC ===
def chunk_resume(data: dict, filename: str) -> list:
    name = data.get("name", "").strip()
    chunks = []

    def make_chunk(section: str, content: str):
        text = f"{name} - {section}\n\n{content.strip()}"
        return {
            "chunk_id": str(uuid.uuid4()),
            "source": filename,
            "name": name,
            "section": section.lower(),
            "text": text
        }

    if data.get("summary"):
        chunks.append(make_chunk("Summary", data["summary"]))

    if data.get("education"):
        edu_text = "\n".join(
            f"{ed.get('degree', '')}, {ed.get('institution', '')}, {ed.get('year', '')}".strip(", ")
            for ed in data["education"]
        )
        chunks.append(make_chunk("Education", edu_text))

    if data.get("experience"):
        exp_text = "\n".join(
            f"{exp.get('title', '')} at {exp.get('company', '')} ({exp.get('duration', '')})"
            + (f" - {exp.get('description', '')}" if exp.get("description") else "")
            for exp in data["experience"]
        )
        chunks.append(make_chunk("Experience", exp_text))

    if data.get("skills"):
        skill_text = ", ".join(data["skills"])
        chunks.append(make_chunk("Skills", skill_text))

    if data.get("projects"):
        proj_text = "\n".join(
            f"{proj.get('title', '')}: {proj.get('description', '')}"
            for proj in data["projects"]
        )
        chunks.append(make_chunk("Projects", proj_text))

    if data.get("certifications"):
        cert_text = "\n".join(
            f"{cert.get('title', '')}, {cert.get('issuer', '')}, {cert.get('year', '')}".strip(", ")
            for cert in data["certifications"]
        )
        chunks.append(make_chunk("Certifications", cert_text))

    if data.get("languages"):
        lang_text = ", ".join(data["languages"])
        chunks.append(make_chunk("Languages", lang_text))

    if data.get("social_profiles"):
        social_text = "\n".join(
            f"{profile.get('platform', '')}: {profile.get('link', '')}"
            for profile in data["social_profiles"]
        )
        chunks.append(make_chunk("Social Profiles", social_text))

    return chunks

# === MAIN PROCESSING ===
def process_file(file_path: Path):
    output_path = OUTPUT_DIR / file_path.with_suffix(".jsonl").name

    if output_path.exists():
        print(f"⏩ Skipping {file_path.name} (already processed)")
        return

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        chunks = chunk_resume(data, file_path.name)

        with open(output_path, "w", encoding="utf-8") as out:
            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")

        print(f"✅ Chunked {file_path.name} into {len(chunks)} chunks.")

    except Exception as e:
        print(f"❌ Failed {file_path.name}: {e}")

def main():
    files = list(INPUT_DIR.glob("*.json"))
    print(f"🔍 Found {len(files)} standardized resumes to chunk.\n")

    for file in tqdm(files, desc="Chunking Resumes"):
        process_file(file)

if __name__ == "__main__":
    main()
