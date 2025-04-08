import os
import json
from pathlib import Path

# === CONFIGURATION ===
STANDARDIZED_DIR = Path("data/standardized_resumes")
REQUIRED_KEYS = {
    "name": str,
    "email": str,
    "phone": str,
    "location": str,
    "summary": str,
    "education": list,
    "experience": list,
    "skills": list,
    "projects": list,
    "certifications": list,
    "languages": list,
    "social_profiles": list
}

def validate_entry_list(entries, required_keys):
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        if not required_keys.issubset(entry.keys()):
            return False
    return True

def validate_resume(resume):
    if not isinstance(resume, dict):
        return False, "Root is not a JSON object"

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in resume:
            return False, f"Missing key: {key}"
        if not isinstance(resume[key], expected_type):
            return False, f"Key '{key}' should be of type {expected_type.__name__}"

    if not validate_entry_list(resume["education"], {"degree", "institution", "year"}):
        return False, "Invalid format in 'education'"

    if not validate_entry_list(resume["experience"], {"title", "company", "duration", "description"}):
        return False, "Invalid format in 'experience'"

    if not validate_entry_list(resume["projects"], {"title", "description"}):
        return False, "Invalid format in 'projects'"

    if not validate_entry_list(resume["certifications"], {"title"}):  # issuer/year/link are optional
        return False, "Invalid format in 'certifications'"

    if not validate_entry_list(resume["social_profiles"], {"platform", "link"}):
        return False, "Invalid format in 'social_profiles'"

    return True, "Valid"

def main():
    files = list(STANDARDIZED_DIR.glob("*.json"))
    print(f"🔍 Found {len(files)} standardized resume files\n")

    for file in files:
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            valid, message = validate_resume(data)
            status = "✅" if valid else "❌"
            print(f"{status} {file.name}: {message}")
        except Exception as e:
            print(f"❌ {file.name}: Failed to parse - {e}")

if __name__ == "__main__":
    main()
