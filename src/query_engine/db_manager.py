import json
import uuid
from pathlib import Path
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME

class ResumeDBManager:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

    def insert_or_update_resume(self, resume: dict):
        """Insert a single resume using UUID as the unique _id."""
        if "_id" not in resume:
            resume["_id"] = str(uuid.uuid4())
        result = self.collection.insert_one(resume)
        print(f"✅ Inserted document ID: {result.inserted_id}")
        return result.inserted_id

    def bulk_insert(self, folder_path: str):
        """Insert all JSON files in a folder with UUIDs."""
        folder = Path(folder_path)
        files = list(folder.glob("*.json"))
        print(f"📂 Found {len(files)} resumes to insert.\n")

        inserted, failed = 0, 0

        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                if "_id" not in doc:
                    doc["_id"] = str(uuid.uuid4())
                self.collection.insert_one(doc)
                print(f"✅ Inserted: {doc.get('name', 'Unknown')} ({file.name})")
                inserted += 1
            except Exception as e:
                print(f"❌ Failed to insert {file.name}: {e}")
                failed += 1

        print(f"\n📊 Summary: Total = {len(files)}, Inserted = {inserted}, Failed = {failed}")

    def find(self, query: dict):
        """Find resumes matching a query."""
        print(f"🔍 Finding resumes matching: {query}")
        results = list(self.collection.find(query))
        print(f"🔎 Found {len(results)} resumes.\n")
        for res in results:
            print(f"- {res.get('name')} | {res.get('email')} | ID: {res.get('_id')}")
        return results

    def update_resume(self, update_data: dict):
        """Update a resume by _id."""
        _id = update_data.pop("_id", None)
        if not _id:
            print("❌ Update failed: '_id' field is required.")
            return
        result = self.collection.update_one({"_id": _id}, {"$set": update_data})
        if result.modified_count:
            print(f"✅ Updated resume with ID {_id}")
        else:
            print(f"⚠️ No resume found or no change for ID {_id}")

    def delete_resume(self, delete_data: dict):
        """Delete a resume by _id."""
        _id = delete_data.get("_id")
        if not _id:
            print("❌ Delete failed: '_id' field is required.")
            return
        result = self.collection.delete_one({"_id": _id})
        if result.deleted_count:
            print(f"🗑️ Deleted resume with ID {_id}")
        else:
            print(f"⚠️ No resume found with ID {_id}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to single resume JSON file")
    parser.add_argument("--folder", help="Path to folder containing multiple JSON files")
    parser.add_argument("--find", help="Find query in JSON format")
    parser.add_argument("--update", help="JSON string with _id and fields to update")
    parser.add_argument("--delete", help="JSON string with _id of resume to delete")

    args = parser.parse_args()
    db = ResumeDBManager()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            doc = json.load(f)
            db.insert_or_update_resume(doc)

    elif args.folder:
        db.bulk_insert(args.folder)

    elif args.find:
        try:
            query = json.loads(args.find)
            db.find(query)
        except Exception as e:
            print(f"❌ Invalid JSON for --find: {e}")

    elif args.update:
        try:
            update_data = json.loads(args.update)
            db.update_resume(update_data)
        except Exception as e:
            print(f"❌ Invalid JSON for --update: {e}")

    elif args.delete:
        try:
            delete_data = json.loads(args.delete)
            db.delete_resume(delete_data)
        except Exception as e:
            print(f"❌ Invalid JSON for --delete: {e}")

    else:
        print("⚠️ Please provide one of --file, --folder, --find, --update, or --delete.")
