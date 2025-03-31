import os
import json
import logging
import chromadb
from tqdm import tqdm

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorStore:
    def __init__(self, input_dir="data/embeddings", db_dir="data/vector_db", collection_name="resumes"):
        self.input_dir = input_dir
        self.db_dir = db_dir
        self.collection_name = collection_name

        os.makedirs(self.db_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def load_jsonl(self, filepath):
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            logging.error(f"❌ Failed to load {filepath}: {e}")
            return []

    def chunk_exists(self, chunk_id):
        try:
            existing = self.collection.get(ids=[chunk_id])
            return bool(existing["ids"])
        except:
            return False

    def add_embeddings(self, records):
        added, skipped = 0, 0
        for rec in tqdm(records, desc="📥 Adding to Vector DB"):
            chunk_id = rec["chunk_id"]
            if self.chunk_exists(chunk_id):
                skipped += 1
                continue

            self.collection.add(
                ids=[chunk_id],
                embeddings=[rec["embedding"]],
                documents=[rec["text"]],
                metadatas=[{
                    "source": rec["source"],
                    "text": rec["text"]
                }]
            )
            added += 1
        logging.info(f"✅ Added {added}, Skipped {skipped}")

    def process(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".jsonl")]
        for file in files:
            logging.info(f"🔍 Processing {file}")
            data = self.load_jsonl(os.path.join(self.input_dir, file))
            if data:
                self.add_embeddings(data)

    def run(self):
        self.process()
        logging.info("🎯 Vector store population complete.")

if __name__ == "__main__":
    VectorStore().run()
