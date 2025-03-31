import os
import json
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingGenerator:
    def __init__(self, input_dir="data/chunked_resumes", output_dir="data/embeddings", model_name="BAAI/bge-base-en-v1.5"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name

        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def load_jsonl(self, filepath):
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            logging.error(f"❌ Error loading {filepath}: {e}")
            return []

    def save_jsonl(self, filename, data):
        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                for entry in data:
                    json.dump(entry, file, ensure_ascii=False)
                    file.write("\n")
            logging.info(f"✅ Saved embeddings to {output_path}")
        except Exception as e:
            logging.error(f"❌ Error saving embeddings for {filename}: {e}")

    def embed_text(self, text):
        try:
            prompt = f"Represent this chunk for retrieval: {text}"
            return self.model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            logging.error(f"❌ Embedding error: {e}")
            return None

    def process_file(self, filepath):
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(output_filepath):
            logging.info(f"⏭️ Skipping {filename}, embeddings already exist.")
            return

        chunks = self.load_jsonl(filepath)
        if not chunks:
            logging.warning(f"⚠️ Skipping {filename}, no valid chunks.")
            return

        embedded = []
        for chunk in tqdm(chunks, desc=f"Embedding {filename}", leave=False):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            emb = self.embed_text(text)
            if emb:
                embedded.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "text": text,
                    "source": chunk.get("source"),
                    "embedding": emb
                })

        if embedded:
            self.save_jsonl(filename, embedded)

    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".jsonl")]
        for file in files:
            self.process_file(os.path.join(self.input_dir, file))

if __name__ == "__main__":
    EmbeddingGenerator().run()
