import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/retriever.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Retriever:
    def __init__(self, db_path="data/vector_db", collection_name="resumes"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        logging.info("🔄 Loaded embedding model: BAAI/bge-base-en-v1.5")

        total = self.collection.count()
        if total == 0:
            logging.warning("⚠️ No embeddings in ChromaDB!")
        else:
            logging.info(f"✅ ChromaDB contains {total} documents.")

    def get_embedding(self, text):
        try:
            prompt = f"Represent this query for retrieval: {text}"
            return self.embedding_model.encode(prompt, normalize_embeddings=True).tolist()
        except Exception as e:
            logging.error(f"❌ Embedding error: {e}")
            return None

    def query(self, query_text, top_k=100):
        logging.info(f"🔍 Querying: {query_text}")
        
        embedding = self.get_embedding(query_text)
        if not embedding:
            return [], []
        print(f"retrieving top {top_k} chunks")
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)

        chunks, sources = [], []
        for meta in results.get("metadatas", [[]])[0]:
            chunks.append(meta["text"])
            sources.append(meta.get("source", "Unknown Source"))

        logging.info(f"✅ Retrieved {len(chunks)} chunks.")
        return chunks
