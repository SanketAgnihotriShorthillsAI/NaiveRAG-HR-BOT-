import logging
import os
from pipeline.retriever import Retriever
from pipeline.generator import Generator

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RAGPipeline:
    def __init__(self, use_openai=False, use_optimize_llm=False):
        logging.info("🚀 Initializing RAG Pipeline...")
        self.retriever = Retriever()
        self.generator = Generator(
            use_openai=use_openai,
            use_optimize_llm=use_optimize_llm
            )
        logging.info("✅ RAG Pipeline initialized!")

    def process_query(self, query):
        logging.info(f"\n⏳ Processing query: {query}")
        retrieved_chunks = self.retriever.query(query, top_k=5)
        answer, response = self.generator.generate_response(query, retrieved_chunks)
        return response
