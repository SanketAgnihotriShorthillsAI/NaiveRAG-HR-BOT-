import os
import sys
from pipeline.pipeline import RAGPipeline

# Fix sys.path for relative import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def check_chroma_db(path="data/vector_db"):
    """Check if Chroma vector DB exists and is non-empty."""
    return any(
        f.startswith("chroma") or f.endswith(".sqlite3") or os.path.isdir(os.path.join(path, f))
        for f in os.listdir(path)
    )

def main():
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists("data/vector_db") or not check_chroma_db("data/vector_db"):
        print("❌ Vector DB not found or empty. Run vector_store.py first.")
        sys.exit(1)

    print("\n🔍 HR RAG Bot is ready.")
    use_openai = input("Use OpenAI for generation? (y/n): ").strip().lower() == "y"
    use_optimize_llm = input("Use OptimizeRag Azure LLM instead? (y/n): ").strip().lower() == "y"
    
    query = input("\nEnter your query (or leave blank for default): ").strip()
    if not query:
        query = "Find candidates with React and UX design internships"

    # Initialize and run pipeline
    rag = RAGPipeline(use_openai=use_openai, use_optimize_llm=use_optimize_llm)
    print("\n⏳ Processing query...")
    result = rag.process_query(query)

    print("\n🧠 Generated Response:\n")
    print(result)

if __name__ == "__main__":
    main()
