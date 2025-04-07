from fastapi import FastAPI, Request
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.pipeline import RAGPipeline
import uvicorn

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str
    use_openai: bool = False
    use_optimize_llm: bool = True
    top_k: int = 50

# Initialize pipeline once
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = RAGPipeline(use_openai=False, use_optimize_llm=True)

@app.post("/query")
async def query_rag(request: QueryRequest):
    global pipeline

    # Re-init pipeline if settings changed (optional)
    pipeline.generator.use_openai = request.use_openai
    pipeline.generator.use_optimize_llm = request.use_optimize_llm

    print(f"🔍 Query: {request.query}")
    print(f"🔧 top_k: {request.top_k}, use_openai: {request.use_openai}, use_optimize_llm: {request.use_optimize_llm}")

    retrieved_chunks = pipeline.retriever.query(request.query, top_k=request.top_k)
    answer, response = await pipeline.generator.generate_response(request.query, retrieved_chunks)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
