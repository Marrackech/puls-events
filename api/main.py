from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rag import RAGSystem
from scripts.build_index import build_chunks, embed_texts
import pandas as pd
import faiss
import pickle
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Puls-Events RAG API",
    description="Assistant intelligent pour les événements culturels à Paris",
    version="1.0.0"
)

rag = RAGSystem()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list

@app.get("/")
def root():
    return {"message": "Puls-Events RAG API", "status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
    result = rag.ask(request.question)
    return result

@app.post("/rebuild")
def rebuild():
    try:
        df = pd.read_csv("data/events.csv")
        chunks, metadatas = build_chunks(df)
        embeddings = embed_texts(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss_path = os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss_index")
        faiss.write_index(index, f"{faiss_path}/index.faiss")
        with open(f"{faiss_path}/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        with open(f"{faiss_path}/metadatas.pkl", "wb") as f:
            pickle.dump(metadatas, f)
        rag._load_index()
        return {"message": f"Index reconstruit avec {index.ntotal} événements"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "events_indexed": rag.index.ntotal}
