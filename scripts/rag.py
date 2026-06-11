import faiss
import pickle
import numpy as np
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss_index")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
TOP_K = int(os.getenv("RAG_TOP_K", 5))

class RAGSystem:
    def __init__(self):
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index()

    def _embed(self, texts):
        response = self.client.embeddings.create(model="mistral-embed", inputs=texts)
        return np.array([e.embedding for e in response.data], dtype="float32")

    def _load_index(self):
        self.index = faiss.read_index(f"{FAISS_PATH}/index.faiss")
        with open(f"{FAISS_PATH}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(f"{FAISS_PATH}/metadatas.pkl", "rb") as f:
            self.metadatas = pickle.load(f)
        print(f"✅ Index chargé ({self.index.ntotal} événements)")

    def search(self, query, k=TOP_K):
        vec = self._embed([query])
        distances, indices = self.index.search(vec, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk":    self.chunks[idx],
                    "metadata": self.metadatas[idx],
                    "score":    float(distances[0][i]),
                })
        return results

    def ask(self, question):
        results = self.search(question)
        context = "\n\n---\n\n".join([r["chunk"] for r in results])
        prompt = f"""Tu es un assistant culturel spécialisé dans les événements à Paris.
Réponds à la question en t'appuyant uniquement sur les événements ci-dessous.
Si aucun événement ne correspond, dis-le clairement.

ÉVÉNEMENTS DISPONIBLES :
{context}

QUESTION : {question}

RÉPONSE :"""
        response = self.client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        return {"question": question, "answer": answer, "sources": [r["metadata"] for r in results]}

def run_demo():
    rag = RAGSystem()
    questions = [
        "Quels concerts sont prévus à Paris ?",
        "Y a-t-il des événements pour les enfants ?",
        "Quels événements culturels ont lieu en juin 2026 ?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ {q}")
        result = rag.ask(q)
        print(f"💬 {result['answer']}")
        print(f"📌 Sources : {[s['titre'] for s in result['sources']]}")
    return True

if __name__ == "__main__":
    run_demo()  # pragma: no cover
