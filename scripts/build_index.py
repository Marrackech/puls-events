import pandas as pd
import faiss
import numpy as np
import pickle
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss_index")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

def embed_texts(texts, batch_size=20):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(model="mistral-embed", inputs=batch)
        all_embeddings.extend([e.embedding for e in response.data])
        print(f"  → {min(i+batch_size, len(texts))}/{len(texts)} vectorisés")
    return np.array(all_embeddings, dtype="float32")

def build_chunks(df):
    chunks, metadatas = [], []
    for _, row in df.iterrows():
        text = f"Titre: {row['titre']}\nDescription: {row['description']}\nLieu: {row['lieu']}\nVille: {row['ville']}\nDate: {row['date_debut']}"
        chunks.append(text)
        metadatas.append({
            "id": row["id"], "titre": row["titre"],
            "lieu": row["lieu"], "ville": row["ville"],
            "date_debut": row["date_debut"], "url": row["url"],
        })
    return chunks, metadatas

def run_build():
    df = pd.read_csv("data/events.csv")
    print(f"📂 {len(df)} événements chargés")
    chunks, metadatas = build_chunks(df)
    print("🔢 Vectorisation via Mistral Embed...")
    embeddings = embed_texts(chunks)
    print("📦 Construction de l'index FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(FAISS_PATH, exist_ok=True)
    faiss.write_index(index, f"{FAISS_PATH}/index.faiss")
    with open(f"{FAISS_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(f"{FAISS_PATH}/metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    print(f"✅ {index.ntotal} vecteurs indexés dans {FAISS_PATH}/")
    return index.ntotal

if __name__ == "__main__":
    run_build()  # pragma: no cover
