import json
import sys
import os
import time
import numpy as np
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rag import RAGSystem

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

def get_embedding(text):
    time.sleep(3)
    response = client.embeddings.create(model="mistral-embed", inputs=[text])
    return np.array(response.data[0].embedding, dtype="float32")

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def ask_with_retry(rag, question, retries=5, delay=20):
    for attempt in range(retries):
        try:
            return rag.ask(question)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"     ⏳ Rate limit, attente {delay}s...")
                time.sleep(delay)
            else:
                raise

def evaluate():
    with open("tests/test_set.json", "r") as f:
        test_set = json.load(f)

    rag = RAGSystem()
    results = []

    print(f"\n{'='*70}")
    print(f"{'ÉVALUATION DU SYSTÈME RAG':^70}")
    print(f"{'='*70}\n")

    for i, item in enumerate(test_set):
        question         = item["question"]
        reponse_attendue = item["reponse_attendue"]

        print(f"[{i+1:02d}] {question}")
        result           = ask_with_retry(rag, question)
        reponse_generee  = result["answer"]

        emb_attendue = get_embedding(reponse_attendue)
        emb_generee  = get_embedding(reponse_generee)
        score_sim    = cosine_similarity(emb_attendue, emb_generee)

        if score_sim >= 0.90:
            label = "✅ Correcte"
        elif score_sim >= 0.75:
            label = "⚠️  Partiellement correcte"
        else:
            label = "❌ Incorrecte"

        results.append({
            "question":         question,
            "reponse_attendue": reponse_attendue,
            "reponse_generee":  reponse_generee,
            "score_similarite": round(score_sim, 4),
            "evaluation":       label,
        })

        print(f"     Score similarité : {score_sim:.4f} → {label}\n")
        time.sleep(15)

    scores     = [r["score_similarite"] for r in results]
    moyenne    = np.mean(scores)
    correctes  = sum(1 for r in results if "✅" in r["evaluation"])
    partielles = sum(1 for r in results if "⚠️"  in r["evaluation"])
    incorrectes = sum(1 for r in results if "❌" in r["evaluation"])

    print(f"{'='*70}")
    print(f"RÉSUMÉ")
    print(f"{'='*70}")
    print(f"  Score moyen de similarité : {moyenne:.4f}")
    print(f"  ✅ Correctes               : {correctes}/{len(results)}")
    print(f"  ⚠️  Partiellement correctes : {partielles}/{len(results)}")
    print(f"  ❌ Incorrectes             : {incorrectes}/{len(results)}")
    print(f"{'='*70}\n")

    with open("tests/evaluation_results.json", "w") as f:
        json.dump({
            "score_moyen": round(moyenne, 4),
            "correctes": correctes,
            "partielles": partielles,
            "incorrectes": incorrectes,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print("📄 Résultats sauvegardés dans tests/evaluation_results.json")
    return moyenne

if __name__ == "__main__":
    evaluate()
