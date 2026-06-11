# Puls-Events — Assistant RAG pour événements culturels à Paris

> POC développé par **Haroun** — Data Scientist Freelance | Juin 2026

Système de recommandation d'événements culturels basé sur une architecture RAG (Retrieval-Augmented Generation) combinant FAISS, Mistral et FastAPI.

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Architecture du système](#2-architecture-du-système)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Utilisation](#6-utilisation)
7. [API REST](#7-api-rest)
8. [Tests](#8-tests)
9. [Docker](#9-docker)
10. [Évaluation](#10-évaluation)
11. [Résultats](#11-résultats)
12. [Pistes d'amélioration](#12-pistes-damélioration)

---

## 1. Contexte et objectifs

Puls-Events est une plateforme de recommandations culturelles personnalisées. Ce POC démontre la faisabilité d'un chatbot intelligent capable de répondre à des questions sur les événements culturels à Paris, en s'appuyant sur les données temps réel de l'API Open Agenda.

**Problème :** Les utilisateurs peinent à trouver rapidement des événements culturels adaptés à leurs envies parmi des centaines d'événements disponibles.

**Solution :** Un assistant conversationnel qui comprend les questions en langage naturel, recherche les événements les plus pertinents par similarité sémantique, et génère une réponse claire et contextualisée.

**Objectifs techniques :**
- Récupérer et indexer les événements culturels parisiens via Open Agenda
- Permettre une recherche sémantique rapide via FAISS
- Générer des réponses naturelles et pertinentes via Mistral
- Exposer le système via une API REST robuste et documentée
- Conteneuriser la solution pour un déploiement facile

---

## 2. Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                        UTILISATEUR                          │
│                    Question en langage naturel              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST /ask
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    API REST (FastAPI)                        │
│                      localhost:8000                         │
│         /ask   /rebuild   /health   /docs (Swagger)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      SYSTÈME RAG                            │
│                                                             │
│  1. Embedding de la question  →  Mistral Embed              │
│  2. Recherche Top-K voisins   →  FAISS IndexFlatL2          │
│  3. Construction du prompt    →  Contexte + Question        │
│  4. Génération de la réponse  →  Mistral Large Latest       │
└───────────┬─────────────────────────┬───────────────────────┘
            │                         │
            ▼                         ▼
┌───────────────────┐     ┌───────────────────────┐
│   Index FAISS     │     │     API Mistral        │
│  vectorstore/     │     │  mistral-embed         │
│  777 événements   │     │  mistral-large-latest  │
│  dimension 1024   │     └───────────────────────┘
└───────────────────┘
            ▲
            │ Indexation
            │
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE DE DONNÉES                       │
│                                                             │
│  Open Agenda API                                            │
│       │                                                     │
│       ▼                                                     │
│  fetch_events.py  →  data/events.csv  (777 événements)      │
│       │                                                     │
│       ▼                                                     │
│  build_index.py   →  Mistral Embed  →  FAISS Index          │
└─────────────────────────────────────────────────────────────┘
```

### Composants principaux

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Collecte | Open Agenda API + Requests | Récupération des événements |
| Stockage | Pandas + CSV | Structuration des données |
| Embeddings | Mistral Embed (dim. 1024) | Vectorisation des textes |
| Index vectoriel | FAISS IndexFlatL2 | Recherche par similarité cosinus |
| LLM | Mistral Large Latest | Génération des réponses |
| Orchestration | Python natif | Pipeline RAG |
| API | FastAPI + Uvicorn | Exposition du système |
| Conteneur | Docker | Déploiement reproductible |

---

## 3. Structure du projet

```
puls-events/
├── api/
│   └── main.py                  # API REST FastAPI
├── scripts/
│   ├── fetch_events.py          # Collecte des données Open Agenda
│   ├── build_index.py           # Vectorisation et indexation FAISS
│   └── rag.py                   # Système RAG (recherche + génération)
├── tests/
│   ├── test_rag.py              # Tests unitaires (100% couverture)
│   ├── evaluate_rag.py          # Script d'évaluation automatique
│   ├── test_set.json            # Jeu de test annoté (10 Q/R)
│   └── evaluation_results.json  # Résultats d'évaluation
├── vectorstore/
│   └── faiss_index/
│       ├── index.faiss          # Index FAISS binaire
│       ├── chunks.pkl           # Textes vectorisés
│       └── metadatas.pkl        # Métadonnées des événements
├── data/
│   └── events.csv               # 777 événements collectés
├── docs/                        # Documentation complémentaire
├── .env.example                 # Template de configuration
├── .gitignore
├── Dockerfile
└── requirements.txt
```

---

## 4. Installation

### Prérequis

- Python 3.11
- Conda
- Docker + Colima (Mac) ou Docker Desktop (Windows/Linux)
- Clé API Mistral — [console.mistral.ai](https://console.mistral.ai)
- Clé API Open Agenda — [openagenda.com](https://openagenda.com)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/haroun/puls-events.git
cd puls-events

# 2. Créer l'environnement conda
conda create -n puls-events python=3.11 -y
conda activate puls-events

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
nano .env  # Remplissez vos clés API
```

---

## 5. Configuration

Éditez le fichier `.env` :

```env
MISTRAL_API_KEY=votre_cle_mistral
OPENAGENDA_API_KEY=votre_cle_openagenda
TARGET_LOCATION=Paris
MISTRAL_MODEL=mistral-large-latest
RAG_TOP_K=5
FAISS_INDEX_PATH=vectorstore/faiss_index
```

> ⚠️ Ne versionnez jamais le fichier `.env` — il est listé dans `.gitignore`.

---

## 6. Utilisation

### Étape 1 — Collecter les données

```bash
python scripts/fetch_events.py
```

Récupère les événements culturels parisiens depuis Open Agenda (10 agendas, ~777 événements) et les sauvegarde dans `data/events.csv`.

### Étape 2 — Construire l'index vectoriel

```bash
python scripts/build_index.py
```

Vectorise chaque événement avec Mistral Embed et construit l'index FAISS dans `vectorstore/faiss_index/`.

### Étape 3 — Tester le chatbot en ligne de commande

```bash
python scripts/rag.py
```

Lance une démonstration avec 3 questions types et affiche les réponses générées avec les sources.

---

## 7. API REST

### Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation Swagger interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

#### `GET /` — Vérification de l'API
```bash
curl http://localhost:8000/
# {"message": "Puls-Events RAG API", "status": "ok"}
```

#### `GET /health` — État du système
```bash
curl http://localhost:8000/health
# {"status": "ok", "events_indexed": 777}
```

#### `POST /ask` — Poser une question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts sont prévus à Paris ?"}'
```

**Réponse :**
```json
{
  "question": "Quels concerts sont prévus à Paris ?",
  "answer": "Voici les concerts prévus à Paris : Concert COGE à l'Église Saint-Marcel les 12, 13, 19 et 20 juin 2026...",
  "sources": [
    {
      "id": 66422059,
      "titre": "Concert COGE",
      "lieu": "Église Saint-Marcel",
      "ville": "Paris",
      "date_debut": "2026-06-12",
      "url": "https://openagenda.com/..."
    }
  ]
}
```

#### `POST /rebuild` — Reconstruire l'index vectoriel
```bash
curl -X POST "http://localhost:8000/rebuild"
# {"message": "Index reconstruit avec 777 événements"}
```

---

## 8. Tests

### Lancer les tests unitaires

```bash
pytest tests/test_rag.py -v
```

### Avec rapport de couverture

```bash
pytest tests/test_rag.py -v --cov=scripts --cov=api --cov-report=term-missing
```

**Résultat obtenu :**

```
Name                      Stmts   Miss  Cover
---------------------------------------------
api/main.py                  50      0   100%
scripts/build_index.py       44      0   100%
scripts/fetch_events.py      64      0   100%
scripts/rag.py               51      0   100%
---------------------------------------------
TOTAL                       209      0   100%
35 passed
```

### Lancer l'évaluation qualité du RAG

```bash
python tests/evaluate_rag.py
```

---

## 9. Docker

### Prérequis (Mac)

```bash
colima start --dns 8.8.8.8
```

### Builder l'image

```bash
docker build -t puls-events .
```

### Lancer le container

```bash
docker run -p 8000:8000 --env-file .env puls-events
```

### Tester depuis le container

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements jazz à Paris ?"}'
```

---

## 10. Évaluation

Le script `tests/evaluate_rag.py` évalue automatiquement la qualité des réponses générées en les comparant aux réponses annotées manuellement (`tests/test_set.json`).

**Méthode :** Score de similarité cosinus entre les embeddings Mistral de la réponse générée et de la réponse attendue.

**Seuils :**

| Score | Évaluation |
|-------|-----------|
| ≥ 0.90 | ✅ Correcte |
| ≥ 0.75 | ⚠️ Partiellement correcte |
| < 0.75 | ❌ Incorrecte |

Les résultats sont sauvegardés dans `tests/evaluation_results.json`.

---

## 11. Résultats

| Métrique | Valeur |
|----------|--------|
| Événements indexés | **777** |
| Score moyen de similarité | **0.8995** |
| Réponses correctes (≥ 0.90) | **6 / 10** |
| Réponses partiellement correctes | **4 / 10** |
| Réponses incorrectes | **0 / 10** |
| Couverture des tests unitaires | **100%** |
| Nombre de tests unitaires | **35** |

**Analyse :** Le système obtient un score moyen de 0.8995, proche du seuil de correction (0.90). Aucune réponse n'est incorrecte, ce qui valide la pertinence du système RAG pour ce POC. Les 4 réponses partiellement correctes concernent des thématiques moins représentées dans les données (littérature, culture étrangère).

---

## 12. Pistes d'amélioration

1. **Enrichissement des données** — Augmenter le nombre d'agendas ciblés et couvrir d'autres villes françaises pour un corpus plus riche.
2. **Chunking avancé** — Découper les longues descriptions en chunks plus fins pour améliorer la précision de la recherche sémantique.
3. **Index FAISS optimisé** — Utiliser `IndexIVFFlat` ou `HNSW` pour de meilleures performances sur de grands volumes de données.
4. **Historique de conversation** — Ajouter la mémoire conversationnelle avec LangChain Memory pour un chatbot plus naturel.
5. **Filtres avancés** — Permettre de filtrer par date, lieu ou type d'événement directement dans la requête API.
6. **Évaluation RAGAS** — Intégrer la bibliothèque Ragas pour des métriques plus complètes (faithfulness, answer relevancy, context recall).
7. **CI/CD** — Automatiser les tests et l'évaluation via GitHub Actions à chaque push.
8. **Interface utilisateur** — Développer un frontend simple (Streamlit ou React) pour les équipes métier.
9. **Mise à jour automatique** — Programmer une tâche cron pour reconstruire l'index régulièrement avec les nouveaux événements.
10. **Sécurisation de l'API** — Ajouter une authentification JWT sur les endpoints sensibles comme `/rebuild`.
README

echo "✅ README.md généré"
