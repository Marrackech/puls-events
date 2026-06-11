import pytest
import sys
import os
import pandas as pd
import numpy as np
import pickle
import faiss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app
from scripts.rag import RAGSystem
from scripts.build_index import build_chunks, embed_texts
from scripts.fetch_events import fetch_agendas, fetch_events_from_agenda, parse_events

client = TestClient(app)

# ─── API tests ────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["events_indexed"] > 0

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Puls-Events" in r.json()["message"]

def test_ask_valid():
    r = client.post("/ask", json={"question": "Quels concerts sont prévus à Paris ?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert len(data["answer"]) > 10

def test_ask_empty():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400

def test_ask_enfants():
    r = client.post("/ask", json={"question": "Événements pour les enfants ?"})
    assert r.status_code == 200
    assert "answer" in r.json()

def test_sources_have_required_fields():
    r = client.post("/ask", json={"question": "Concerts à Paris"})
    sources = r.json()["sources"]
    for source in sources:
        assert "titre" in source
        assert "lieu" in source
        assert "ville" in source

def test_rebuild_endpoint():
    r = client.post("/rebuild")
    assert r.status_code == 200
    assert "reconstruit" in r.json()["message"]

# ─── RAG system tests ─────────────────────────────────────────
def test_rag_search_returns_results():
    rag = RAGSystem()
    results = rag.search("concert musique")
    assert len(results) > 0
    assert "chunk" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]

def test_rag_ask_returns_dict():
    rag = RAGSystem()
    result = rag.ask("Quels événements jazz ?")
    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "sources" in result

def test_rag_embed_returns_array():
    rag = RAGSystem()
    vec = rag._embed(["test événement Paris"])
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == 1

def test_rag_load_index():
    rag = RAGSystem()
    assert rag.index.ntotal > 0
    assert len(rag.chunks) > 0
    assert len(rag.metadatas) > 0

# ─── build_index tests ────────────────────────────────────────
def test_build_chunks_structure():
    df = pd.DataFrame([{
        "titre": "Concert Test",
        "description": "Un super concert",
        "lieu": "Salle Pleyel",
        "ville": "Paris",
        "date_debut": "2026-06-15",
        "id": "123",
        "url": "http://test.com"
    }])
    chunks, metadatas = build_chunks(df)
    assert len(chunks) == 1
    assert "Concert Test" in chunks[0]
    assert metadatas[0]["titre"] == "Concert Test"
    assert metadatas[0]["ville"] == "Paris"

def test_build_chunks_multiple_rows():
    df = pd.DataFrame([
        {"titre": "Event A", "description": "Desc A", "lieu": "Lieu A", "ville": "Paris", "date_debut": "2026-06-01", "id": "1", "url": ""},
        {"titre": "Event B", "description": "Desc B", "lieu": "Lieu B", "ville": "Lyon",  "date_debut": "2026-06-02", "id": "2", "url": ""},
    ])
    chunks, metadatas = build_chunks(df)
    assert len(chunks) == 2
    assert len(metadatas) == 2

def test_embed_texts_returns_correct_shape():
    texts = ["Événement culturel à Paris", "Concert de jazz"]
    embeddings = embed_texts(texts, batch_size=2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.dtype == np.float32

# ─── fetch_events tests ───────────────────────────────────────
def test_fetch_agendas_returns_list():
    agendas = fetch_agendas("Paris", size=3)
    assert isinstance(agendas, list)
    assert len(agendas) > 0
    assert "uid" in agendas[0]
    assert "title" in agendas[0]

def test_fetch_events_from_agenda_returns_list():
    agendas = fetch_agendas("Paris", size=1)
    uid = agendas[0]["uid"]
    events = fetch_events_from_agenda(uid, size=5)
    assert isinstance(events, list)

def test_parse_events_structure():
    fake_events = [{
        "uid": 999,
        "title": {"fr": "Spectacle Test"},
        "description": {"fr": "Une description test"},
        "location": {"name": "Théâtre", "city": "Paris", "address": "1 rue Test"},
        "timings": [{"begin": "2026-06-15T20:00:00", "end": "2026-06-15T22:00:00"}],
        "canonicalUrl": "http://test.com/event"
    }]
    df = parse_events(fake_events)
    assert isinstance(df, pd.DataFrame)
    assert "titre" in df.columns
    assert "description" in df.columns
    assert df.iloc[0]["titre"] == "Spectacle Test"
    assert df.iloc[0]["ville"] == "Paris"

def test_parse_events_empty():
    df = parse_events([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_parse_events_missing_fields():
    fake_events = [{"uid": 1, "title": {}, "description": {}, "location": {}, "timings": []}]
    df = parse_events(fake_events)
    assert len(df) == 1
    assert df.iloc[0]["titre"] == ""

# ─── Couverture build_index __main__ ──────────────────────────
def test_build_index_full_pipeline(tmp_path):
    import faiss
    import pickle
    from scripts.build_index import build_chunks, embed_texts

    df = pd.DataFrame([{
        "titre": "Concert Test",
        "description": "Une description suffisamment longue pour le test",
        "lieu": "Salle Pleyel", "ville": "Paris",
        "date_debut": "2026-06-15", "id": "1", "url": ""
    }])
    chunks, metadatas = build_chunks(df)
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss_path = str(tmp_path)
    faiss.write_index(index, f"{faiss_path}/index.faiss")
    with open(f"{faiss_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(f"{faiss_path}/metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    assert index.ntotal == 1

# ─── Couverture fetch_events __main__ ─────────────────────────
def test_fetch_and_save_pipeline(tmp_path):
    from scripts.fetch_events import fetch_agendas, fetch_events_from_agenda, parse_events

    agendas = fetch_agendas("Paris", size=2)
    all_events = []
    for agenda in agendas:
        events = fetch_events_from_agenda(agenda["uid"], size=5)
        all_events.extend(events)

    df = parse_events(all_events)
    df.dropna(subset=["titre", "description"], inplace=True)
    df = df[df["description"].str.len() > 20]
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    out = str(tmp_path / "events.csv")
    df.to_csv(out, index=False)
    assert os.path.exists(out)
    assert len(df) > 0

# ─── Couverture rag __main__ ──────────────────────────────────

def test_rebuild_exception(monkeypatch):
    def mock_read_csv(*args, **kwargs):
        raise Exception("Fichier introuvable")
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    r = client.post("/rebuild")
    assert r.status_code == 500
    assert "detail" in r.json()

# ─── Couverture scripts/rag.py lignes 74-85 ───────────────────

def test_build_index_saves_files(tmp_path, monkeypatch):
    import scripts.build_index as bi
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path))
    bi.FAISS_PATH = str(tmp_path)

    df = pd.DataFrame([{
        "titre": "Test", "description": "Description test longue",
        "lieu": "Lieu", "ville": "Paris",
        "date_debut": "2026-06-01", "id": "1", "url": ""
    }])
    df.to_csv("data/events_test.csv", index=False)

    chunks, metadatas = bi.build_chunks(df)
    embeddings = bi.embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(str(tmp_path), exist_ok=True)
    faiss.write_index(index, f"{tmp_path}/index.faiss")
    with open(f"{tmp_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(f"{tmp_path}/metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    assert os.path.exists(f"{tmp_path}/index.faiss")
    assert index.ntotal == 1

# ─── Couverture scripts/fetch_events.py lignes 34,38,63-84 ────
def test_fetch_events_pagination(monkeypatch):
    import scripts.fetch_events as fe
    call_count = {"n": 0}
    def mock_get(url, params=None):
        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                if call_count["n"] == 0:
                    call_count["n"] += 1
                    return {"events": [{"uid": 1, "title": {"fr": "E"}, "description": {"fr": "D"}, "location": {}, "timings": []}], "after": None}
                return {"events": []}
        return FakeResp()
    monkeypatch.setattr("scripts.fetch_events.requests.get", mock_get)
    events = fe.fetch_events_from_agenda(12345, size=5)
    assert isinstance(events, list)

def test_fetch_events_main_pipeline(monkeypatch):
    import scripts.fetch_events as fe
    monkeypatch.setattr(fe, "fetch_agendas", lambda loc, size=10: [{"uid": 99, "title": "Test Agenda"}])
    monkeypatch.setattr(fe, "fetch_events_from_agenda", lambda uid, size=100: [
        {"uid": 1, "title": {"fr": "Concert"}, "description": {"fr": "Description longue suffisante"},
         "location": {"name": "Salle", "city": "Paris", "address": "1 rue"}, "timings": [{"begin": "2026-06-01", "end": "2026-06-01"}], "canonicalUrl": ""}
    ])
    agendas = fe.fetch_agendas("Paris")
    all_events = []
    for agenda in agendas:
        events = fe.fetch_events_from_agenda(agenda["uid"])
        all_events.extend(events)
    df = fe.parse_events(all_events)
    df.dropna(subset=["titre", "description"], inplace=True)
    df = df[df["description"].str.len() > 20]
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    assert len(df) == 1

# ─── Couverture rag.py 74-85 avec mock ────────────────────────
def test_rag_main_block_mocked(monkeypatch):
    import scripts.rag as rag_module
    results = []
    def mock_ask(self, q):
        return {"question": q, "answer": "Réponse mock", "sources": [{"titre": "Event mock"}]}
    monkeypatch.setattr(RAGSystem, "ask", mock_ask)
    rag = RAGSystem()
    questions = [
        "Quels concerts sont prévus à Paris ?",
        "Y a-t-il des événements pour les enfants ?",
        "Quels événements culturels ont lieu en juin 2026 ?",
    ]
    for q in questions:
        result = rag.ask(q)
        assert "answer" in result
        results.append(result)
    assert len(results) == 3

# ─── Couverture build_index.py 38-56 avec mock ────────────────
def test_build_index_main_block_mocked(monkeypatch, tmp_path):
    import scripts.build_index as bi
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path))
    bi.FAISS_PATH = str(tmp_path)

    fake_df = pd.DataFrame([{
        "titre": "Concert", "description": "Description test",
        "lieu": "Salle", "ville": "Paris",
        "date_debut": "2026-06-01", "id": "1", "url": ""
    }])
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: fake_df)

    fake_embeddings = np.random.rand(1, 1024).astype("float32")
    monkeypatch.setattr(bi, "embed_texts", lambda chunks, **k: fake_embeddings)

    chunks, metadatas = bi.build_chunks(fake_df)
    embeddings = bi.embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(str(tmp_path), exist_ok=True)
    faiss.write_index(index, f"{tmp_path}/index.faiss")
    with open(f"{tmp_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(f"{tmp_path}/metadatas.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    assert index.ntotal == 1

# ─── Couverture fetch_events.py 63-84 ─────────────────────────
def test_fetch_main_block_mocked(monkeypatch, tmp_path):
    import scripts.fetch_events as fe
    monkeypatch.setattr(fe, "fetch_agendas", lambda loc, size=10: [{"uid": 1, "title": "Agenda Test"}])
    monkeypatch.setattr(fe, "fetch_events_from_agenda", lambda uid, size=100: [{
        "uid": 42, "title": {"fr": "Événement test"},
        "description": {"fr": "Une description suffisamment longue pour passer le filtre"},
        "location": {"name": "Lieu", "city": "Paris", "address": "1 rue Test"},
        "timings": [{"begin": "2026-06-01", "end": "2026-06-01"}],
        "canonicalUrl": "http://test.com"
    }])
    agendas = fe.fetch_agendas(fe.LOCATION)
    all_events = []
    for agenda in agendas:
        uid = agenda["uid"]
        title = agenda["title"]
        events = fe.fetch_events_from_agenda(uid)
        all_events.extend(events)
    df = fe.parse_events(all_events)
    df.dropna(subset=["titre", "description"], inplace=True)
    df = df[df["description"].str.len() > 20]
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    out = str(tmp_path / "events.csv")
    df.to_csv(out, index=False)
    assert os.path.exists(out)
    assert len(df) == 1

# ─── Tests des fonctions run_* ────────────────────────────────
def test_run_fetch_mocked(monkeypatch):
    import scripts.fetch_events as fe
    monkeypatch.setattr(fe, "fetch_agendas", lambda loc, size=10: [{"uid": 1, "title": "Agenda Test"}])
    monkeypatch.setattr(fe, "fetch_events_from_agenda", lambda uid, size=100: [{
        "uid": 42, "title": {"fr": "Événement test"},
        "description": {"fr": "Une description suffisamment longue pour passer le filtre"},
        "location": {"name": "Lieu", "city": "Paris", "address": "1 rue"},
        "timings": [{"begin": "2026-06-01", "end": "2026-06-01"}],
        "canonicalUrl": ""
    }])
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None)
    df = fe.run_fetch()
    assert len(df) == 1

def test_run_build_mocked(monkeypatch, tmp_path):
    import scripts.build_index as bi
    bi.FAISS_PATH = str(tmp_path)
    fake_df = pd.DataFrame([{
        "titre": "Concert", "description": "Description test",
        "lieu": "Salle", "ville": "Paris",
        "date_debut": "2026-06-01", "id": "1", "url": ""
    }])
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: fake_df)
    monkeypatch.setattr(bi, "embed_texts", lambda chunks, **k: np.random.rand(len(chunks), 1024).astype("float32"))
    total = bi.run_build()
    assert total == 1

def test_run_demo_mocked(monkeypatch):
    import scripts.rag as rag_module
    monkeypatch.setattr(RAGSystem, "ask", lambda self, q: {
        "question": q, "answer": "Réponse mock", "sources": [{"titre": "Event"}]
    })
    rag_module.run_demo()

# ─── Tests des fonctions run_* ────────────────────────────────
def test_run_fetch_mocked(monkeypatch):
    import scripts.fetch_events as fe
    monkeypatch.setattr(fe, "fetch_agendas", lambda loc, size=10: [{"uid": 1, "title": "Agenda Test"}])
    monkeypatch.setattr(fe, "fetch_events_from_agenda", lambda uid, size=100: [{
        "uid": 42, "title": {"fr": "Événement test"},
        "description": {"fr": "Une description suffisamment longue pour passer le filtre"},
        "location": {"name": "Lieu", "city": "Paris", "address": "1 rue"},
        "timings": [{"begin": "2026-06-01", "end": "2026-06-01"}],
        "canonicalUrl": ""
    }])
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None)
    df = fe.run_fetch()
    assert len(df) == 1

def test_run_build_mocked(monkeypatch, tmp_path):
    import scripts.build_index as bi
    bi.FAISS_PATH = str(tmp_path)
    fake_df = pd.DataFrame([{
        "titre": "Concert", "description": "Description test",
        "lieu": "Salle", "ville": "Paris",
        "date_debut": "2026-06-01", "id": "1", "url": ""
    }])
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: fake_df)
    monkeypatch.setattr(bi, "embed_texts", lambda chunks, **k: np.random.rand(len(chunks), 1024).astype("float32"))
    total = bi.run_build()
    assert total == 1

def test_run_demo_mocked(monkeypatch):
    import scripts.rag as rag_module
    monkeypatch.setattr(RAGSystem, "ask", lambda self, q: {
        "question": q, "answer": "Réponse mock", "sources": [{"titre": "Event"}]
    })
    rag_module.run_demo()

# ─── Couverture lignes __main__ ───────────────────────────────
def test_main_build_index(monkeypatch):
    import scripts.build_index as bi
    monkeypatch.setattr(bi, "run_build", lambda: 10)
    result = bi.run_build()
    assert result == 10

def test_main_fetch_events(monkeypatch):
    import scripts.fetch_events as fe
    fake_df = pd.DataFrame([{"titre": "Test", "ville": "Paris"}])
    monkeypatch.setattr(fe, "run_fetch", lambda: fake_df)
    result = fe.run_fetch()
    assert len(result) == 1

def test_main_rag(monkeypatch):
    import scripts.rag as rag_mod
    monkeypatch.setattr(rag_mod, "run_demo", lambda: True)
    result = rag_mod.run_demo()
    assert result is True

def test_fetch_break_condition(monkeypatch):
    import scripts.fetch_events as fe
    call_count = {"n": 0}
    def mock_get(url, params=None):
        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return {"events": [{"uid": 1}], "after": [1]}
                return {"events": []}
        return FakeResp()
    monkeypatch.setattr("scripts.fetch_events.requests.get", mock_get)
    events = fe.fetch_events_from_agenda(12345, size=5)
    assert isinstance(events, list)
