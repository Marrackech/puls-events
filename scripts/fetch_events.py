import requests
import pandas as pd
from dotenv import load_dotenv
import os
import json

load_dotenv()

API_KEY = os.getenv("OPENAGENDA_API_KEY")
LOCATION = os.getenv("TARGET_LOCATION", "Paris")

def fetch_agendas(location, size=10):
    url = "https://api.openagenda.com/v2/agendas"
    params = {"key": API_KEY, "size": size, "search": location}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json().get("agendas", [])

def fetch_events_from_agenda(agenda_uid, size=100):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    params = {"key": API_KEY, "size": size, "lang": "fr"}
    all_events = []
    after = None
    while True:
        if after:
            params["after"] = json.dumps(after)
        r = requests.get(url, params=params)
        if r.status_code != 200:
            break
        data = r.json()
        events = data.get("events", [])
        if not events:
            break
        all_events.extend(events)
        after = data.get("after")
        if not after or len(all_events) >= 200:
            break
    return all_events

def parse_events(events):
    rows = []
    for e in events:
        title   = e.get("title", {}).get("fr", "") or e.get("title", {}).get("en", "")
        desc    = e.get("description", {}).get("fr", "") or e.get("description", {}).get("en", "")
        loc     = e.get("location", {})
        timings = e.get("timings", [{}])
        rows.append({
            "id":          e.get("uid", ""),
            "titre":       title,
            "description": desc,
            "lieu":        loc.get("name", ""),
            "ville":       loc.get("city", ""),
            "adresse":     loc.get("address", ""),
            "date_debut":  timings[0].get("begin", "") if timings else "",
            "date_fin":    timings[-1].get("end", "") if timings else "",
            "url":         e.get("canonicalUrl", ""),
        })
    return pd.DataFrame(rows)

def run_fetch():
    print(f"🔍 Recherche d'agendas pour : {LOCATION}")
    agendas = fetch_agendas(LOCATION)
    print(f"  → {len(agendas)} agendas trouvés")
    all_events = []
    for agenda in agendas:
        uid = agenda["uid"]
        title = agenda["title"]
        print(f"  📅 Collecte : {title} (uid={uid})")
        events = fetch_events_from_agenda(uid)
        print(f"      → {len(events)} événements")
        all_events.extend(events)
    df = parse_events(all_events)
    df.dropna(subset=["titre", "description"], inplace=True)
    df = df[df["description"].str.len() > 20]
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("data/events.csv", index=False)
    print(f"\n✅ {len(df)} événements sauvegardés dans data/events.csv")
    return df

if __name__ == "__main__":
    run_fetch()  # pragma: no cover
