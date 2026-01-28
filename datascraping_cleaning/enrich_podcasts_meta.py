import os
import time
import hashlib
import requests
import pandas as pd
import feedparser
from urllib.parse import quote_plus
from dotenv import load_dotenv
from tqdm import tqdm


# Konfiguration


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("PODCASTINDEX_API_KEY")
API_SECRET = os.getenv("PODCASTINDEX_API_SECRET")

RAW_CSV = "podcasts_raw_global.csv"        # Eingabe: Trending-Ergebnis
OUT_CSV = "podcasts_raw_enriched.csv"      # Ausgabe: mit episode_count + last_update_time
REQUEST_SLEEP = 0.8                       # Rate-Limiting



# Hilfsfunktionen


def build_headers(api_key: str, api_secret: str) -> dict:
    ts = int(time.time())
    auth = hashlib.sha1((api_key + api_secret + str(ts)).encode("utf-8")).hexdigest()
    return {
        "User-Agent": "MasterThesisPodcast/1.2",
        "X-Auth-Date": str(ts),
        "X-Auth-Key": api_key,
        "Authorization": auth,
    }


def call_api(url: str, headers: dict) -> dict:
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_feed_details_by_url(headers: dict, rss_url: str):
    """
    Holt Feed-Details über /podcasts/byfeedurl.
    Rückgabe: Dict (feed) oder None.
    """
    if not rss_url or not isinstance(rss_url, str):
        return None

    url_encoded = quote_plus(rss_url)
    url = f"https://api.podcastindex.org/api/1.0/podcasts/byfeedurl?url={url_encoded}"

    try:
        data = call_api(url, headers)
    except requests.RequestException:
        return None

    feed = data.get("feed")
    return feed if isinstance(feed, dict) else None


def count_episodes_from_rss(rss_url: str, timeout: int = 8):
    """
    RSS-Feed herunterladen und parsen.
    Hinweis: RSS enthält oft nur die letzten N Episoden -> Ergebnis ist eine Untergrenze.
    """
    if not rss_url or not isinstance(rss_url, str):
        return None

    try:
        resp = requests.get(rss_url, timeout=timeout, headers={"User-Agent": "PodcastCrawler/1.0"})
        resp.raise_for_status()
    except Exception:
        return None

    try:
        parsed = feedparser.parse(resp.content)
        return len(parsed.entries) if hasattr(parsed, "entries") else None
    except Exception:
        return None


def to01(x) -> int:
    """Normalisiert bool/str/int in 0/1."""
    return 1 if str(x).strip().lower() in ("1", "true", "yes") else 0



# Hauptlogik


def main():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API-Key/Secret fehlen. In .env setzen oder als Umgebungsvariablen.")

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    print(f"Ausgangsdatensatz (roh): {len(df)} Zeilen")

    if "rss_url" not in df.columns:
        raise KeyError("Spalte 'rss_url' fehlt – ohne RSS-URL können keine Details geladen werden.")

    headers = build_headers(API_KEY, API_SECRET)

    # Zielspalten anlegen (falls nicht vorhanden)
    for col in ["episode_count", "last_update_time", "explicit", "dead", "episode_source"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Nur Zeilen enrich-en, wo episode_count oder last_update_time fehlt
    mask_need_fetch = df["episode_count"].isna() | df["last_update_time"].isna()
    idx_to_fetch = df.index[mask_need_fetch]

    print(f"Feeds mit fehlenden Metadaten: {len(idx_to_fetch)} – Enrichment via API + RSS-Fallback…")

    cache = {}  # rss_url -> (ep, lut, expl01, dead01, source)

    for idx in tqdm(idx_to_fetch, desc="Enriching feeds"):
        rss_url = df.at[idx, "rss_url"]

        if not isinstance(rss_url, str) or not rss_url.strip():
            continue

        if rss_url in cache:
            ep, lut, expl01, dead01, source = cache[rss_url]
        else:
            feed = fetch_feed_details_by_url(headers, rss_url)

            ep = None
            lut = None
            expl01 = None
            dead01 = None
            source = "none"

            # 1) Primär: API
            if feed is not None:
                ep = feed.get("episodeCount")
                lut = feed.get("lastUpdateTime")
                expl01 = to01(feed.get("explicit"))
                dead01 = to01(feed.get("dead"))

            # 2) Fallback: RSS nur für episode_count, wenn API ep fehlt
            if ep in (None, "", "null"):
                rss_ep = count_episodes_from_rss(rss_url)
                if rss_ep is not None:
                    ep = rss_ep
                    source = "rss"
                else:
                    source = "none"
            else:
                source = "api"

            # Falls API gar nicht da war: expl/dead bleiben evtl. None -> als 0 setzen
            if expl01 is None:
                expl01 = 0
            if dead01 is None:
                dead01 = 0

            cache[rss_url] = (ep, lut, expl01, dead01, source)

        # Werte in DataFrame übernehmen
        df.at[idx, "episode_count"] = ep
        if lut is not None:
            df.at[idx, "last_update_time"] = lut

        df.at[idx, "explicit"] = expl01
        df.at[idx, "dead"] = dead01
        df.at[idx, "episode_source"] = source

        time.sleep(REQUEST_SLEEP)

    # Typen bereinigen
    df["episode_count"] = pd.to_numeric(df["episode_count"], errors="coerce")
    df["last_update_time"] = pd.to_numeric(df["last_update_time"], errors="coerce")

    # explicit/dead sicher als 0/1 int
    df["explicit"] = df["explicit"].apply(to01).astype(int)
    df["dead"] = df["dead"].apply(to01).astype(int)

    # Zusammenfassung
    print("\n=== Zusammenfassung EpisodeCount-Quellen ===")
    print(df["episode_source"].value_counts(dropna=False))

    print("\nBeispielzeilen (Titel, EpisodeCount, Source, LastUpdate):")
    print(df[["title", "episode_count", "episode_source", "last_update_time"]].head(10))

    # Speichern
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Enriched-Datensatz gespeichert: {OUT_CSV}")


if __name__ == "__main__":
    main()
