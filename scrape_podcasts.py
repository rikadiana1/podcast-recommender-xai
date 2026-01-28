import os
import time
import hashlib
import requests
import pandas as pd
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import ast

# KONFIGURATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("PODCASTINDEX_API_KEY")
API_SECRET = os.getenv("PODCASTINDEX_API_SECRET")

TARGET_FEEDS = 1000  # Zielgröße – abhängig davon, was das Endpoint liefert
OUT_CSV = "podcasts_raw_global.csv"


def build_headers(api_key: str, api_secret: str) -> dict:
    ts = int(time.time())
    auth = hashlib.sha1((api_key + api_secret + str(ts)).encode("utf-8")).hexdigest()
    return {
        "User-Agent": "MasterThesisPodcast/1.0",
        "X-Auth-Date": str(ts),
        "X-Auth-Key": api_key,
        "Authorization": auth,
    }


def call_api(url: str, headers: dict) -> dict:
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_trending_once(headers: dict, max_n: int = TARGET_FEEDS) -> list:
   
    url = f"https://api.podcastindex.org/api/1.0/podcasts/trending?max={max_n}"
    data = call_api(url, headers)
    feeds = data.get("feeds", []) or []
    print(f"Trending-Feeds erhalten: {len(feeds)}")
    return feeds


def normalize_feed(feed: dict) -> dict:
    cats = feed.get("categories") or {}
    if isinstance(cats, dict):
        cats_list = sorted({str(v) for v in cats.values()})
    elif isinstance(cats, list):
        cats_list = sorted({str(x) for x in cats})
    else:
        cats_list = []
    categories_str = ", ".join(cats_list)

    lang_raw = (feed.get("language") or "").strip()

    return {
        "title": feed.get("title"),
        "description": feed.get("description") or "",
        "language": lang_raw,
        "categories": categories_str,
        "author": feed.get("author"),
        "rss_url": feed.get("url"),
        "itunes_id": feed.get("itunesId"),
        "episode_count": feed.get("episodeCount"),
        "explicit": feed.get("explicit"),
        "dead": feed.get("dead"),
        "locked": feed.get("locked"),
        "last_update_time": feed.get("lastUpdateTime"),
    }


def normalize_lang_raw(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.lower().strip()
    if raw.startswith("de"):
        return "de"
    if raw.startswith("en") or raw in {"eng", "en-us", "en-gb"}:
        return "en"
    return raw


def detect_lang_from_text(title: str, desc: str) -> str:
    text = ((title or "") + " " + (desc or "")).strip()
    if not text or len(text.split()) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def choose_final_lang(row):
    if row.get("language_norm"):
        return row["language_norm"]
    detected = row.get("language_detected")
    if isinstance(detected, str) and detected not in ("unknown", ""):
        return detected
    return "unknown"


def parse_categories(raw):
    """
    Normalisiert das 'categories'-Feld aus der API.
    Rückgabe: Liste von Kategorien (Strings).
    """
    if not isinstance(raw, str) or raw.strip() == "":
        return []

    raw = raw.strip()

    # Versuch
    if raw.startswith("{") and raw.endswith("}"):
        try:
            d = ast.literal_eval(raw)
            if isinstance(d, dict):
                return [str(v).strip() for v in d.values() if str(v).strip()]
        except Exception:
            pass  # fällt zurück auf Kommasplit

    # Fallback: Kommagetrennt
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def main():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API-Key/Secret fehlen. In .env setzen oder als Umgebungsvariablen.")

    headers = build_headers(API_KEY, API_SECRET)

    print("Hole Trending-Podcasts (einmaliger Call, ohne Paging)...")
    feeds = fetch_trending_once(headers, max_n=TARGET_FEEDS)

    if not feeds:
        print("Keine Daten erhalten – bitte API-Key/Secret oder Endpoint prüfen.")
        return

    rows = [normalize_feed(f) for f in feeds]
    df = pd.DataFrame(rows)

    # Dedupe
    before = len(df)
    df = df.drop_duplicates(subset=["rss_url"]).reset_index(drop=True)
    print(f"Deduplikation via rss_url: {before} -> {len(df)} Zeilen")

    # Sprache
    df["language_raw"] = df["language"].fillna("").astype(str)
    df["language_norm"] = df["language_raw"].apply(normalize_lang_raw)

    mask_need_detect = (df["language_norm"] == "") | (df["language_norm"] == "und") | (df["language_norm"] == "unknown")
    df.loc[mask_need_detect, "language_detected"] = df[mask_need_detect].apply(
        lambda row: detect_lang_from_text(row.get("title", ""), row.get("description", "")),
        axis=1,
    )
    df["language_detected"] = df["language_detected"].fillna("")
    df["language_final"] = df.apply(choose_final_lang, axis=1)

    # Kategorienlisten & primäre Kategorie erzeugen
    df["categories_list"] = df["categories"].apply(parse_categories)
    df["primary_category"] = df["categories_list"].apply(lambda lst: lst[0] if lst else None)

    cols = [
        "title", "author",
        "language", "language_raw", "language_norm", "language_detected", "language_final",
        "categories", "categories_list", "primary_category",
        "description", "rss_url", "itunes_id",
        "episode_count", "explicit", "dead", "locked", "last_update_time",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ gespeichert: {OUT_CSV} — {len(df)} Zeilen")

    # Auswertungen
    print("\nSprachverteilung (language_final):")
    print(df["language_final"].value_counts().head(15))

    print("\nVerteilung der primären Kategorien (primary_category):")
    print(df["primary_category"].value_counts().head(30))


if __name__ == "__main__":
    main()
