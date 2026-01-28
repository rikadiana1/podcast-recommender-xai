import pandas as pd
import re
from datetime import datetime

RAW_CSV = "podcasts_raw_enriched.csv"   
CLEAN_CSV = "podcasts_clean.csv"


def clean_text(text: str) -> str:
    """Einfache Textbereinigung: URLs, HTML, Sonderzeichen entfernen, kleinschreiben."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # URLs entfernen
    text = re.sub(r"http\S+", " ", text)
    # HTML-Tags entfernen
    text = re.sub(r"<.*?>", " ", text)
    # Sonstige Sonderzeichen (Emojis etc.)
    text = re.sub(r"[^a-z0-9äöüß ]", " ", text)
    # Mehrfache Leerzeichen
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    
    # 1) Einlesen
   
    df = pd.read_csv(RAW_CSV)
    print(f"Ausgangsdatensatz: {len(df)} Zeilen")
    print("Spalten:", list(df.columns))

    
    # 2) Pflichtfelder prüfen
   
    required_cols = ["title", "description", "rss_url", "language_final", "primary_category"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Pflichtspalte fehlt: {col}")

    before = len(df)
    df = df.dropna(subset=required_cols)
    print(f"Nach Entfernen fehlender Pflichtfelder: {before} -> {len(df)} Zeilen")

  
    # 3) Nur englischsprachige Feeds
  
    before = len(df)
    df = df[df["language_final"] == "en"].reset_index(drop=True)
    print(f"Filter language_final == 'en': {before} -> {len(df)} Zeilen")

    # 4) Tote Feeds entfernen (dead == 1)
   
    if "dead" in df.columns:
        df["dead"] = pd.to_numeric(df["dead"], errors="coerce").fillna(0).astype(int)
        before = len(df)
        df = df[df["dead"] == 0].reset_index(drop=True)
        print(f"Filter dead == 0 (aktive Feeds): {before} -> {len(df)} Zeilen")
    else:
        print("Spalte 'dead' fehlt – keine Filterung nach aktiven Feeds.")


    # 5) Beschreibungslänge filtern

    df["description"] = df["description"].astype(str)
    df["desc_word_count"] = df["description"].str.split().str.len()
    df["desc_char_len"] = df["description"].str.len()

    MIN_WORDS = 20
    MIN_CHARS = 100

    before = len(df)
    df = df[(df["desc_word_count"] >= MIN_WORDS) & (df["desc_char_len"] >= MIN_CHARS)].reset_index(drop=True)
    print(
        f"Filter Beschreibung (>= {MIN_WORDS} Wörter, >= {MIN_CHARS} Zeichen): "
        f"{before} -> {len(df)} Zeilen"
    )


  
    # 6) Textfeld für NLP erzeugen

    df["title"] = df["title"].astype(str)
    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["text_full"] = (df["title_clean"] + " " + df["description_clean"]).str.strip()

    before = len(df)
    df = df[df["text_full"].str.len() >= 80].reset_index(drop=True)
    print(f"Filter text_full >= 80 Zeichen: {before} -> {len(df)} Zeilen")

   
    # 7) Episodenanzahl & last_update_time säubern
 
    if "episode_count" in df.columns:
        df["episode_count"] = pd.to_numeric(df["episode_count"], errors="coerce").fillna(0)
        df.loc[df["episode_count"] < 0, "episode_count"] = 0
        df["episode_count_capped"] = df["episode_count"].clip(upper=1000)
    else:
        print("Spalte 'episode_count' fehlt – keine numerische Bereinigung dafür.")

    if "last_update_time" in df.columns:
        df["last_update_time"] = pd.to_numeric(df["last_update_time"], errors="coerce")
        # zusätzliche Datumsdarstellung (optional)
        df["last_update_dt"] = df["last_update_time"].apply(
            lambda x: datetime.utcfromtimestamp(x) if pd.notna(x) and x > 0 else pd.NaT
        )
    else:
        print("⚠️ Spalte 'last_update_time' fehlt – keine zeitliche Auswertung möglich.")

    # Explicit-Flag numerisch machen, falls vorhanden
    if "explicit" in df.columns:
        df["explicit"] = pd.to_numeric(df["explicit"], errors="coerce").fillna(0).astype(int)

  
    # 8) Auswertung direkt im Cleaning
 
    print("\n=== Grundlegende Auswertung des bereinigten Datensatzes ===")
    print(f"Finale Zeilenanzahl: {len(df)}")

    # Kategorien
    print("\nAnzahl unterschiedlicher Kategorien (primary_category):", df["primary_category"].nunique())
    print("Top-10 Kategorien:")
    print(df["primary_category"].value_counts().head(10))
    
    # categories als Liste aufspalten
    df["categories_list"] = df["categories"].fillna("").str.split(",")

    # trimmen der Strings
    df["categories_list"] = df["categories_list"].apply(lambda lst: [c.strip() for c in lst if c.strip()])

    # Kategorien explodieren
    cat_exploded = df.explode("categories_list")

    # Häufigkeiten berechnen
    category_counts = cat_exploded["categories_list"].value_counts()

    print("\nTop-20 Kategorien über *alle* Category-Tags:")
    print(category_counts.head(20))


    # Sprache
    print("\nSprachverteilung (language_final):")
    print(df["language_final"].value_counts())

    # Episodenanzahl
    if "episode_count" in df.columns:
        print("\nEpisodenanzahl (episode_count) – Beschreibung:")
        print(df["episode_count"].describe())
        print(f"Durchschnittliche Episodenanzahl: {df['episode_count'].mean():.2f}")

    # Letztes Update (ungefährer Zeitraum)
    if "last_update_time" in df.columns:
        valid_times = df["last_update_dt"].dropna()
        if not valid_times.empty:
            print("\nZeitliche Verteilung (last_update_dt):")
            print("Frühestes Update:", valid_times.min())
            print("Spätestes Update:", valid_times.max())
            print("Medianes Update:", valid_times.median())
        else:
            print("Keine gültigen last_update_dt-Werte für zeitliche Auswertung.")
    # Beschreibungslängen
    print("\nBeschreibungslängen (desc_char_len) – Beschreibung:")
    print(df["desc_char_len"].describe())
    print(f"Durchschnittliche Beschreibungslänge (Zeichen): {df['desc_char_len'].mean():.1f}")
    print(f"Durchschnittliche Beschreibungslänge (Wörter): {df['desc_word_count'].mean():.1f}")

    # Explicit-Flag (falls vorhanden)
    if "explicit" in df.columns:
        print("\nExplizit-Flag-Verteilung (explicit):")
        print(df["explicit"].value_counts())

    
    # 9) Spaltenauswahl & Speichern
  
    keep_cols = [
        "title",
        "author",
        "primary_category",
        "categories",
        "categories_list" if "categories_list" in df.columns else None,
        "language_final",
        "description",
        "title_clean",
        "description_clean",
        "text_full",
        "rss_url",
        "itunes_id",
        "episode_count",
        "episode_count_capped" if "episode_count_capped" in df.columns else None,
        "last_update_time" if "last_update_time" in df.columns else None,
        "last_update_dt" if "last_update_dt" in df.columns else None,
        "explicit" if "explicit" in df.columns else None,
        "dead",
        "locked" if "locked" in df.columns else None,
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in df.columns]
    df = df[keep_cols]

    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"Bereinigter Datensatz gespeichert: {CLEAN_CSV}")
    print(f"   Spalten: {list(df.columns)}")


if __name__ == "__main__":
    main()
