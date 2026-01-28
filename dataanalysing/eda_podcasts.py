import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Setup

DATA_CLEAN = "podcasts_clean_v2.csv"
FIG_DIR = "figures_eda"

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_CLEAN)
print(f"Geladener Datensatz: {len(df)} Zeilen")
print("Spalten:", list(df.columns))

# last_update_dt sicher in echtes Datetime-Format bringen, falls vorhanden
if "last_update_dt" in df.columns:
    df["last_update_dt"] = pd.to_datetime(df["last_update_dt"], errors="coerce")



# 1. Grundberechnungen

# Beschreibungslängen
df["description"] = df["description"].astype(str)
df["desc_char_len"] = df["description"].str.len()
df["desc_word_count"] = df["description"].str.split().str.len()

# Textlänge text_full
df["text_full"] = df["text_full"].astype(str)
df["text_full_len"] = df["text_full"].str.len()

# last_update_dt ggf. herstellen
if "last_update_dt" not in df.columns and "last_update_time" in df.columns:
    df["last_update_time"] = pd.to_numeric(df["last_update_time"], errors="coerce")
    df["last_update_dt"] = df["last_update_time"].apply(
        lambda x: datetime.utcfromtimestamp(x) if pd.notna(x) and x > 0 else pd.NaT
    )

# episode_count numeric
if "episode_count" in df.columns:
    df["episode_count"] = pd.to_numeric(df["episode_count"], errors="coerce")


# 2. Verteilung der Primary Categories

plt.figure(figsize=(10, 6))
df["primary_category"].value_counts().plot(kind="bar")
plt.title("Distribution of Primary Categories")
plt.xlabel("Primary Category")
plt.ylabel("Number of Podcasts")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "primary_category_distribution.png"), dpi=300)
plt.close()

# Statistik-Tabelle nach primary_category
cat_primary_stats = (
    df.groupby("primary_category")
      .agg(
          n_podcasts=("title", "count"),
          mean_desc_chars=("desc_char_len", "mean"),
          mean_desc_words=("desc_word_count", "mean"),
          mean_episode_count=("episode_count", "mean"),
      )
      .sort_values("n_podcasts", ascending=False)
)
cat_primary_stats.to_csv(os.path.join(FIG_DIR, "category_primary_stats.csv"))


# 3. Verteilung ALLER Kategorien (aus 'categories')

if "categories" in df.columns:
    df["categories_list"] = df["categories"].fillna("").str.split(",")
    df["categories_list"] = df["categories_list"].apply(
        lambda lst: [c.strip() for c in lst if c.strip()]
    )
    cat_exploded = df.explode("categories_list")
    category_all_counts = cat_exploded["categories_list"].value_counts().reset_index()
    category_all_counts.columns = ["category", "count"]
    category_all_counts.to_csv(os.path.join(FIG_DIR, "category_all_counts.csv"), index=False)

    plt.figure(figsize=(10, 6))
    category_all_counts.head(20).set_index("category")["count"].plot(kind="bar")
    plt.title("Top 20 Categories (All Tags)")
    plt.xlabel("Category")
    plt.ylabel("Number of Podcasts (tag-level)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "all_categories_top20.png"), dpi=300)
    plt.close()
else:
    print("'categories' nicht in df.columns – All-Category-Analyse wird übersprungen.")


# 4. Textlänge (text_full)

plt.figure(figsize=(10, 6))
sns.histplot(df["text_full_len"], bins=40, kde=True)
plt.title("Distribution of Text Length (Characters) in text_full")
plt.xlabel("Characters in text_full")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "text_full_length_distribution.png"), dpi=300)
plt.close()

# Textlänge pro Primary Category
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="primary_category", y="text_full_len")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Characters in text_full")
plt.title("Text Length by Primary Category")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "text_full_length_by_category.png"), dpi=300)
plt.close()


# 5. Episodenanzahl

if "episode_count" in df.columns:
    non_null_ep = df["episode_count"].dropna()
    if not non_null_ep.empty:
        # leichte Kappung, damit Verteilung sichtbar bleibt
        ep_cap = non_null_ep.quantile(0.99)
        ep_plot = non_null_ep.clip(upper=ep_cap)

        plt.figure(figsize=(10, 6))
        sns.histplot(ep_plot, bins=40, kde=True)
        plt.title("Distribution of Episode Count (capped at 99th percentile)")
        plt.xlabel("Episodes")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "episode_count_distribution.png"), dpi=300)
        plt.close()

        # Episodenanzahl nach Kategorie
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="primary_category", y="episode_count")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Episodes")
        plt.title("Episode Count by Primary Category")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "episode_count_by_category.png"), dpi=300)
        plt.close()
    else:
        print("episode_count ist leer – Episodenplots werden übersprungen.")
else:
    print("'episode_count' nicht in df.columns – Episodenplots werden übersprungen.")


# 6. Aktualität (last_update_dt)

if "last_update_dt" in df.columns:
    valid_dt = df["last_update_dt"].dropna()
    if not valid_dt.empty:
        plt.figure(figsize=(10, 6))
        # nach Datum gruppieren 
        dt_counts = valid_dt.dt.date.value_counts().sort_index()
        dt_counts.plot(kind="bar")
        plt.title("Number of Podcasts by Last Update Date")
        plt.xlabel("Date")
        plt.ylabel("Number of Podcasts")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "last_update_date_distribution.png"), dpi=300)
        plt.close()
    else:
        print("last_update_dt enthält keine gültigen Werte.")
else:
    print("'last_update_dt' nicht in df.columns – Aktualitätsplot wird übersprungen.")


# 7. Explicit-Flag

if "explicit" in df.columns:
    df["explicit"] = pd.to_numeric(df["explicit"], errors="coerce").fillna(0).astype(int)

    plt.figure(figsize=(6, 4))
    df["explicit"].value_counts().sort_index().plot(kind="bar")
    plt.title("Explicit Flag Distribution")
    plt.xticks([0, 1], ["Clean (0)", "Explicit (1)"], rotation=0)
    plt.ylabel("Number of Podcasts")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "explicit_flag_distribution.png"), dpi=300)
    plt.close()
else:
    print("'explicit' nicht in df.columns – Explicit-Plot wird übersprungen.")



print("EDA abgeschlossen.")
print(f"Alle Plots und Tabellen liegen im Ordner: {FIG_DIR}")
