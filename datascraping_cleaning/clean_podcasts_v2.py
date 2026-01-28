import os
import numpy as np
import pandas as pd


# Config

DATA_IN = "podcasts_clean.csv"
SIM_PATH = "nlp_outputs/similarity_tfidf.npy"   

OUT_CLEAN = "podcasts_clean_v2.csv"
OUT_REMOVED_XLSX = "podcasts_removed_template_series.xlsx"

SIM_THRESHOLD = 0.90     # ab wann semantisch "fast identisch"
MIN_NEIGHBORS = 10       # wie viele ähnliche nötig sind → Serie


# Load data

df = pd.read_csv(DATA_IN)
sim = np.load(SIM_PATH)



# Serien erkennen

np.fill_diagonal(sim, 0.0)  # sich selbst nicht zählen

neighbors = (sim >= SIM_THRESHOLD).sum(axis=1)

df["template_neighbors"] = neighbors
df["is_template_series"] = df["template_neighbors"] >= MIN_NEIGHBORS

print("🔍 Als Template/Series erkannt:", df["is_template_series"].sum())


# Split: behalten vs. entfernen

df_removed = df[df["is_template_series"]].copy()
df_clean2 = df[~df["is_template_series"]].copy()


# Speichern

df_clean2.to_csv(OUT_CLEAN, index=False, encoding="utf-8")
print("✅ Bereinigter Datensatz gespeichert:", OUT_CLEAN)

# Excel mit entfernten Podcasts
df_removed.sort_values(
    by="template_neighbors",
    ascending=False
).to_excel(OUT_REMOVED_XLSX, index=False)

print("Entfernte Serien gespeichert:", OUT_REMOVED_XLSX)


# Kurze Diagnose

print("\nBeispiele entfernte Podcasts:")
print(
    df_removed[["title", "primary_category", "template_neighbors"]]
    .head(10)
)
