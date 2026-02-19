import os
import numpy as np
import pandas as pd
from joblib import load

# Pfade
OUT_DIR = r"C:\Users\rikad\Documents\podcast-recommender-xai\nlp_recommender\nlp_outputs_v2"
DATA_DIR = r"C:\Users\rikad\Documents\podcast-recommender-xai\data"

TEXT_CSV = os.path.join(DATA_DIR, "podcasts_clean_v2.csv")
VECTORIZER_PATH = os.path.join(OUT_DIR, "tfidf_vectorizer.joblib")
TOPK_IDX_NPY = os.path.join(OUT_DIR, "topk_indices_text.npy")
TOPK_SCORES_NPY = os.path.join(OUT_DIR, "topk_scores_text.npy")

OUT_XLSX = os.path.join(OUT_DIR, "intrinsic_top5_terms.xlsx")

# Seeds / Parameter
SEEDS = [
    "The Diddy Diaries",
    "NEO420 Talks",
    "Locked On NHL"
]
TOP_K = 5
TOP_TERMS = 8

# Laden
df = pd.read_csv(TEXT_CSV)
topk_idx = np.load(TOPK_IDX_NPY)
topk_scores = np.load(TOPK_SCORES_NPY)
vectorizer = load(VECTORIZER_PATH)

# TF-IDF Matrix
X = vectorizer.transform(df["text_full"].fillna("").astype(str))
feature_names = np.array(vectorizer.get_feature_names_out())


def find_seed_index(title_query: str) -> int:
    # exakter match
    hit = df.index[df["title"] == title_query].tolist()
    if hit:
        return int(hit[0])

    # contains fallback
    hit = df.index[df["title"].fillna("").str.contains(title_query, regex=False)].tolist()
    if hit:
        return int(hit[0])

    raise ValueError(f"Seed nicht gefunden: {title_query}")


def top_overlap_terms_with_scores(i: int, j: int, topn: int = 8) -> str:
    """
    Liefert die Top-Overlapping-Terme inkl. Beitragswert (TF-IDF_i * TF-IDF_j),
    sortiert absteigend nach Beitrag.
    Format: "term (0.0123), term2 (0.0098), ..."
    """
    prod = X[i].multiply(X[j])  # sparse overlap: elementwise product
    if prod.nnz == 0:
        return ""

    idx = prod.indices
    vals = prod.data

    order = np.argsort(vals)[::-1][:topn]

    terms_with_scores = [
        f"{feature_names[idx[k]]} ({vals[k]:.4f})"
        for k in order
    ]
    return ", ".join(terms_with_scores)


# Excel-Tabelle
rows = []
for seed_query in SEEDS:
    seed_i = find_seed_index(seed_query)

    rec_indices = topk_idx[seed_i][:TOP_K]
    rec_scores = topk_scores[seed_i][:TOP_K]

    for rank, (j, s) in enumerate(zip(rec_indices, rec_scores), start=1):
        j = int(j)
        if j < 0:
            continue

        rows.append({
            "seed_title": df.at[seed_i, "title"],
            "seed_category": df.at[seed_i, "primary_category"],
            "rank": rank,
            "rec_title": df.at[j, "title"],
            "rec_category": df.at[j, "primary_category"],
            "similarity": float(s),
            "top_terms": top_overlap_terms_with_scores(seed_i, j, topn=TOP_TERMS)
        })

out = pd.DataFrame(rows).sort_values(["seed_title", "rank"])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="top5_recs_terms", index=False)

print("Excel erstellt:", OUT_XLSX)
print("Zeilen:", len(out))

