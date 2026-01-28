
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Config

OUT_DIR = "nlp_outputs"
DATA_PATH = "podcasts_clean.csv"

SIM_TFIDF_PATH = os.path.join(OUT_DIR, "similarity_tfidf.npy")   # TFIDF+SVD cosine matrix
EMB_PATH = os.path.join(OUT_DIR, "embeddings_st.npy")            # optional SentenceTransformer embeddings

TOP_K = 10
QUERY_INDEX = 0

# Only recommend if similarity >= THRESH (prevents weak matches)
THRESH = 0.05

# Fallback settings 
FALLBACK_USE_EMBEDDINGS_IF_AVAILABLE = True
FALLBACK_USE_TFIDF_WITHIN_CATEGORY = True

EXCEL_OUT = os.path.join(OUT_DIR, "explanations.xlsx")


def load_df():
    df = pd.read_csv(DATA_PATH)
    required = {"title", "rss_url", "primary_category", "text_full"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")
    df["text_full"] = df["text_full"].astype(str)
    return df


def recommend_from_similarity_matrix(sim: np.ndarray, df: pd.DataFrame, q_idx: int, k: int, thresh: float):
    scores = sim[q_idx].astype(float).copy()
    scores[q_idx] = -np.inf  # remove self

    good = np.where(scores >= thresh)[0]
    if len(good) == 0:
        return None

    ranked = good[np.argsort(scores[good])[::-1]]
    top_idx = ranked[:k]

    recs = df.loc[top_idx, ["title", "rss_url", "primary_category"]].copy()
    recs.insert(0, "rank", range(1, len(recs) + 1))
    recs["similarity_score"] = scores[top_idx]
    recs["source"] = f"tfidf_svd_matrix(thresh={thresh})"
    return recs


def fallback_embeddings(df: pd.DataFrame, q_idx: int, k: int, thresh: float):
    if not os.path.exists(EMB_PATH):
        return None

    emb = np.load(EMB_PATH).astype(np.float32)
    # embeddings assumed normalized (dot product = cosine)
    scores = emb[q_idx] @ emb.T
    scores[q_idx] = -np.inf

    good = np.where(scores >= thresh)[0]
    if len(good) == 0:
        return None

    ranked = good[np.argsort(scores[good])[::-1]]
    top_idx = ranked[:k]

    recs = df.loc[top_idx, ["title", "rss_url", "primary_category"]].copy()
    recs.insert(0, "rank", range(1, len(recs) + 1))
    recs["similarity_score"] = scores[top_idx]
    recs["source"] = f"embeddings(thresh={thresh})"
    return recs


def fallback_tfidf_within_category(df: pd.DataFrame, q_idx: int, k: int, thresh: float):
    query_cat = df.loc[q_idx, "primary_category"]
    mask = (df["primary_category"] == query_cat).values
    candidate_idx = np.where(mask)[0]

    # if category too small, use all
    if len(candidate_idx) < 5:
        candidate_idx = np.arange(len(df))

    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        sublinear_tf=True,
    )
    X = vec.fit_transform(df["text_full"].tolist())

    sims = cosine_similarity(X[q_idx], X).ravel().astype(float)
    sims[q_idx] = -np.inf

    # restrict to candidates
    sims_c = sims[candidate_idx]
    good_local = np.where(sims_c >= thresh)[0]
    if len(good_local) == 0:
        return None

    ranked_local = good_local[np.argsort(sims_c[good_local])[::-1]]
    top_local = ranked_local[:k]
    top_idx = candidate_idx[top_local]

    recs = df.loc[top_idx, ["title", "rss_url", "primary_category"]].copy()
    recs.insert(0, "rank", range(1, len(recs) + 1))
    recs["similarity_score"] = sims[top_idx]
    recs["source"] = f"tfidf_fallback_same_category({query_cat},thresh={thresh})"
    return recs


def main():
    print(f"Using OUT_DIR: {OUT_DIR}")
    print(f"Using DATA_PATH: {DATA_PATH}")

    df = load_df()

    q_idx = QUERY_INDEX
    print("\nQUERY:")
    print(df.loc[q_idx, ["title", "primary_category"]])

    # 1) primary method: TFIDF+SVD similarity matrix
    if not os.path.exists(SIM_TFIDF_PATH):
        raise FileNotFoundError(f"Missing similarity matrix: {SIM_TFIDF_PATH}")

    sim = np.load(SIM_TFIDF_PATH)
    recs = recommend_from_similarity_matrix(sim, df, q_idx, TOP_K, THRESH)

    # 2) fallbacks (only if no matches)
    if recs is None and FALLBACK_USE_EMBEDDINGS_IF_AVAILABLE:
        print("No TFIDF+SVD matches above threshold. Trying embeddings fallback...")
        recs = fallback_embeddings(df, q_idx, TOP_K, thresh=max(0.2, THRESH))  # embeddings threshold usually higher

    if recs is None and FALLBACK_USE_TFIDF_WITHIN_CATEGORY:
        print("No embedding matches (or embeddings missing). Trying TF-IDF within-category fallback...")
        recs = fallback_tfidf_within_category(df, q_idx, TOP_K, THRESH)

    if recs is None:
        print("No recommendations found above threshold.")
        return

    print("\nTop-k source:", recs["source"].iloc[0])
    print("\nRECS:")
    print(recs)

    # Save output
    try:
        recs.to_excel(EXCEL_OUT, index=False)
        print(f"Excel gespeichert: {EXCEL_OUT}")
    except Exception as e:
        csv_out = EXCEL_OUT.replace(".xlsx", ".csv")
        recs.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"Excel failed ({e}).CSV gespeichert: {csv_out}")


if __name__ == "__main__":
    main()
