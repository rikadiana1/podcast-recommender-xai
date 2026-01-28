import os
import re
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer


# CONFIG

DATA_PATH = "podcasts_clean_v2.csv"
OUT_DIR = "nlp_outputs_v2"
os.makedirs(OUT_DIR, exist_ok=True)

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.8

USE_SVD = True
SVD_COMPONENTS = 100
RANDOM_STATE = 42

TOPK_STORE = 50           # how many neighbors to store
TOPN_SIM_STATS = 10       # stats computed on top-n neighbors
TOPK_MIN_SIM = 0.0        # store only neighbors with sim >= threshold

USE_SBERT = True
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


SAVE_FULL_SIMILARITY = True 

DOMAIN_PHRASE_STOPWORDS = {
    "new episode", "new episodes", "listen now", "available on",
    "subscribe now", "every week", "each week", "every day",
    "new podcast", "weekly podcast", "daily podcast"
}

DOMAIN_STOPWORDS = {
    "podcast", "podcasts", "show", "shows", "episode", "episodes",
    "daily", "weekly", "new", "latest", "live", "radio",
    "host", "hosts", "guest", "guests",
    "subscribe", "subscribed", "subscription",
    "follow", "following", "listen", "listening", "download", "stream", "available",
    "spotify", "itunes", "apple", "google", "youtube", "patreon", "newsletter",
    "website", "link", "links"
}



# TEXT PREP

def normalize_text(s: str) -> str:
    
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^0-9a-zäöüß\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def remove_phrase_stopwords(texts, phrases):
    if not phrases:
        return texts
    # normalize phrases 
    phrases = [normalize_text(p) for p in phrases]
    escaped = [re.escape(p) for p in phrases if p]
    if not escaped:
        return texts
    pat = re.compile(r"(?i)\b(" + "|".join(e.replace(r"\ ", r"\s+") for e in escaped) + r")\b")
    return [re.sub(pat, " ", t) for t in texts]


def top_df_terms(texts, top_n=50, min_len=3):
    cv = CountVectorizer(
        stop_words=None,
        token_pattern=r"(?u)\b[0-9a-zA-Zäöüß][0-9a-zA-Zäöüß]{1,}\b",
        lowercase=True
    )
    X = cv.fit_transform(texts)
    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    vocab = np.array(cv.get_feature_names_out())
    order = np.argsort(df_counts)[::-1]

    out = []
    for term, dfc in zip(vocab[order], df_counts[order]):
        if len(term) < min_len:
            continue
        out.append((term, int(dfc)))
        if len(out) >= top_n:
            break
    return out



# KNN HELPERS 

def knn_topk_cosine(X, k: int):

    n = X.shape[0]
    if n <= 1:
        return (
            np.full((n, 0), -1, dtype=np.int32),
            np.full((n, 0), -1.0, dtype=np.float32),
        )

    k = min(k, n - 1)
    # ask for k+1 because self is included as nearest with distance 0
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    # drop self (first column)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    sims = (1.0 - distances).astype(np.float32)
    return indices.astype(np.int32), sims


def apply_min_sim_filter(indices, sims, min_sim: float):
    """
    Keep only neighbors with sim >= min_sim; fill the rest with -1.
    """
    if min_sim is None:
        return indices, sims

    idx_out = indices.copy()
    sim_out = sims.copy()

    mask = sim_out >= float(min_sim)
    idx_out[~mask] = -1
    sim_out[~mask] = -1.0
    return idx_out, sim_out


def topn_stats_from_topk(sims_topk: np.ndarray, topn: int):
    """
    Compute per-row mean(topn) and max based ONLY on stored neighbors.
    Ignores -1.0 entries (filtered-out neighbors).
    """
    n = sims_topk.shape[0]
    if sims_topk.size == 0:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    topn = min(topn, sims_topk.shape[1])
    # take first topn
    block = sims_topk[:, :topn].copy()

    # ignore invalid
    valid = block >= 0.0
    # mean over valid entries (avoid division by 0)
    sums = np.where(valid, block, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    mean = np.divide(sums, np.maximum(counts, 1), dtype=np.float32).astype(np.float32)

    # max over valid entries
    block[~valid] = -1.0
    mx = block.max(axis=1).astype(np.float32)

    return mean, mx



# MAIN

def main():
    df = pd.read_csv(DATA_PATH)

    required_cols = {"text_full", "primary_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # normalize + remove boilerplate phrases
    df["text_full"] = df["text_full"].astype(str).map(normalize_text)
    texts = remove_phrase_stopwords(df["text_full"].tolist(), DOMAIN_PHRASE_STOPWORDS)

    # mapping export
    mapping_cols = [c for c in ["title", "rss_url", "primary_category"] if c in df.columns]
    df[mapping_cols].to_csv(
        os.path.join(OUT_DIR, "index_mapping.csv"),
        index=False,
        encoding="utf-8"
    )


    # DF diagnostics
    df_candidates = top_df_terms(texts, top_n=50)
    pd.DataFrame(df_candidates, columns=["term", "doc_freq"]).to_csv(
        os.path.join(OUT_DIR, "corpus_df_top_terms.csv"),
        index=False,
        encoding="utf-8"
    )

    # stopwords: english + domain
    stopwords = sorted(set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS))

    # TF-IDF
    print("=== TF-IDF ===")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        stop_words=stopwords,
        token_pattern=r"(?u)\b[0-9a-zA-Zäöüß][0-9a-zA-Zäöüß]{1,}\b",
        sublinear_tf=True
    )

    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    print(f"TF-IDF shape={X_tfidf.shape}, vocab={len(feature_names)}")

    # lowest idf terms diagnostics
    idf_df = pd.DataFrame({"term": feature_names, "idf": tfidf_vectorizer.idf_}).sort_values("idf", ascending=True)
    idf_df.head(100).to_csv(
        os.path.join(OUT_DIR, "lowest_idf_terms.csv"),
        index=False,
        encoding="utf-8"
    )

    # optional SVD (LSA)
    if USE_SVD:
        print("=== SVD (LSA) ===")
        n_components = min(SVD_COMPONENTS, X_tfidf.shape[1] - 1)
        if n_components < 2:
            raise ValueError(f"SVD_COMPONENTS invalid for n_features={X_tfidf.shape[1]} (n_components={n_components})")

        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        X_vec = svd.fit_transform(X_tfidf).astype(np.float32)

        explained = float(svd.explained_variance_ratio_.sum())
        print(f"SVD shape={X_vec.shape}, explained_var_sum={explained:.3f}")

        dump(svd, os.path.join(OUT_DIR, "svd_tfidf.joblib"))
        np.save(os.path.join(OUT_DIR, "X_tfidf_svd.npy"), X_vec)
    else:
        svd = None
        X_vec = X_tfidf  # keep sparse
        explained = None

    # normalize
    X_norm = normalize(X_vec, norm="l2", axis=1)

    # TF-IDF top-k via KNN 
    print("=== TF-IDF KNN (cosine) ===")
    topk_idx_text, topk_scores_text = knn_topk_cosine(X_norm, k=TOPK_STORE)
    topk_idx_text, topk_scores_text = apply_min_sim_filter(topk_idx_text, topk_scores_text, TOPK_MIN_SIM)

    mean_top10_tfidf, max_sim_tfidf = topn_stats_from_topk(topk_scores_text, TOPN_SIM_STATS)
    print(f"TF-IDF stats: mean_top{TOPN_SIM_STATS}={mean_top10_tfidf.mean():.4f}, max_mean={max_sim_tfidf.mean():.4f}")

    # save TF-IDF artifacts
    dump(tfidf_vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    np.save(os.path.join(OUT_DIR, "topk_indices_text.npy"), topk_idx_text)
    np.save(os.path.join(OUT_DIR, "topk_scores_text.npy"), topk_scores_text)

    # optional full sim 
    if SAVE_FULL_SIMILARITY:
        from sklearn.metrics.pairwise import cosine_similarity
        sim_tfidf = cosine_similarity(X_norm)
        np.save(os.path.join(OUT_DIR, "similarity_tfidf.npy"), sim_tfidf.astype(np.float32))

    # SBERT
    if USE_SBERT:
        print("=== SBERT ===")
        model = SentenceTransformer(EMBEDDING_MODEL)
        X_emb = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        print("=== SBERT KNN (cosine) ===")
        topk_idx_emb, topk_scores_emb = knn_topk_cosine(X_emb, k=TOPK_STORE)
        topk_idx_emb, topk_scores_emb = apply_min_sim_filter(topk_idx_emb, topk_scores_emb, TOPK_MIN_SIM)

        mean_top10_emb, max_sim_emb = topn_stats_from_topk(topk_scores_emb, TOPN_SIM_STATS)
        print(f"SBERT stats: mean_top{TOPN_SIM_STATS}={mean_top10_emb.mean():.4f}, max_mean={max_sim_emb.mean():.4f}")

        np.save(os.path.join(OUT_DIR, "X_embeddings.npy"), X_emb)
        np.save(os.path.join(OUT_DIR, "topk_indices_embeddings.npy"), topk_idx_emb)
        np.save(os.path.join(OUT_DIR, "topk_scores_embeddings.npy"), topk_scores_emb)

        if SAVE_FULL_SIMILARITY:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_emb = cosine_similarity(X_emb)
            np.save(os.path.join(OUT_DIR, "similarity_embeddings.npy"), sim_emb.astype(np.float32))
    else:
        mean_top10_emb = np.zeros(len(df), dtype=np.float32)
        max_sim_emb = np.zeros(len(df), dtype=np.float32)

    # metrics export
    metrics = pd.DataFrame({
        "title": df["title"] if "title" in df.columns else [""] * len(df),
        "rss_url": df["rss_url"] if "rss_url" in df.columns else [""] * len(df),
        "primary_category": df["primary_category"].astype(str),
        "text_full_len": df["text_full"].str.len().astype(int),

        "mean_top10_sim_tfidf": mean_top10_tfidf,
        "max_sim_tfidf": max_sim_tfidf,
        "mean_top10_sim_sbert": mean_top10_emb,
        "max_sim_sbert": max_sim_emb
    })
    metrics.to_csv(os.path.join(OUT_DIR, "podcast_similarity_metrics.csv"), index=False, encoding="utf-8")

    # quick sanity recs
    example_idx = 0
    print("\nExample recommendations (TF-IDF top-5, stored >= min_sim):")
    ex_idx = topk_idx_text[example_idx][:5]
    ex_idx = ex_idx[ex_idx >= 0]
    if ex_idx.size:
        cols = ["primary_category"]
        if "title" in df.columns:
            cols.insert(0, "title")
        print(df.iloc[ex_idx][cols])
    else:
        print("(none above threshold)")

    if USE_SBERT:
        print("\nExample recommendations (SBERT top-5, stored >= min_sim):")
        ex_idx = topk_idx_emb[example_idx][:5]
        ex_idx = ex_idx[ex_idx >= 0]
        if ex_idx.size:
            cols = ["primary_category"]
            if "title" in df.columns:
                cols.insert(0, "title")
            print(df.iloc[ex_idx][cols])
        else:
            print("(none above threshold)")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()