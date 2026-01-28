
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump



# Configuration

DATA_PATH = "podcasts_clean.csv"
OUT_DIR = "nlp_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.8

SVD_COMPONENTS = 100          
RANDOM_STATE = 42


def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    required_cols = {"title", "rss_url", "primary_category", "text_full"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    texts = df["text_full"].astype(str).tolist()
    print(f"Loaded dataset with {len(texts)} podcasts")

    # Save index → podcast mapping
    df[["title", "rss_url", "primary_category"]].to_csv(
        os.path.join(OUT_DIR, "index_mapping.csv"),
        index=False,
        encoding="utf-8"
    )

    # 2) TF-IDF
    print("\n=== TF-IDF Vectorization ===")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        sublinear_tf=True,
    )

    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    # 3) TruncatedSVD (LSA)
    print("\n=== TruncatedSVD (LSA) ===")
    # ensure n_components is valid (< n_features)
    n_components = min(SVD_COMPONENTS, X_tfidf.shape[1] - 1)
    if n_components < 2:
        raise ValueError(
            f"SVD_COMPONENTS too small/invalid for n_features={X_tfidf.shape[1]} "
            f"(computed n_components={n_components})."
        )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    X_tfidf_svd = svd.fit_transform(X_tfidf)

    # Normalize vectors => cosine similarity is stable and comparable
    X_tfidf_svd = Normalizer(copy=False).fit_transform(X_tfidf_svd)

    print(f"SVD matrix shape: {X_tfidf_svd.shape}")
    print(f"Explained variance (sum): {svd.explained_variance_ratio_.sum():.3f}")

    # 4) Cosine Similarity (TF-IDF + SVD)
    print("\n=== Cosine Similarity (TF-IDF + SVD) ===")
    sim_tfidf = cosine_similarity(X_tfidf_svd)

    print(f"Similarity matrix shape: {sim_tfidf.shape}")
    print(f"Similarity sanity check (self-similarity): {sim_tfidf[0, 0]:.3f}")
    print(f"Min/Max similarity: {sim_tfidf.min():.6f} / {sim_tfidf.max():.6f}")

    # Save artifacts
    dump(tfidf_vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    dump(svd, os.path.join(OUT_DIR, "svd_tfidf.joblib"))

    np.save(os.path.join(OUT_DIR, "X_tfidf_svd.npy"), X_tfidf_svd.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "similarity_tfidf.npy"), sim_tfidf.astype(np.float32))

    print("TF-IDF pipeline saved ")


if __name__ == "__main__":
    main()
