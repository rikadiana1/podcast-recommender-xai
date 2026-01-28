import os
import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
import shap
from sklearn.linear_model import LogisticRegression


# CONFIG

DATA_PATH = "podcasts_clean_v2.csv"

MODEL_DIR = "nlp_outputs_v2/train_model"
NLP_DIR = "nlp_outputs_v2"  
OUT_DIR = os.path.join(MODEL_DIR, "shap_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ranker_logreg.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

PAIRS_PATH = os.path.join(MODEL_DIR, "pairs_test.csv")
X_BG_PATH = os.path.join(MODEL_DIR, "shap_background_X_train.npz")

# SBERT embeddings used by the hybrid model
SBERT_EMB_PATH = os.path.join(NLP_DIR, "X_embeddings.npy")

TOP_TERMS = 20
N_SEEDS = 3
RANDOM_STATE = 42


# HELPERS

def load_and_normalize_embeddings(path: str) -> np.ndarray:
    """Load SBERT embeddings and L2-normalize defensively."""
    X = np.load(path).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / norms).astype(np.float32)

def sbert_cosine(seed_idx: int, cand_idx: int, X_sbert: np.ndarray) -> float:
    """Cosine similarity; embeddings are normalized so cosine = dot product."""
    return float(np.dot(X_sbert[seed_idx], X_sbert[cand_idx]))

def build_pair_features(
    X_tfidf: sparse.csr_matrix,
    X_sbert: np.ndarray,
    seed_idx: int,
    cand_idx: int
) -> sparse.csr_matrix:
    """
    Build features EXACTLY like the hybrid training code:
      - TF-IDF overlap (sparse)
      - + 1 column: SBERT cosine similarity
    """
    X_text = X_tfidf[seed_idx].multiply(X_tfidf[cand_idx])  # (1, vocab)
    sem = np.array([[sbert_cosine(seed_idx, cand_idx, X_sbert)]], dtype=np.float32)  # (1, 1)
    return sparse.hstack([X_text, sparse.csr_matrix(sem)], format="csr")


# MAIN

def main():
    print("Loading artifacts...")

    df = pd.read_csv(DATA_PATH)

    model: LogisticRegression = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
    feature_names = load(FEATURE_NAMES_PATH)

    pairs = pd.read_csv(PAIRS_PATH)
    X_bg = sparse.load_npz(X_BG_PATH)

    if "text_full" not in df.columns:
        raise KeyError("Dataset must contain 'text_full' column.")

    # TF-IDF document matrix
    X_tfidf = vectorizer.transform(df["text_full"].astype(str).tolist()).tocsr()

    # SBERT embeddings
    if not os.path.isfile(SBERT_EMB_PATH):
        raise FileNotFoundError(
            f"SBERT embeddings not found at: {SBERT_EMB_PATH}\n"
            "This SHAP script is for the hybrid model (TF-IDF overlap + SBERT cosine)."
        )
    X_sbert = load_and_normalize_embeddings(SBERT_EMB_PATH)

    # Sanity checks: dimensions must match training
    expected_dim = X_tfidf.shape[1] + 1
    if X_bg.shape[1] != expected_dim:
        raise ValueError(
            f"Background matrix has {X_bg.shape[1]} features, expected {expected_dim} "
            f"(tfidf_vocab={X_tfidf.shape[1]} + 1 sbert feature)."
        )
    if len(feature_names) != expected_dim:
        raise ValueError(
            f"feature_names length={len(feature_names)} does not match expected_dim={expected_dim}."
        )
    if X_sbert.shape[0] != X_tfidf.shape[0]:
        raise ValueError(
            f"Embeddings rows ({X_sbert.shape[0]}) != number of items ({X_tfidf.shape[0]})."
        )

    # seed selection from test pairs
    rng = np.random.default_rng(RANDOM_STATE)
    unique_seeds = pairs["seed_idx"].unique()
    if len(unique_seeds) < N_SEEDS:
        print(f"⚠️ Only {len(unique_seeds)} unique seeds available; using all.")
        seeds = unique_seeds
    else:
        seeds = rng.choice(unique_seeds, size=N_SEEDS, replace=False)

    print("Initializing SHAP LinearExplainer...")
    explainer = shap.LinearExplainer(
        model,
        X_bg,
        feature_perturbation="interventional"
    )

    for seed in seeds:
        subset = pairs[pairs["seed_idx"] == seed]

        # select 2 positives + 2 negatives for this seed
        pos = subset[subset["y"] == 1].head(2)
        neg = subset[subset["y"] == 0].head(2)
        cases = pd.concat([pos, neg], axis=0)

        for _, row in cases.iterrows():
            s = int(row["seed_idx"])
            c = int(row["cand_idx"])
            y = int(row["y"])

            X_pair = build_pair_features(X_tfidf, X_sbert, s, c)
            proba = model.predict_proba(X_pair)[0, 1]

            # SHAP values
            shap_vals = explainer.shap_values(X_pair)

            # ensure dense 1D vector
            if sparse.issparse(shap_vals):
                shap_vals = shap_vals.toarray().ravel()
            else:
                shap_vals = np.asarray(shap_vals).ravel()

            expl = pd.DataFrame({
                "seed_idx": s,
                "seed_title": df.loc[s, "title"] if "title" in df.columns else "",
                "seed_category": df.loc[s, "primary_category"] if "primary_category" in df.columns else "",
                "cand_idx": c,
                "cand_title": df.loc[c, "title"] if "title" in df.columns else "",
                "cand_category": df.loc[c, "primary_category"] if "primary_category" in df.columns else "",
                "label": y,
                "model_proba_relevant": float(proba),
                "term": feature_names,
                "shap_value": shap_vals
            })

            expl["direction"] = np.where(
                expl["shap_value"] > 0,
                "supports_relevant",
                "supports_not_relevant"
            )

            expl = expl.reindex(
                expl["shap_value"].abs().sort_values(ascending=False).index
            ).head(TOP_TERMS)

            out_path = os.path.join(OUT_DIR, f"shap_seed{s}_cand{c}_y{y}.csv")
            expl.to_csv(out_path, index=False, encoding="utf-8")
            print(f"Saved: {out_path}")

    print("SHAP explanations finished.")


if __name__ == "__main__":
    main()
