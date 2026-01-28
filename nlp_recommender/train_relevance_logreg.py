import os, json
import numpy as np
import pandas as pd
from joblib import load, dump
from scipy import sparse

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# CONFIG

DATA_PATH = "podcasts_clean_v2.csv"
NLP_DIR = "nlp_outputs_v2"
OUT_DIR = os.path.join(NLP_DIR, "train_model")
os.makedirs(OUT_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(NLP_DIR, "tfidf_vectorizer.joblib")
TOPK_IDX_PATH = os.path.join(NLP_DIR, "topk_indices_text.npy")

# SBERT embeddings
# Produced by your earlier pipeline as: np.save(os.path.join(OUT_DIR, "X_embeddings.npy"), X_emb)
SBERT_EMB_PATH = os.path.join(NLP_DIR, "X_embeddings.npy")
USE_SBERT_FEATURE = True  # set False to train TF-IDF overlap only

# weak labels: positives are top K_POS from baseline KNN
K_POS = 10
NEG_PER_POS = 1

# hard negatives come from ranks [K_POS:HARD_POOL_K) in baseline list
HARD_POOL_K = 50

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Logistic regression
C = 1.0
MAX_ITER = 4000

# Ranking eval (against proxy positives)
EVAL_K = 10
EVAL_POOL = 200

# SHAP background size (rows)
SHAP_BG_N = 2000


# HELPERS

def _ensure_2d_col(x: np.ndarray) -> np.ndarray:
    """Ensure shape (n, 1)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def sbert_cosine_feature(seeds: np.ndarray, cands: np.ndarray, X_sbert: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between SBERT embeddings for (seed, cand) pairs.
    Assumes embeddings are L2-normalized; then cosine == dot product.
    Returns shape (n_pairs, 1).
    """
    seeds = seeds.astype(np.int32, copy=False)
    cands = cands.astype(np.int32, copy=False)
    # dot product row-wise
    sims = np.sum(X_sbert[seeds] * X_sbert[cands], axis=1, dtype=np.float32)
    return _ensure_2d_col(sims)

def sample_negatives(
    seed: int,
    positives: np.ndarray,
    topk_row: np.ndarray,
    n_items: int,
    n_neg: int,
    rng: np.random.Generator
) -> np.ndarray:
    pos_set = set(map(int, positives.tolist()))
    pos_set.add(int(seed))

    hard_pool_k = min(HARD_POOL_K, len(topk_row))
    hard_pool = [int(x) for x in topk_row[K_POS:hard_pool_k] if int(x) not in pos_set]

    negs = []
    # take hard negatives first
    while len(negs) < n_neg and hard_pool:
        negs.append(hard_pool.pop())

    # fill remaining with random negatives
    while len(negs) < n_neg:
        j = int(rng.integers(0, n_items))
        if j not in pos_set and j not in negs:
            negs.append(j)

    return np.array(negs, dtype=np.int32)

def build_pairs(
    X_tfidf: sparse.csr_matrix,
    topk_idx: np.ndarray,
    seed_items: np.ndarray,
    rng: np.random.Generator,
    X_sbert: np.ndarray | None = None
):
    """
    Build (seed, candidate) pairs with weak labels:
      - positives: baseline top K_POS neighbors
      - negatives: hard negatives from later ranks + random

    Features:
      - TF-IDF overlap: X_tfidf[seed] * X_tfidf[cand]  (sparse)
      - optional SBERT cosine similarity as 1 extra column (dense -> sparse)
    """
    seeds, cands, y = [], [], []
    k_pos = min(K_POS, topk_idx.shape[1])

    for s in seed_items.astype(np.int32, copy=False):
        pos = topk_idx[s, :k_pos].astype(np.int32)

        # positives
        for c in pos:
            seeds.append(int(s)); cands.append(int(c)); y.append(1)

        # negatives
        n_neg = len(pos) * NEG_PER_POS
        negs = sample_negatives(
            seed=int(s),
            positives=pos,
            topk_row=topk_idx[s].astype(np.int32),
            n_items=X_tfidf.shape[0],
            n_neg=n_neg,
            rng=rng
        )
        for c in negs:
            seeds.append(int(s)); cands.append(int(c)); y.append(0)

    seeds = np.array(seeds, dtype=np.int32)
    cands = np.array(cands, dtype=np.int32)
    y = np.array(y, dtype=np.int8)

    # Feature block 1: TF-IDF overlap (sparse)
    X_text = X_tfidf[seeds].multiply(X_tfidf[cands])  # csr

    # Optional Feature block 2: SBERT cosine similarity (1 column)
    if X_sbert is not None:
        X_sem = sbert_cosine_feature(seeds, cands, X_sbert)  # (n,1) dense
        X_sem_sp = sparse.csr_matrix(X_sem)
        X = sparse.hstack([X_text, X_sem_sp], format="csr")
    else:
        X = X_text

    pairs_df = pd.DataFrame({"seed_idx": seeds, "cand_idx": cands, "y": y})
    return X, y, pairs_df

def build_features_for_eval(
    X_tfidf: sparse.csr_matrix,
    seeds: np.ndarray,
    cands: np.ndarray,
    X_sbert: np.ndarray | None = None
):
    """
    Build features for evaluation pairs (seed, candidate), same structure as training.
    """
    seeds = seeds.astype(np.int32, copy=False)
    cands = cands.astype(np.int32, copy=False)

    X_text = X_tfidf[seeds].multiply(X_tfidf[cands])

    if X_sbert is not None:
        X_sem = sbert_cosine_feature(seeds, cands, X_sbert)
        X = sparse.hstack([X_text, sparse.csr_matrix(X_sem)], format="csr")
    else:
        X = X_text

    return X

def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k].astype(float)
    if rels.size == 0:
        return 0.0
    return float(np.sum(rels / np.log2(np.arange(2, rels.size + 2))))

def ndcg_at_k(rels: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(rels, k)
    ideal = np.sort(rels)[::-1]
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0.0 else float(dcg / idcg)


# MAIN

def main():
    df = pd.read_csv(DATA_PATH)

    if "text_full" not in df.columns:
        raise KeyError("Dataset must contain 'text_full' column.")

    # load TF-IDF vectorizer and transform texts
    vectorizer = load(VECTORIZER_PATH)
    X_tfidf = vectorizer.transform(df["text_full"].astype(str).tolist()).tocsr()

    # load baseline neighbor indices
    topk_idx = np.load(TOPK_IDX_PATH).astype(np.int32)

    # optional: load SBERT embeddings
    X_sbert = None
    if USE_SBERT_FEATURE:
        if not os.path.isfile(SBERT_EMB_PATH):
            raise FileNotFoundError(
                f"USE_SBERT_FEATURE=True but embeddings not found at: {SBERT_EMB_PATH}\n"
                f"Expected file from earlier pipeline: X_embeddings.npy"
            )
        X_sbert = np.load(SBERT_EMB_PATH).astype(np.float32)
        if X_sbert.shape[0] != X_tfidf.shape[0]:
            raise ValueError(
                f"SBERT embeddings rows ({X_sbert.shape[0]}) != n_items ({X_tfidf.shape[0]})."
            )
        # If embeddings are not normalized, cosine != dot.
        norms = np.linalg.norm(X_sbert, axis=1, keepdims=True) + 1e-12
        X_sbert = (X_sbert / norms).astype(np.float32)

    n_items = X_tfidf.shape[0]
    items = np.arange(n_items, dtype=np.int32)

    # split by seed items (train/test seeds disjoint)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(items, groups=items))
    train_items, test_items = items[train_idx], items[test_idx]

    rng = np.random.default_rng(RANDOM_STATE)

    # build pair datasets
    X_train, y_train, pairs_train = build_pairs(X_tfidf, topk_idx, train_items, rng, X_sbert=X_sbert)
    X_test,  y_test,  pairs_test  = build_pairs(
        X_tfidf, topk_idx, test_items, np.random.default_rng(RANDOM_STATE + 1), X_sbert=X_sbert
    )

    # feature names (TF-IDF terms + optional SBERT similarity)
    feature_names = vectorizer.get_feature_names_out().tolist()
    if X_sbert is not None:
        feature_names.append("SBERT_cosine_similarity")

    # train logistic regression ranker
    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=C,
        max_iter=MAX_ITER,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # threshold-free eval on pair classification
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    # ranking eval per held-out seed (re-rank baseline candidate pool)
    pool_k = min(EVAL_POOL, topk_idx.shape[1])
    k_pos = min(K_POS, topk_idx.shape[1])

    precisions, recalls, ndcgs = [], [], []
    for seed in test_items:
        positives = set(map(int, topk_idx[seed, :k_pos].tolist()))

        cand_pool = topk_idx[seed, :pool_k].astype(np.int32)
        cand_pool = cand_pool[cand_pool != seed]
        if cand_pool.size == 0:
            continue

        seeds_arr = np.full_like(cand_pool, seed, dtype=np.int32)
        X_eval = build_features_for_eval(X_tfidf, seeds_arr, cand_pool, X_sbert=X_sbert)
        scores = model.predict_proba(X_eval)[:, 1]

        order = np.argsort(-scores)
        ranked = cand_pool[order][:EVAL_K]

        hits = np.array([1 if int(c) in positives else 0 for c in ranked], dtype=np.int32)
        precisions.append(float(hits.sum() / max(EVAL_K, 1)))
        recalls.append(float(hits.sum() / max(len(positives), 1)))
        ndcgs.append(ndcg_at_k(hits, EVAL_K))

    p_at_k = float(np.mean(precisions)) if precisions else 0.0
    r_at_k = float(np.mean(recalls)) if recalls else 0.0
    ndcg_k = float(np.mean(ndcgs)) if ndcgs else 0.0

    print(f"AUC={auc:.4f}  AP={ap:.4f}")
    print(f"Precision@{EVAL_K}={p_at_k:.4f}  Recall@{EVAL_K}={r_at_k:.4f}  NDCG@{EVAL_K}={ndcg_k:.4f}")

    # SHAP background save
    bg_n = min(SHAP_BG_N, X_train.shape[0])
    bg_idx = np.random.default_rng(RANDOM_STATE).choice(X_train.shape[0], size=bg_n, replace=False)
    sparse.save_npz(os.path.join(OUT_DIR, "shap_background_X_train.npz"), X_train[bg_idx])

    # save artifacts
    dump(model, os.path.join(OUT_DIR, "ranker_logreg.joblib"))
    dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    dump(feature_names, os.path.join(OUT_DIR, "feature_names.joblib"))

    pairs_train.to_csv(os.path.join(OUT_DIR, "pairs_train.csv"), index=False, encoding="utf-8")
    pairs_test.to_csv(os.path.join(OUT_DIR, "pairs_test.csv"), index=False, encoding="utf-8")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "AUC": auc,
            "AP": ap,
            f"Precision@{EVAL_K}": p_at_k,
            f"Recall@{EVAL_K}": r_at_k,
            f"NDCG@{EVAL_K}": ndcg_k,
            "K_POS": K_POS,
            "NEG_PER_POS": NEG_PER_POS,
            "HARD_POOL_K": HARD_POOL_K,
            "TEST_SIZE": TEST_SIZE,
            "C": C,
            "MAX_ITER": MAX_ITER,
            "EVAL_POOL": pool_k,
            "SHAP_BG_N": bg_n,
            "USE_SBERT_FEATURE": bool(X_sbert is not None),
            "note": (
                "Weak supervision: positives are baseline TF-IDF top-k neighbors. "
                + ("Features: TF-IDF overlap + SBERT cosine similarity." if X_sbert is not None else "Features: TF-IDF overlap only.")
            )
        }, f, ensure_ascii=False, indent=2)

    print("Saved model + SHAP background to:", OUT_DIR)


if __name__ == "__main__":
    main()
