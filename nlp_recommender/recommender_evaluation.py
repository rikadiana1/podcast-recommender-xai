import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONFIGURATION

DATA_PATH = "podcasts_clean_v2.csv"
NLP_DIR = "nlp_outputs_v2"
EVAL_DIR = os.path.join(NLP_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

K_LIST = [5, 10]

# REQUIRED TF-IDF ARTIFACTS
SIM_TFIDF_PATH = os.path.join(NLP_DIR, "similarity_tfidf.npy")
TOPK_TEXT_IDX_PATH = os.path.join(NLP_DIR, "topk_indices_text.npy")
TOPK_TEXT_SCO_PATH = os.path.join(NLP_DIR, "topk_scores_text.npy")

# OPTIONAL SBERT ARTIFACTS
SIM_SBERT_PATH = os.path.join(NLP_DIR, "similarity_embeddings.npy")
TOPK_SBERT_IDX_PATH = os.path.join(NLP_DIR, "topk_indices_embeddings.npy")
TOPK_SBERT_SCO_PATH = os.path.join(NLP_DIR, "topk_scores_embeddings.npy")

# IMPORTANT:

MIN_VALID_SIM = 0.0  


# VALIDITY MASK

def valid_mask(topk_idx: np.ndarray, topk_scores: np.ndarray, k: int, min_score: float = 0.0):
    k = min(k, topk_idx.shape[1])
    idx_k = topk_idx[:, :k]
    sco_k = topk_scores[:, :k]
    mask = (idx_k >= 0) & (sco_k >= min_score)
    return idx_k, sco_k, mask



# METRICS

def coherence_from_topk(topk_idx: np.ndarray, topk_scores: np.ndarray, k: int, min_score: float = 0.0):
    idx_k, sco_k, mask = valid_mask(topk_idx, topk_scores, k, min_score)
    sco = np.where(mask, sco_k, np.nan)

    mean_k = np.nanmean(sco, axis=1)
    max_k = np.nanmax(sco, axis=1)
    return mean_k, max_k


def similarity_distribution(topk_idx: np.ndarray, topk_scores: np.ndarray, k: int, min_score: float = 0.0):
    idx_k, sco_k, mask = valid_mask(topk_idx, topk_scores, k, min_score)
    return sco_k[mask]


def valid_rate_at_k(topk_idx: np.ndarray, topk_scores: np.ndarray, k: int, min_score: float = 0.0):
    idx_k, sco_k, mask = valid_mask(topk_idx, topk_scores, k, min_score)
    has_any = mask.any(axis=1)
    return float(has_any.mean()), int(has_any.sum())


def category_consistency_at_k(
    df: pd.DataFrame,
    topk_idx: np.ndarray,
    topk_scores: np.ndarray,
    k: int,
    cat_col: str = "primary_category",
    min_score: float = 0.0,
):
    idx_k, sco_k, mask = valid_mask(topk_idx, topk_scores, k, min_score)

    seed_cat = df[cat_col].astype(str).values
    rec_cat_all = df[cat_col].astype(str).values

    out = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df)):
        valid = idx_k[i][mask[i]]
        if valid.size == 0:
            continue
        rec_cats = rec_cat_all[valid]
        out[i] = float((rec_cats == seed_cat[i]).mean())

    return out


def coverage_at_k(topk_idx: np.ndarray, topk_scores: np.ndarray, n_items: int, k: int, min_score: float = 0.0):
    idx_k, sco_k, mask = valid_mask(topk_idx, topk_scores, k, min_score)
    valid_items = idx_k[mask]

    if valid_items.size == 0:
        return 0.0, 0

    unique_items = np.unique(valid_items)
    return float(len(unique_items) / n_items), int(len(unique_items))


def intra_list_diversity(sim_matrix: np.ndarray, topk_idx: np.ndarray, topk_scores: np.ndarray, k: int, min_score: float = 0.0):
    """
    ILD per seed:
      ILD = 1 - mean pairwise similarity among valid items in recommendation list
    """
    idx_k, sco_k, mask = valid_mask(topk_idx, topk_scores, k, min_score)
    n = idx_k.shape[0]
    ild = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        recs = idx_k[i][mask[i]]
        if recs.size < 2:
            continue

        S = sim_matrix[np.ix_(recs, recs)].astype(float)
        np.fill_diagonal(S, np.nan)
        mean_pair_sim = np.nanmean(S)
        ild[i] = 1.0 - float(mean_pair_sim)

    return ild


def summarize_metric(name: str, per_seed_values: np.ndarray):
    v = np.asarray(per_seed_values, dtype=float)

    # If everything is NaN (e.g., no valid recs), return NaNs
    if np.all(np.isnan(v)):
        return {"metric": name, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    return {
        "metric": name,
        "mean": float(np.nanmean(v)),
        "median": float(np.nanmedian(v)),
        "std": float(np.nanstd(v)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
    }



# PLOTTING HELPERS

def save_hist(values, title, outpath, bins=60):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return

    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_bar(names, vals, title, outpath):
    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# EVALUATION CORE

def evaluate_variant(df, variant_name, sim_matrix, topk_idx, topk_scores):
    n_items = len(df)
    rows = []

    for k in K_LIST:
        # 0) Valid-rate @k (how many seeds have at least 1 valid neighbor)
        valid_frac, valid_count = valid_rate_at_k(topk_idx, topk_scores, k, min_score=MIN_VALID_SIM)
        rows.append({
            "metric": f"{variant_name} valid_rate@{k}",
            "mean": valid_frac, "median": valid_frac, "std": 0.0, "min": valid_frac, "max": valid_frac
        })
        rows.append({
            "metric": f"{variant_name} valid_count@{k}",
            "mean": float(valid_count), "median": float(valid_count), "std": 0.0,
            "min": float(valid_count), "max": float(valid_count)
        })

        # 1) Coherence
        mean_k, max_k = coherence_from_topk(topk_idx, topk_scores, k, min_score=MIN_VALID_SIM)
        rows.append(summarize_metric(f"{variant_name} mean@{k}", mean_k))
        rows.append(summarize_metric(f"{variant_name} max@{k}", max_k))

        # 2) Similarity distribution
        dist = similarity_distribution(topk_idx, topk_scores, k, min_score=MIN_VALID_SIM)
        save_hist(
            dist,
            title=f"{variant_name}: Similarity Distribution Top-{k} (valid only)",
            outpath=os.path.join(EVAL_DIR, f"{variant_name}_sim_distribution_top{k}.png"),
            bins=60
        )

        # 3) Category consistency
        cat_cons = category_consistency_at_k(df, topk_idx, topk_scores, k, min_score=MIN_VALID_SIM)
        rows.append(summarize_metric(f"{variant_name} category_consistency@{k}", cat_cons))

        # 4) Coverage
        cov_frac, cov_count = coverage_at_k(topk_idx, topk_scores, n_items, k, min_score=MIN_VALID_SIM)
        rows.append({
            "metric": f"{variant_name} coverage@{k}",
            "mean": cov_frac, "median": cov_frac, "std": 0.0, "min": cov_frac, "max": cov_frac
        })
        rows.append({
            "metric": f"{variant_name} coverage_count@{k}",
            "mean": float(cov_count), "median": float(cov_count), "std": 0.0,
            "min": float(cov_count), "max": float(cov_count)
        })

        # 5) ILD
        ild = intra_list_diversity(sim_matrix, topk_idx, topk_scores, k, min_score=MIN_VALID_SIM)
        rows.append(summarize_metric(f"{variant_name} ILD@{k}", ild))

    return pd.DataFrame(rows)



# MAIN

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    if "primary_category" not in df.columns:
        raise KeyError("Dataset needs 'primary_category' for category-consistency evaluation.")

    # Load REQUIRED TF-IDF artifacts
    if not (os.path.isfile(SIM_TFIDF_PATH) and os.path.isfile(TOPK_TEXT_IDX_PATH) and os.path.isfile(TOPK_TEXT_SCO_PATH)):
        raise FileNotFoundError("Missing required TF-IDF artifacts in nlp_outputs_v2.")

    sim_tfidf = np.load(SIM_TFIDF_PATH)
    topk_text_idx = np.load(TOPK_TEXT_IDX_PATH)
    topk_text_sco = np.load(TOPK_TEXT_SCO_PATH)

    # Evaluate TFIDF_TEXT
    print("Evaluating: TFIDF_TEXT")
    res_text = evaluate_variant(
        df=df,
        variant_name="TFIDF_TEXT",
        sim_matrix=sim_tfidf,
        topk_idx=topk_text_idx,
        topk_scores=topk_text_sco
    )

    results = [res_text]
    variants = ["TFIDF_TEXT"]

    # Optional SBERT evaluation 
    has_sbert = (
        os.path.isfile(SIM_SBERT_PATH)
        and os.path.isfile(TOPK_SBERT_IDX_PATH)
        and os.path.isfile(TOPK_SBERT_SCO_PATH)
    )

    if has_sbert:
        print("Evaluating: SBERT")
        sim_sbert = np.load(SIM_SBERT_PATH)
        topk_sbert_idx = np.load(TOPK_SBERT_IDX_PATH)
        topk_sbert_sco = np.load(TOPK_SBERT_SCO_PATH)

        res_sbert = evaluate_variant(
            df=df,
            variant_name="SBERT",
            sim_matrix=sim_sbert,
            topk_idx=topk_sbert_idx,
            topk_scores=topk_sbert_sco
        )
        results.append(res_sbert)
        variants.append("SBERT")
    else:
        print("SBERT artifacts not found -> skipping SBERT (expected behavior).")

    # Save summary
    all_res = pd.concat(results, ignore_index=True)
    out_csv = os.path.join(EVAL_DIR, "offline_evaluation_summary.csv")
    all_res.to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ Saved summary:", out_csv)

    # Comparison plots @10 (if present)
    def get_mean(metric_name):
        m = all_res.loc[all_res["metric"] == metric_name, "mean"]
        return float(m.iloc[0]) if len(m) else np.nan

    mean10 = [get_mean(f"{v} mean@10") for v in variants]
    ild10 = [get_mean(f"{v} ILD@10") for v in variants]
    cat10 = [get_mean(f"{v} category_consistency@10") for v in variants]
    cov10 = [get_mean(f"{v} coverage@10") for v in variants]
    val10 = [get_mean(f"{v} valid_rate@10") for v in variants]

    save_bar(variants, mean10, "Mean Similarity @10 (Coherence, valid only)", os.path.join(EVAL_DIR, "compare_mean_at10.png"))
    save_bar(variants, ild10, "Intra-List Diversity @10 (valid only)", os.path.join(EVAL_DIR, "compare_ild_at10.png"))
    save_bar(variants, cat10, "Category Consistency @10 (valid only)", os.path.join(EVAL_DIR, "compare_category_at10.png"))
    save_bar(variants, cov10, "Coverage @10 (valid only)", os.path.join(EVAL_DIR, "compare_coverage_at10.png"))
    save_bar(variants, val10, "Valid-rate @10 (share of seeds with >=1 valid rec)", os.path.join(EVAL_DIR, "compare_valid_rate_at10.png"))

    print("Saved plots to:", EVAL_DIR)


if __name__ == "__main__":
    main()
