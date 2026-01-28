import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG
DATA_PATH = "podcasts_clean_v2.csv"
NLP_DIR = "nlp_outputs_v2"
OUT_DIR = os.path.join(NLP_DIR, "fairness_bias")
os.makedirs(OUT_DIR, exist_ok=True)

TOPK_PATH = os.path.join(NLP_DIR, "topk_indices_text.npy")  # change if needed
K_LIST = [5, 10]
Q = 4  # episode quantiles


def gini(x):
    x = np.sort(np.asarray(x, float))
    s = x.sum()
    if s <= 0:
        return np.nan
    n = len(x)
    return (n + 1 - 2 * np.sum(np.cumsum(x)) / s) / n


def main():
    df = pd.read_csv(DATA_PATH)
    if "primary_category" not in df.columns:
        raise KeyError("primary_category required")

    topk = np.load(TOPK_PATH).astype(int)

    # episode count 
    if "episode_count" in df.columns:
        ep = pd.to_numeric(df["episode_count"], errors="coerce").fillna(0).clip(lower=0)
    elif "episode_count_capped" in df.columns:
        ep = pd.to_numeric(df["episode_count_capped"], errors="coerce").fillna(0).clip(lower=0)
    else:
        ep = pd.Series([0] * len(df))

    # quantile bins (0..Q-1)
    bins = pd.qcut(ep.rank(method="first"), Q, labels=False)

    cats = df["primary_category"].astype(str).values

    summary_rows = []

    for k in K_LIST:
        k = min(k, topk.shape[1])
        rec = topk[:, :k].ravel()
        rec = rec[rec >= 0]  # IMPORTANT: ignore -1

        # Category Exposure
        corpus_share = pd.Series(cats).value_counts(normalize=True)
        rec_share = pd.Series(cats[rec]).value_counts(normalize=True) if rec.size else pd.Series(dtype=float)

        exp = pd.concat(
            [corpus_share.rename("corpus_share"), rec_share.rename("recommendation_share")],
            axis=1
        ).fillna(0.0)
        exp["rec_vs_corpus_ratio"] = exp["recommendation_share"] / exp["corpus_share"].replace(0, np.nan)
        exp.to_csv(os.path.join(OUT_DIR, f"category_exposure_at{k}.csv"), encoding="utf-8")

        # Category Entropy + Gini 
        ent_list, gini_list = [], []
        for i in range(topk.shape[0]):
            rec_i = topk[i, :k]
            rec_i = rec_i[rec_i >= 0]
            if rec_i.size == 0:
                ent_list.append(np.nan)
                gini_list.append(np.nan)
                continue
            vc = pd.Series(cats[rec_i]).value_counts()
            p = (vc / vc.sum()).values
            ent_list.append(float(-(p * np.log2(p)).sum()))
            gini_list.append(float(gini(vc.values)))

        summary_rows.append({"metric": f"CategoryEntropy@{k}", "mean": float(np.nanmean(ent_list)), "std": float(np.nanstd(ent_list))})
        summary_rows.append({"metric": f"CategoryGini@{k}", "mean": float(np.nanmean(gini_list)), "std": float(np.nanstd(gini_list))})

        # Long-tail Exposure
        if rec.size:
            lt = bins.iloc[rec].value_counts(normalize=True).sort_index()
            lt_df = lt.rename("exposure_share").reset_index()
            # robust renaming (first col name varies)
            lt_df = lt_df.rename(columns={lt_df.columns[0]: "episode_quantile"})
            lt_df["episode_quantile"] = lt_df["episode_quantile"].astype(int).apply(lambda x: f"Q{x+1}")
        else:
            lt_df = pd.DataFrame(columns=["episode_quantile", "exposure_share"])

        lt_df.to_csv(os.path.join(OUT_DIR, f"longtail_exposure_at{k}.csv"), index=False, encoding="utf-8")

    # summary
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "fairness_bias_summary.csv"), index=False, encoding="utf-8")

    # simple plots @10
    k = 10
    ce = pd.read_csv(os.path.join(OUT_DIR, f"category_exposure_at{k}.csv"), index_col=0)
    top10 = ce.sort_values("recommendation_share", ascending=False).head(10)

    plt.figure()
    plt.bar(top10.index.astype(str), top10["rec_vs_corpus_ratio"].fillna(0))
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Category Exposure Ratio @{k}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"plot_category_exposure_at{k}.png"), dpi=200)
    plt.close()

    lt10 = pd.read_csv(os.path.join(OUT_DIR, f"longtail_exposure_at{k}.csv"))
    if not lt10.empty:
        plt.figure()
        plt.bar(lt10["episode_quantile"], lt10["exposure_share"])
        plt.title(f"Long-Tail Exposure @{k}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"plot_longtail_exposure_at{k}.png"), dpi=200)
        plt.close()

    print("Done:", OUT_DIR)


if __name__ == "__main__":
    main()
