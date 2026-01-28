# Podcast Recommender with XAI (TF-IDF / SBERT / SHAP / Fairness)

This repository contains an end-to-end pipeline for building an NLP-based podcast recommender system.
It includes data collection (PodcastIndex API), preprocessing, similarity-based recommendation (TF-IDF + optional SBERT),
offline evaluation, fairness/bias analysis, and explainability using SHAP.

## Project Structure

Data scraping scripts:
- `scrape_podcasts.py` – Download trending podcasts via PodcastIndex API (requires `.env`)
- `enrich_podcasts_meta.py` – Enrich feeds with metadata (episode_count, last_update_time, explicit/dead), API + RSS fallback
- `clean_podcasts.py` – Clean/filter dataset and create NLP-ready text fields
- `clean_podcasts_v2.py` – Remove “template/series-like” duplicates using similarity thresholding

  Data analysing scripts: 
- `eda_podcasts.py` – Exploratory analysis & plots
- `analysis_report.py` – Excel report generation

Core recommender, explainability & analysis:
- `nlp_recommender.py` – Builds TF-IDF (optional SVD/LSA) + SBERT embeddings + KNN top-k neighbors; stores artifacts
- `recommender_evaluation.py` – Offline evaluation (coherence, coverage, diversity, category consistency, valid-rate)
- `fairness-bias.py` – Fairness/bias analysis (category exposure, entropy, gini, long-tail exposure)
- `train_relevance_logreg.py` – Weakly-supervised ranking model (LogReg) trained on top-k pseudo labels
- `shap_explainer.py` – SHAP explanations for the hybrid ranking model

Data & outputs:
- `data/` – CSVs, reports, generated outputs 
- `nlp_outputs/`, `nlp_outputs_v2/` – generated model artifacts, similarity matrices, plots, evaluation outputs

## Setup

Create environment & install dependencies

pip install -r requirements.txt
