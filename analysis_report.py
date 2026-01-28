import pandas as pd

CLEAN_CSV = "podcasts_clean.csv"
OUT_XLSX = "podcast_analysis_report.xlsx"

def main():
    df = pd.read_csv(CLEAN_CSV)

  
    # 1) Kategorien aufspalten
  
    df["categories_list"] = df["categories"].fillna("").str.split(",")
    df["categories_list"] = df["categories_list"].apply(
        lambda lst: [c.strip() for c in lst if c.strip()]
    )

    # Exploded Dataset (jede Kategorie als eigener Eintrag)
    cat_exploded = df.explode("categories_list")

    # Kategoriehäufigkeiten
    category_counts = cat_exploded["categories_list"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    # Sprachverteilung
    language_counts = df["language_final"].value_counts().reset_index()
    language_counts.columns = ["language", "count"]

    # Beschreibungslängen
    df["desc_char_len"] = df["description"].astype(str).str.len()
    df["desc_word_count"] = df["description"].astype(str).str.split().str.len()

    desc_stats = df[["desc_char_len", "desc_word_count"]].describe()

    # Episoden-Stats (falls vorhanden)
    if "episode_count" in df.columns:
        episode_stats = df["episode_count"].describe()
    else:
        episode_stats = pd.Series(dtype=float)

 
    # 2) Excel-Report erstellen
  
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Dataset", index=False)
        cat_exploded.to_excel(writer, sheet_name="Categories_Exploded", index=False)
        category_counts.to_excel(writer, sheet_name="Category_Counts", index=False)
        language_counts.to_excel(writer, sheet_name="Languages", index=False)
        desc_stats.to_excel(writer, sheet_name="Description_Stats")
        episode_stats.to_excel(writer, sheet_name="Episode_Stats")

    print(f"Excel-Report gespeichert als: {OUT_XLSX}")
    

if __name__ == "__main__":
    main()
