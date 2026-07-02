# 🌍 Global Water Scarcity & Freshwater Quality Analysis

Applies classification and clustering to real-world water datasets to predict whether water is safe to drink and to identify which countries face the most severe water stress.

## Problem / Motivation

Freshwater is one of the most stressed resources on the planet, yet the two sides of the problem — *is the water we have safe to drink?* and *which countries are running out of it?* — are rarely analyzed together. This project combines a water-quality dataset and a World Bank water-stress indicator to answer both questions: what physicochemical properties determine whether water is potable, and which countries are under the most severe water stress, using a single, reproducible data mining pipeline.

## Approach

- **Data cleaning & EDA**: missing-value imputation (mean fill) on the water quality dataset, correlation heatmap, and distribution analysis of global water-stress percentages.
- **Feature scaling**: Min-Max normalization applied to all water-quality features before modeling.
- **Classification — Random Forest**: predicts water potability (safe / not safe) from 9 physicochemical parameters (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity), using `class_weight="balanced"` to address class imbalance. Evaluated with accuracy, precision, recall, F1-score, a confusion matrix, and feature-importance ranking.
- **Clustering — K-Means**: groups countries into Low / Moderate / High water-stress categories based on the 2022 World Bank "freshwater withdrawal as % of renewable resources" indicator. Optimal cluster count chosen via the elbow method and validated with a silhouette score. Cluster labels are ranked by centroid value so "High Water Stress" always corresponds to the highest-withdrawal cluster, rather than an arbitrary KMeans label.

## Results
- Random Forest achieved 66% accuracy on the held-out test set.
- Silhouette score of 0.95 confirms well-separated water-stress clusters.

- Random Forest classifier evaluated on a held-out 20% test split (stratified), with accuracy, precision/recall/F1, and a confusion matrix saved to `outputs/confusion_matrix.png`.
- Feature importance ranking (`outputs/feature_importance.png`) highlights which chemical parameters most influence potability predictions.
- K-Means clustering (k=3, chosen via `outputs/elbow_method.png`) groups countries into Low / Moderate / High water-stress bands, visualized in `outputs/water_stress_clusters.png`.
- Top 15 most water-stressed countries for 2022 are ranked in `outputs/top_15_water_stressed.png`.

*(Run `code.py` to regenerate all figures locally — see below.)*

## Tech Stack

- **Language**: Python 3
- **Data handling**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: scikit-learn (`RandomForestClassifier`, `KMeans`, `MinMaxScaler`, `train_test_split`)

## How to Run It

**Option 1 — Google Colab (recommended, zero setup)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open a new Colab notebook.
2. Upload `code.py` (or paste its contents into cells) along with `water_potability.csv`, `Water scarcity.csv`, and `Metadata_Indicator.csv`.
3. Update the three file paths in **Step 2** to point at the uploaded files (e.g. `/content/water_potability.csv`), then run all cells.

**Option 2 — Run locally**

```bash
# 1. Clone the repo
git clone https://github.com/HAZIQ-ABDULLAH/Global-Water-Scarcity-Freshwater-Quality-Analysis-Using-Data-Mining-Techniques.git
cd Global-Water-Scarcity-Freshwater-Quality-Analysis-Using-Data-Mining-Techniques

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python code.py
```


## Data Sources

| Dataset | Source | Used for |
|---|---|---|
| Water Quality (`water_potability.csv`) | [Kaggle](https://www.kaggle.com/) | Potability classification |
| Global Water Stress (`Water scarcity.csv`) | World Bank Open Data, 2022 | Country-level stress clustering |
| Metadata (`Metadata_Indicator.csv`) | World Bank | Indicator definitions |
