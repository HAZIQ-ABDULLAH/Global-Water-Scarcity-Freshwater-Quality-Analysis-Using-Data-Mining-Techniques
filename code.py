# ============================================================
# STEP 1: ENVIRONMENT SETUP & LIBRARY IMPORTS
# Purpose: Ensure reproducibility and stable execution
# ============================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

plt.style.use("default")
sns.set_context("notebook")


# ============================================================
# STEP 2: DATA LOADING (RAW DATA INGESTION)
# ============================================================

water_quality = pd.read_csv(
    "C:/Users/LAPTOP INSIDE/OneDrive/Desktop/Global-Water-Scarcity-Project/water_potability.csv"
)

water_stress = pd.read_csv(
    "C:/Users/LAPTOP INSIDE/OneDrive/Desktop/Global-Water-Scarcity-Project/Water scarcity.csv",
    skiprows=4
)

metadata_indicator = pd.read_csv(
    "C:/Users/LAPTOP INSIDE/OneDrive/Desktop/Global-Water-Scarcity-Project/Metadata_Indicator.csv"
)


# ============================================================
# STEP 3: DATA UNDERSTANDING & QUALITY CHECK
# ============================================================

print("Water Quality Dataset Shape:", water_quality.shape)
print("Water Stress Dataset Shape:", water_stress.shape)

print("\nMissing Values (Water Quality):")
print(water_quality.isnull().sum())


# ============================================================
# STEP 4: DATA CLEANING & PREPROCESSING
# ============================================================

# Fill missing numerical values with column mean
water_quality.fillna(water_quality.mean(), inplace=True)

# Select latest year for global water stress
water_stress_clean = water_stress[['Country Name', '2022']].copy()
water_stress_clean.rename(
    columns={'2022': 'Water_Stress_Percent'},
    inplace=True
)
water_stress_clean.dropna(inplace=True)


# ============================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    water_quality.corr(),
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Matrix of Water Quality Parameters")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# Water stress distribution
plt.figure(figsize=(6,4))
sns.boxplot(y=water_stress_clean['Water_Stress_Percent'])
plt.title("Global Water Stress Distribution (%)")
plt.ylabel("Water Stress (%)")
plt.savefig("outputs/water_stress_distribution.png")
plt.show()


# ============================================================
# STEP 6: FEATURE SCALING
# ============================================================

X = water_quality.drop('Potability', axis=1)
y = water_quality['Potability']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# STEP 7: CLASSIFICATION MODEL (RANDOM FOREST)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)


# ============================================================
# STEP 8: MODEL EVALUATION & INTERPRETATION
# ============================================================

y_pred = rf_model.predict(X_test)

print("\nClassification Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Water Potability")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# Feature Importance
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(6,4))
feature_importance.plot(kind="bar")
plt.title("Feature Importance for Water Potability")
plt.ylabel("Importance Score")
plt.savefig("outputs/feature_importance.png")
plt.show()


# ============================================================
# STEP 9: ELBOW METHOD FOR CLUSTER SELECTION
# ============================================================

inertia = []

for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(water_stress_clean[['Water_Stress_Percent']])
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2,6), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters")
plt.savefig("outputs/elbow_method.png")
plt.show()


# ============================================================
# STEP 10: K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(n_clusters=3, random_state=42)
water_stress_clean['Cluster'] = kmeans.fit_predict(
    water_stress_clean[['Water_Stress_Percent']]
)

print("Silhouette Score:",
      silhouette_score(
          water_stress_clean[['Water_Stress_Percent']],
          water_stress_clean['Cluster']
      ))


# ============================================================
# STEP 11: CLUSTER INTERPRETATION & VISUALIZATION
# ============================================================

cluster_labels = {
    0: "Low Water Stress",
    1: "Moderate Water Stress",
    2: "High Water Stress"
}

water_stress_clean['Stress_Level'] = (
    water_stress_clean['Cluster'].map(cluster_labels)
)

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=water_stress_clean,
    x='Water_Stress_Percent',
    y='Cluster',
    hue='Stress_Level',
    palette='Set2'
)
plt.title("Country Clusters Based on Water Stress Levels")
plt.xlabel("Water Stress (%)")
plt.ylabel("Cluster Group")
plt.savefig("outputs/water_stress_clusters.png")
plt.show()


# ============================================================
# STEP 12: KEY INSIGHTS & HIGH-RISK COUNTRIES
# ============================================================

top_countries = water_stress_clean.sort_values(
    by='Water_Stress_Percent',
    ascending=False
).head(15)

print("\nTop 15 Water-Stressed Countries:")
print(top_countries[['Country Name', 'Water_Stress_Percent', 'Stress_Level']])

plt.figure(figsize=(7,4))
sns.barplot(
    data=top_countries,
    x='Water_Stress_Percent',
    y='Country Name',
    palette='Reds_r'
)
plt.title("Top 15 Most Water-Stressed Countries (2022)")
plt.xlabel("Water Stress (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("outputs/top_15_water_stressed.png")
plt.show()

# Metadata Explanation
if 'LongDefinition' in metadata_indicator.columns:
    print("\nWorld Bank Indicator Description:")
    print(metadata_indicator.loc[0, 'LongDefinition'])
