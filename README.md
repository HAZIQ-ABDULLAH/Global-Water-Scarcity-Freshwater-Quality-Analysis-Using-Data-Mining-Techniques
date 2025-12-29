🌍 Global Water Scarcity & Freshwater Quality Analysis
Using Data Mining & Machine Learning Techniques
📌 Project Overview

This project analyzes global freshwater quality and water stress levels using real-world datasets from Kaggle and the World Bank.
The objective is to apply data mining techniques to:

Predict water potability (safe vs unsafe)

Identify countries facing severe water stress

Extract actionable insights for environmental and policy awareness

The project follows a complete data science pipeline, making it suitable for academic evaluation and professional portfolios.

🎯 Objectives

Analyze physicochemical parameters affecting water potability

Build a classification model to predict potable water

Cluster countries based on water stress severity

Visualize patterns and extract meaningful insights

📂 Datasets Used
1. Water Quality Dataset (Kaggle)

Contains chemical and physical parameters such as:

pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, TOC, Turbidity

Target variable: Potability (0 = Not Safe, 1 = Safe)

2. Global Water Stress Dataset (World Bank)

Indicator: Freshwater withdrawal as % of renewable resources

Year used: 2022

Source: World Bank Open Data

3. Metadata

Official indicator definitions from World Bank for interpretation

🛠️ Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Random Forest Classifier

K-Means Clustering

🔄 Project Workflow

Environment setup & reproducibility

Data loading from multiple sources

Data cleaning & missing value handling

Exploratory Data Analysis (EDA)

Feature scaling (Min-Max Normalization)

Classification using Random Forest

Model evaluation (Accuracy, Precision, Recall, F1-score)

Feature importance analysis

Elbow method for cluster selection

Clustering countries using K-Means

Cluster interpretation & visualization

Extraction of high-risk countries

🤖 Machine Learning Models
🔹 Random Forest Classifier

Purpose: Predict water potability

Handles non-linearity and feature importance well

Class imbalance handled using class_weight="balanced"

🔹 K-Means Clustering

Purpose: Group countries by water stress severity

Clusters:

Low Water Stress

Moderate Water Stress

High Water Stress

Evaluated using Silhouette Score

📊 Key Results & Insights

Certain chemical parameters have strong influence on water safety

Multiple countries show extreme water stress, signaling urgent concern

Clustering helps classify countries into risk categories

Data-driven evidence supports global water sustainability challenges
