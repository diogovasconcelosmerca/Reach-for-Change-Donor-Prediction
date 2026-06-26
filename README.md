# Reach for Change: Predicting Donor Response for Social Good

> **Civic Support Alliance (CSA)** project aimed at transitioning from mass-solicitation outreach to a data-driven targeted strategy. By replacing untargeted solicitations with a machine learning model, we optimize outreach efficiency and reduce donor fatigue.

---

## 🎯 Business Problem & Objective

Despite the growing number of charitable causes, repeated and untargeted solicitations have led to **donor fatigue**, reducing engagement and long-term support for humanitarian initiatives.

The primary goal is to develop a **supervised learning classification model** to predict donor response. This enables the CSA to focus outreach efforts on high-probability leads, maximising campaign efficiency while minimising unnecessary contact and donor disengagement.

**Research Question:**
> *"Based on historical and demographic profiles, what is the probability of a specific individual donating if contacted during this campaign?"*

---

## 📊 Dataset & Methodology

### Dataset Overview
- **Records:** 13,560 individuals.
- **Features:** 41 independent variables based on the **RFM Framework** (Recency, Frequency, Monetary).
- **Target:** Binary classification (Donor vs Non-Donor).
- **Imbalance:** 75/25 towards non-donors. 

### Methodology (CRISP-DM)
The project strictly follows the **CRISP-DM** methodology with a rigorous anti-leakage protocol:
1. **Holdout Validation:** Stratified 80/20 train/validation split reserved upfront.
2. **Exploratory Data Analysis (EDA):** In-depth analysis on training data only.
3. **Robust Preprocessing:** Missing value imputation, log-transformations, robust scaling, and categorical encodings (fit exclusively on training data).
4. **Feature Selection:** 3-stage Filter FS (Variance Threshold → Correlation Redundancy → Mutual Information SelectKBest).
5. **Model Screening:** Evaluation of 8 algorithms across Probabilistic, Discriminant, Linear, Tree, and Ensemble families.
6. **Hyperparameter Tuning:** Randomized and Grid Search with 5-fold CV to identify robust parameters and minimize optimization bias.
7. **Threshold Optimization:** F1-score maximization utilizing `TunedThresholdClassifierCV`.

---

## 🚀 Results & Performance

Because accuracy is uninformative under high class imbalance, **F1-score** was used as the primary evaluation metric.

- **Final Deployed Model:** Stacking Ensemble with passthrough and optimized decision threshold.
- **Validation F1-Score:** `0.416` (optimized threshold = `0.415`, yielding a +0.035 improvement over the default 0.5 cut-off).
- **Kaggle Public Score:** `0.423` F1-score, perfectly aligned with validation estimates (no overfitting).

**Conclusion:** Extensive sensitivity analysis proved the task is **data-limited** (max mutual information of 0.014), making the F1 cap a ceiling of the dataset's predictability rather than a model constraint.

---

## 🧠 Skills Demonstrated

- **Machine Learning Architecture:** Stacking Ensembles, Hyperparameter Tuning (Random/Grid Search).
- **Data Preprocessing & Feature Engineering:** RFM Framework, Log-transformations, Robust Scaling.
- **Imbalanced Data Handling:** Threshold Optimization, Stratified K-Fold CV, Class Weighting.
- **Feature Selection:** Variance Thresholds, Correlation Redundancy (Theil's U, Spearman), Mutual Information.
- **Anti-Leakage Protocols:** Strict separation of fit/transform steps without relying solely on Sklearn Pipelines to maintain transparency.

---

## 💻 Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3 |
| **Machine Learning** | scikit-learn (StackingClassifier, TunedThresholdClassifierCV, RF, GBM, etc.), scipy |
| **Data Manipulation** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 📁 Project Structure

```text
Reach_for_Change_DataMining/
├── README.md
├── .gitignore
└── Reach_for_Change_Predicting_Donor_Response.ipynb   # Complete ML Pipeline & Analysis
```

---

## 👥 Authors

**Data Mining II** — MSc Information Management (BI)  
Nova IMS — Universidade Nova de Lisboa (2025/2026)

- [Diogo Merca](https://github.com/diogovasconcelosmerca)
- [Madalina Noje](https://github.com/madalinanoje)
- Alexandre Duarte
- Matilde Cordeiro
