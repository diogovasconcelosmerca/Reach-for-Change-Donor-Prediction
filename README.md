# Reach for Change: Predicting Donor Response for Social Good

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview

**Civic Support Alliance (CSA)** project aimed at transitioning from mass-solicitation outreach to a data-driven targeted strategy. By replacing untargeted solicitations with a robust machine learning model, we optimize outreach efficiency and fundamentally reduce donor fatigue.

> **Academic Context:** Data Mining II, Master's in Data Science & Advanced Analytics (BI) — Nova IMS (2025/2026)

---

## 🎯 Business Problem & Objective

Despite the growing number of charitable causes, repeated and untargeted solicitations have led to **donor fatigue**, reducing engagement and long-term support for humanitarian initiatives. 

The primary goal is to develop a **supervised learning classification model** to predict donor response. This enables the CSA to focus outreach efforts on high-probability leads, maximising campaign efficiency while minimising unnecessary contact and donor disengagement.

**Research Question:**
> *"Based on historical and demographic profiles, what is the probability of a specific individual donating if contacted during this campaign?"*

---

## 🚀 Key Results & Performance

Because accuracy is uninformative under high class imbalance (a majority predictor achieves ~75% accuracy by identifying 0 donors), **F1-score** was used as the primary evaluation metric.

| Model / Metric | Outcome |
|:-------:|---------|
| **Final Deployed Model** | Stacking Ensemble with passthrough and optimized decision threshold |
| **Validation F1-Score** | `0.416` (optimized threshold = `0.415`, yielding a +0.035 improvement over the default cut-off) |
| **Kaggle Public Score** | `0.423` F1-score (perfectly aligned with validation, zero overfitting) |

> **Analytic Conclusion:** Extensive sensitivity analysis proved the task is **data-limited** (max mutual information of 0.014), meaning the achieved F1 score reflects the dataset's inherent predictability ceiling rather than a model constraint.

---

## 🛠️ Methodology

The project strictly follows the **CRISP-DM** methodology with a rigorous anti-leakage protocol:

```text
Raw Data (13,560 customers, 41 features)
    │
    ▼
Holdout Validation ──── Stratified 80/20 train/validation split (anti-leakage foundation)
    │
    ▼
Exploratory Data Analysis ──── Target signal, Missing patterns, Feature redundancies
    │
    ▼
White-Box Preprocessing ──── Imputation, Log-transforms, Robust Scaling, Encoding
    │
    ▼
Feature Selection ──── Variance Threshold ➔ Correlation Redundancy ➔ Mutual Information
    │
    ▼
Model Screening ──── 8 Algorithms evaluated (Probabilistic, Linear, Tree, Ensemble)
    │
    ▼
Hyperparameter Tuning ──── Random & Grid Search with robust selection + Threshold Optimization
    │
    ▼
Final Model ──── Stacking Ensemble (F1 = 0.416, optimized cut-off at 0.415)
```

---

## 🔍 The "White-Box" Preprocessing Approach

To guarantee total transparency and absolute control over data leakage, this project deliberately avoids the use of `sklearn.pipeline.Pipeline` wrappers. Instead, we implemented a rigorous **White-Box Preprocessing Approach**:

- **Explicit Transformations:** Every step (missing value imputation, log-transformations, robust scaling, and categorical encodings) is explicitly fitted on the training set and subsequently applied to the validation set.
- **Intermediate State Inspection:** This transparency allows us to inspect intermediate states (e.g., imputer medians, encoder vocabularies, variance matrices) that are typically hidden inside a black-box Pipeline.
- **Manual Anti-Leakage:** By manually enforcing the strict separation of `fit` and `transform` steps, we demonstrate a deeper, fundamental understanding of data science mechanics and prevent implicit data leakage, ensuring maximum model integrity.

---

## 🧠 Skills Demonstrated

- **Machine Learning Architecture** — Stacking Ensembles, Hyperparameter Tuning (Random/Grid Search).
- **Data Preprocessing & Feature Engineering** — RFM Framework, Log-transformations, Robust Scaling.
- **Imbalanced Data Handling** — Threshold Optimization (`TunedThresholdClassifierCV`), Stratified K-Fold CV, Class Weighting.
- **Feature Selection** — Variance Thresholds, Correlation Redundancy (Theil's U, Spearman), Mutual Information.
- **Anti-Leakage Protocols** — White-box manual preprocessing avoiding black-box Pipeline dependencies.

---

## 💻 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Machine Learning** | scikit-learn (StackingClassifier, TunedThresholdClassifierCV, RF, GBM, etc.), scipy |
| **Data Manipulation** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/diogovasconcelosmerca/Reach-for-Change-Donor-Prediction.git
cd Reach-for-Change-Donor-Prediction

# Open the Jupyter Notebook
jupyter notebook Reach_for_Change_Predicting_Donor_Response.ipynb
```

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
