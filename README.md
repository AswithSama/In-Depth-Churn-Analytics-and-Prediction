# Customer Churn Prediction

A machine learning project to predict customer churn using the **Customer Churn Records** dataset. The pipeline covers data preprocessing, exploratory data analysis, feature engineering, feature selection, supervised classification, unsupervised clustering, and model evaluation.

---

## Dataset

The dataset contains **10,000 customer records** from a bank, with the following features:

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country of the customer (France, Spain, Germany) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Number of years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card (0/1) |
| `IsActiveMember` | Whether the customer is an active member (0/1) |
| `EstimatedSalary` | Estimated annual salary |
| `Complain` | Whether the customer has filed a complaint (0/1) |
| `Satisfaction Score` | Customer satisfaction score (1–5) |
| `Card Type` | Type of card held (DIAMOND, GOLD, SILVER, PLATINUM) |
| `Point Earned` | Reward points earned |
| `Exited` | **Target variable** — 1 if churned, 0 if retained |

**Target class distribution:**
- Not Exited: ~79.6%
- Exited: ~20.4%

---

## Project Pipeline

### 1. Data Loading & Cleaning

- Loaded dataset using `pandas`.
- Dropped non-informative columns: `RowNumber`, `CustomerId`, `Surname`.

### 2. Exploratory Data Analysis (EDA)

- Plotted histograms for all features to understand distributions.
- Visualized the class imbalance using a **pie chart** of churned vs. retained customers.
- Analyzed each categorical feature against the churn label using **stacked bar charts**.

### 3. Feature Engineering

- **Balance** was binarized into two categories:
  - `Low`: Balance < 50,000
  - `High`: Balance ≥ 50,000
- Categorical features with ≤ 11 unique values were identified for encoding.
- Remaining numerical features were treated as continuous.

### 4. Encoding

- Applied **One-Hot Encoding** to all object-type columns using `pd.get_dummies`.
- Converted the resulting DataFrame to integer type.
- Dropped redundant encoded columns to prevent multicollinearity:
  - `Balance_Low`, `CardType_DIAMOND`, `Geography_Germany`, `Gender_Female`

### 5. Correlation Analysis

- Generated a **correlation heatmap** using `seaborn` to inspect feature relationships and multicollinearity.

### 6. Feature Selection

- Used **SelectKBest** with ANOVA F-value (`f_classif`) to select the **top 9 most informative features**.

### 7. Train/Test Split

- Split data into **80% training** and **20% testing** using `train_test_split` with `random_state=42`.

### 8. Feature Scaling

- Applied **MinMaxScaler** to normalize continuous features independently on train and test sets.

### 9. Model Training

Four classifiers were trained:

| Model | Library |
|---|---|
| Logistic Regression | `sklearn` |
| Random Forest | `sklearn` |
| Support Vector Machine (SVC) | `sklearn` |
| XGBoost | `xgboost` |

---

## Classification Results

### Accuracy & Classification Report

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression | 80.28% | 0.52 | 0.25 | 0.34 |
| Random Forest | 83.35% | 0.61 | 0.50 | 0.55 |
| XGBoost | 84.21% | 0.64 | 0.51 | 0.57 |
| **SVM** | **84.76%** | **0.75** | **0.37** | **0.49** |

### Confusion Matrices

Confusion matrices were generated for all four models on the test set (n=1988):

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---|---|---|---|
| Logistic Regression | 1494 | 93 | 299 | 102 |
| Random Forest | 1456 | 131 | 200 | 201 |
| SVM | 1537 | 50 | 253 | 148 |
| XGBoost | 1470 | 117 | 197 | 204 |

Key observations:
- **SVM** produces the fewest false positives (50), making it the most conservative in flagging churners.
- **XGBoost** and **Random Forest** detect the most actual churners (TP = 204 and 201), offering better recall at the cost of more false positives.
- All models miss a significant portion of churners (high false negatives), reflecting the class imbalance challenge.

### Precision-Recall Curves

Precision-Recall curves were plotted for all models to evaluate performance across classification thresholds — especially relevant given the class imbalance:

| Model | Average Precision (AP) |
|---|---|
| Logistic Regression | 0.45 |
| Random Forest | 0.59 |
| SVM | 0.59 |
| **XGBoost** | **0.64** |

- **XGBoost** achieves the highest AP score (0.64), indicating the best overall precision-recall trade-off.
- Logistic Regression trails significantly (AP = 0.45), confirming it struggles with the non-linear churn boundary.
- Random Forest and SVM are comparable in AP despite differing in raw accuracy.

---

## Clustering Analysis

Unsupervised clustering was applied to explore whether natural groupings in the data align with churn patterns. **KMeans with k=2** was used as the clustering algorithm, with three dimensionality reduction techniques for visualization.

### PCA Visualization

- Applied **Principal Component Analysis (PCA)** to reduce features to 2 components.
- The PCA plot shows two spatially separated clusters largely divided along **Principal Component 1** (likely dominated by the Balance feature due to its large scale).
- The separation is clean but axis-aligned, suggesting PCA captures variance from high-magnitude features rather than subtle churn-related patterns.

### UMAP Visualization

- Applied **UMAP** (Uniform Manifold Approximation and Projection) for non-linear dimensionality reduction.
- The UMAP plot reveals more structured, non-linear cluster boundaries, with two distinct groupings showing tighter intra-cluster cohesion.
- This indicates meaningful local relationships in the data that linear methods like PCA cannot capture.

### t-SNE Visualization

- Applied **t-SNE** for high-dimensional neighborhood preservation.
- The t-SNE plot shows **3 discernible clusters** (purple, teal, yellow), suggesting the data may have more than 2 natural groupings.
- Cluster boundaries are interleaved, reflecting the complexity of churn behavior across overlapping customer profiles.

### Clustering Takeaways

- PCA is dominated by scale effects; UMAP and t-SNE reveal more meaningful structure.
- The 3-cluster structure seen in t-SNE warrants exploring k=3 in future iterations.
- An **Elbow Method** (WCSS/inertia-based) function was implemented to determine the optimal number of clusters programmatically.

---

## Cross-Validation

5-fold cross-validation was applied to Logistic Regression to assess model generalization:

- Features were standardized using **StandardScaler** before cross-validation.
- Scoring metric: **Accuracy**
- Outputs include per-fold scores, mean accuracy, and standard deviation across folds.

This step validates that model performance is not overly sensitive to any specific train/test split.

---

## Key Findings & Recommendations

- **Best accuracy:** SVM (84.76%), but XGBoost offers the best precision-recall balance (AP = 0.64).
- **Class imbalance** is the primary bottleneck — consider applying SMOTE or class weighting in future iterations.
- **Clustering with t-SNE** hints at 3 underlying customer segments; profiling these segments could inform targeted retention strategies.
- **Feature leakage caution:** MinMaxScaler should be fit only on training data and applied (transform only) to the test set to prevent data leakage.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
umap-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost umap-learn
```

---

## File Structure

```
├── Customer-Churn-Records.csv        # Raw dataset
├── churn_analysis.ipynb              # Main notebook
└── README.md
```