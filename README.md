# 🔮 Python / Machine Learning | Customer Churn Prediction & Segmentation for an E-commerce Company

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Tool-Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

<p align="center">
  <img src="Churn_Prediction/Banner.png" width="100%">
</p>

_Predict which customers are likely to leave, and group them into segments so the company can run the right promotions for each group._

- **Business Question:** Who will churn next, and what kind of customer are they?
- **Domain:** E-commerce
- **Tools:** Python (pandas, scikit-learn, seaborn, matplotlib)

Author: Bạch Minh Nam

---

## 📑 Table of Contents
1. [📌 Background & Overview](#-background--overview)
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)
3. [⚒️ Main Process](#️-main-process)
4. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)
5. [🗂️ Project Structure](#️-project-structure)
6. [🚀 How to Run This Project](#-how-to-run-this-project)

---

## 📌 Background & Overview

### 🎯 Objective

An e-commerce company wants to reduce the number of customers who stop using their platform (churned users). To do this, the team needs to:

✔️ Understand how churned customers behave, so the team knows what warning signs to look for.

✔️ Build a machine learning model to predict which customers are likely to churn.

✔️ Segment churned customers into smaller groups, so the marketing team can design targeted promotions for each group instead of a one-size-fits-all campaign.

### 👤 Who is this project for?

✔️ Marketing and CRM teams, to design better retention programs.

✔️ Data analysts, to understand churn behavior patterns.

✔️ Business decision-makers, to prioritize which customer groups to act on first.

---

## 📂 Dataset Description & Data Structure

### 📌 Data Source
- 📁 Dataset: `churn_prediction`
- 📄 Format: `.xlsx`

### 📊 Data Structure

<details>
<summary>One main table containing customer behavior and account information - Click to view full table schema</summary>
| Column Name | Data Type | Description |
|---|---|---|
| `CustomerID` | INT | Unique ID for each customer |
| `Churn` | INT | 1 = churned, 0 = stayed |
| `Tenure` | FLOAT | How long the customer has been with the company |
| `PreferredLoginDevice` | TEXT | Device the customer usually logs in with |
| `CityTier` | INT | City tier (1, 2, or 3) |
| `WarehouseToHome` | FLOAT | Distance from warehouse to customer's home |
| `PreferPayment` | TEXT | Preferred payment method |
| `Gender` | TEXT | Customer gender |
| `HourSpendOnApp` | FLOAT | Hours spent on the app or website |
| `NumberOfDeviceRegistered` | INT | Number of devices registered |
| `PreferedOrderCat` | TEXT | Preferred product category in the last month |
| `SatisfactionScore` | INT | Customer satisfaction score |
| `MaritalStatus` | TEXT | Marital status |
| `NumberOfAddress` | INT | Number of saved addresses |
| `Complain` | INT | 1 = filed a complaint in the last month |
 
</details>

---

## ⚒️ Main Process

### 🔍 Part A - EDA (Exploratory Data Analysis)

#### Step 1: Data Cleaning

Before any analysis, the data needs to be clean and complete.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('/content/churn_prediction.xlsx')

# Overview of the data and missing values
print(df.head())
print(df.info())
print(df.describe(include='all'))
```

```python
# Missing data
df.isnull().sum()

# Duplicate rows
df.duplicated().sum()
```

**Missing values:** filled with the median for numeric columns, and the mode (most common value) for text columns.

```python
# Median for numeric columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Mode for categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

**Duplicate rows:** removed completely.

```python
df.drop_duplicates(inplace=True)
```

**Outliers:** detected using the IQR method (see Step 4) and removed for `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderCount`, and other numeric columns. This step prevents extreme values from distorting the model.

#### Step 2: Univariate Analysis

Looking at each column individually to understand the overall data shape.

```python
# Churn ratio
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Churned (1) vs Stayed (0)')
plt.show()

# Tenure distribution
sns.histplot(df['Tenure'], bins=30, kde=True)
plt.title('Tenure distribution')
plt.show()
```

Key findings:
- The dataset is imbalanced: customers who stayed (`Churn` = 0) are about 5 times more than those who churned (`Churn` = 1).
- Customers tend to churn early in their time with the platform: `Tenure` distribution is right-skewed, meaning new customers leave more often.

#### Step 3: Bivariate Analysis - Churn Behavior by Group

This step compares churned vs. non-churned customers across different dimensions to find which groups are most at risk.

```python
sns.countplot(x='Complain', hue='Churn', data=df)
plt.title('Complaint vs Churn')
plt.show()
```

```python
plt.figure(figsize=(10,5))
sns.kdeplot(df[df['Churn']==1]['Tenure'], label='Churned', shade=True)
sns.kdeplot(df[df['Churn']==0]['Tenure'], label='Stayed', shade=True)
plt.title('Tenure distribution: churned vs stayed')
plt.show()
```

```python
# Compare churn rate across customer groups
cat_features_to_check = [
    'CityTier', 'PreferedOrderCat', 'PreferredLoginDevice',
    'MaritalStatus', 'PreferPayment', 'Gender'
]
cat_features_to_check = [c for c in cat_features_to_check if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features_to_check):
    sns.countplot(x=col, hue='Churn', data=df, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Churn rate by {col}')
    axes[i].tick_params(axis='x', rotation=30)

for j in range(len(cat_features_to_check), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Print churn rate (%) per group
for col in cat_features_to_check:
    print('--- Churn rate (%) by ' + col + ' ---')
    churn_rate = df.groupby(col)['Churn'].mean() * 100
    print(churn_rate.round(2))
    print()
```

Key observations:

- **`Complain`:** Customers who filed complaints churned at a much higher rate than those who didn't. This is the clearest behavioral signal for churn.
- **`Tenure`:** Churned customers tend to have shorter time with the company. Customers in their first few months are the most likely to leave.
- **`CityTier`:** Customers in Tier 1 and Tier 3 cities churn more than Tier 2. This may be due to differences in delivery quality or service coverage.
- **`PreferedOrderCat`:** Customers who often buy Mobile Phones and Laptops & Accessories show higher churn; these are high-value categories where a bad experience likely has a bigger impact.
- **`MaritalStatus`:** Single customers churn more than married or divorced ones, possibly because they have less long-term commitment to a platform.
- **`Gender`:** Male customers show slightly higher churn in both count and rate.

#### Step 4: Outlier Treatment

Outliers were removed using the IQR (Interquartile Range) method on 8 numeric columns. Keeping outliers would skew both the EDA findings and the machine learning model's performance.

```python
cols_to_fix = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear',
               'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

for col in cols_to_fix:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Outliers cleaned for column: {col}")
```

#### Step 5: Feature Importance with Random Forest (Base Model)

Before building the full model, a basic Random Forest was trained to see which features matter most for predicting churn. This helps confirm that the analysis in Step 3 is on the right track.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables into numbers
le = LabelEncoder()
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df[col])

# Train base model
X = df_encoded.drop(['Churn', 'CustomerID'], axis=1)
y = df_encoded['Churn']
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Visualize feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
importances.plot(kind='bar')
plt.title('Top Features')
plt.show()
```

---

### 🤖 Part B - Supervised Learning (Churn Prediction)

The goal here is to build a model that can predict whether a customer will churn or not.

#### Step 1: Train-Test Split

The dataset was split into 80% for training and 20% for testing. `stratify=y` was used to make sure the churn ratio is the same in both sets, important because the data is imbalanced.

```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop(['Churn', 'CustomerID'], axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
```

#### Step 2: Train Random Forest Model

A Random Forest Classifier was chosen because it handles mixed data types well, is not easily affected by noise, and provides feature importance scores.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# class_weight='balanced' tells the model to pay more attention
# to the minority class (churned customers)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]  # probability, used for AUC
```

Settings used:
- `n_estimators=200`: 200 decision trees
- `max_depth=12`: limits tree depth to avoid overfitting
- `class_weight='balanced'`: pays more attention to the minority class (churned customers), since there are far fewer of them

#### Step 3: Model Evaluation

```python
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

The model was evaluated using three metrics:
- **Classification Report:** precision, recall, and F1-score for both churn and non-churn groups.
- **ROC-AUC Score:** measures how well the model separates churned from non-churned. Closer to 1.0 is better.
- **Confusion Matrix:** shows how many churned customers the model correctly caught vs. missed.

#### Step 4: Hyperparameter Tuning

`GridSearchCV` was used to test different combinations of settings:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='balanced_accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
```

```python
# Feature importance for the tuned model
best_clf = grid_search.best_estimator_
feats = {feature: importance for feature, importance in zip(X.columns, best_clf.feature_importances_)}

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True).reset_index()

plt.figure(figsize=(10, 8))
plt.barh(importances['index'], importances['Gini-importance'], color='skyblue')
plt.title('Feature Importance (Tuned Model)')
plt.show()
```

| Parameter | Values Tested |
|---|---|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5 |
| `bootstrap` | True, False |

The best settings found were then applied to produce the final tuned model.

---

### 🔵 Part C - Unsupervised Learning (Customer Segmentation)

Once churned customers are identified, the next question is: are all churned customers the same? If not, the company can offer different promotions to different groups.

Only churned customers (`Churn` = 1) were used in this part.

#### Step 1: Find the Right Number of Clusters

Using all numeric columns at once did not produce clear clusters: the Elbow Method showed no obvious "bend" and Silhouette Scores were too low. The solution was to test 4 different feature sets and compare results.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Only keep churned customers
churn_df = df_encoded[df_encoded['Churn'] == 1].copy()

# Feature sets to test:
#   Set1: Behavioral & value (tenure, spend, satisfaction, app usage...)
#   Set2: Delivery & service experience (warehouse distance, complaints...)
#   Set3: All numeric columns (baseline, for comparison)
#   Set4: Same as Set1 but without SatisfactionScore
feature_sets = {
    'Set1_Behavioral_Value': [
        'Tenure', 'CashbackAmount', 'SatisfactionScore',
        'Complain', 'OrderCount', 'DaySinceLastOrder', 'HourSpendOnApp'
    ],
    'Set2_Logistics_Service': [
        'WarehouseToHome', 'Complain', 'SatisfactionScore',
        'OrderAmountHikeFromlastYear', 'CouponUsed', 'DaySinceLastOrder'
    ],
    'Set3_All_Numeric_Baseline': [
        col for col in churn_df.drop(['Churn', 'CustomerID'], axis=1).columns
    ],
    'Set4_Behavioral_NoSatisfaction': [
        'Tenure', 'CashbackAmount', 'Complain',
        'OrderCount', 'DaySinceLastOrder', 'HourSpendOnApp'
    ],
}

for name in feature_sets:
    feature_sets[name] = [c for c in feature_sets[name] if c in churn_df.columns]

K_RANGE = range(2, 9)  # K=1 has no meaning for Silhouette Score
results = {}

for set_name, cols in feature_sets.items():
    print('================= ' + set_name + ' =================')
    print('Columns used:', cols)

    X_subset = churn_df[cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    wcss = []
    sil_scores = []

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    results[set_name] = {
        'K_RANGE': list(K_RANGE),
        'wcss': wcss,
        'silhouette': sil_scores,
        'scaled_features': X_scaled,
        'columns': cols,
    }

    print('K | Silhouette Score | WCSS')
    for k, s, w in zip(K_RANGE, sil_scores, wcss):
        print(f'{k} | {s:.4f}            | {w:.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(K_RANGE), wcss, marker='o', linestyle='--')
    axes[0].set_title('Elbow Method - ' + set_name)
    axes[0].set_xlabel('Number of clusters (K)')
    axes[0].set_ylabel('WCSS')

    axes[1].plot(list(K_RANGE), sil_scores, marker='o', linestyle='--', color='green')
    axes[1].set_title('Silhouette Score - ' + set_name)
    axes[1].set_xlabel('Number of clusters (K)')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()
    print()
```

| Feature Set | Description | Result |
|---|---|---|
| Set1 - Behavioral & Value | `Tenure`, `Cashback`, `SatisfactionScore`, `Complain`, `OrderCount`, `DaySinceLastOrder`, `HourSpendOnApp` | Best: clear peak at K=6, acceptable at K=3 |
| Set2 - Logistics & Service | `WarehouseToHome`, `Complain`, `SatisfactionScore`, `CouponUsed`, `DaySinceLastOrder` | OK but weaker than Set1 |
| Set3 - All Numeric | All numeric columns combined | Silhouette too low (0.07-0.11) |
| Set4 - Behavioral (no Satisfaction) | Same as Set1 but without `SatisfactionScore` | Score keeps rising with K, no clear peak |

Final choice: Set1 with K = 3. K=3 was chosen over K=6 because it gives 3 distinct, meaningful groups that are practical for the marketing team to act on.

#### Step 2: Run Final Clustering & Label Each Group

```python
CHOSEN_SET = 'Set1_Behavioral_Value'
CHOSEN_K = 3

final_scaled = results[CHOSEN_SET]['scaled_features']

kmeans_final = KMeans(n_clusters=CHOSEN_K, init='k-means++', random_state=42, n_init=10)
churn_df['Cluster'] = kmeans_final.fit_predict(final_scaled)

print('Clustering done with feature set:', CHOSEN_SET, ', K =', CHOSEN_K)
print('Columns used:', results[CHOSEN_SET]['columns'])
print(churn_df['Cluster'].value_counts().sort_index())
```

KMeans was run with K=3 on Set1 features (after `StandardScaler` normalization). Each churned customer was assigned to one of three clusters.

#### Step 3: Cluster Profiling - Who is in Each Group?

```python
profile_cols = [
    'Tenure', 'CashbackAmount', 'WarehouseToHome',
    'Complain', 'SatisfactionScore', 'OrderCount',
    'DaySinceLastOrder', 'HourSpendOnApp'
]
profile_cols = [c for c in profile_cols if c in churn_df.columns]

cluster_summary = churn_df.groupby('Cluster')[profile_cols].mean()
print('Average feature values per cluster:')
display(cluster_summary)

print()
print('Customer count per cluster:')
print(churn_df['Cluster'].value_counts().sort_index())
```

| Metric | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Count | 247 customers | 220 customers | 281 customers |
| `Tenure` (avg) | 3.51 | Medium | 3.99 (highest) |
| `CashbackAmount` (avg) | 163.8 (highest) | Medium | 131.5 (lowest) |
| `Complain` rate | ~98% | 0% | ~55% |
| `SatisfactionScore` | Medium | 3.79 (highest) | Medium |
| `DaySinceLastOrder` | Medium | 3.03 (highest, long gap) | Medium |
| `OrderCount` | Medium | Medium | 1.06 (lowest) |

🔴 **Cluster 0 - "Long-term, high-value customers with complaints"**
These customers stayed the longest and got the most cashback, but nearly all of them filed a complaint. They also live farthest from the warehouse. They are leaving because of service and delivery problems, not because they don't like the platform.

🟡 **Cluster 1 - "Happy but quietly drifting away"**
No complaints, highest satisfaction, but the longest gap since their last order. They are not angry, they are just becoming less active over time and eventually stopping.

🟢 **Cluster 2 - "Loyal but under-rewarded"**
Highest `Tenure` (longest time with the company), but lowest `CashbackAmount` and lowest `OrderCount`. They have been around the longest but seem to feel they are not getting enough value back for their loyalty.

---

## 🔎 Final Conclusion & Recommendations

🔴 **Cluster 0 - "Long-term, high-value customers with complaints"**

✔️ Reach out personally to resolve their complaints; these are high-value customers and losing them hurts the most.
✔️ Offer free shipping or faster delivery to reduce the impact of long warehouse distances.
✔️ Send a special cashback voucher as a "thank you for staying with us" to bring them back.

🟡 **Cluster 1 - "Happy but quietly drifting away"**

✔️ Don't apologize; there's nothing wrong with their experience. Instead, re-engage them with time-limited flash sales or new product alerts.
✔️ Send a short survey to understand why they stopped buying; it may be a change in needs, not dissatisfaction.

🟢 **Cluster 2 - "Loyal but under-rewarded"**

✔️ Increase cashback rates or send discount vouchers for their next purchase; they are currently getting the least value back.
✔️ Create a loyalty program specifically for long-`Tenure` customers so they feel recognized for their commitment.
✔️ Send personalized product recommendations to encourage more orders, since their `OrderCount` is the lowest.

---

## 🗂️ Project Structure

```
Churn_Prediction/
    ├── Banner.png
    ├── Customer_Churn_Prediction.ipynb     # Main notebook (EDA + ML + Clustering)
    └── churn_prediction.xlsx               # Raw dataset
```

---

## 🚀 How to Run This Project

**1. Clone the repository**
```bash
git clone https://github.com/TascoGitGud/Customer-Churn-Prediction-and-Segmentation-for-an-E-commerce-Company.git
```

**2. Install the required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

**3. Open the notebook**
```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```
