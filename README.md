# 🤖 QSkill AI/ML Internship Tasks — June 2026

> **Intern:** Goranshu  
> **Domain:** Artificial Intelligence & Machine Learning  
> **Duration:** 1st June 2026 – 1st July 2026  
> **Organization:** QSkill

Three complete machine learning projects covering classification, NLP, and regression — built with scikit-learn, pandas, matplotlib, and seaborn.

---

## 📁 Project Structure

```
qskill-aiml-internship/
│
├── task1_iris_classification.py       # Iris Flower Classification
├── task2_spam_detector.py             # Spam Mail Detector
├── task3_house_price_prediction.py    # House Price Prediction
│
├── outputs/
│   ├── task1_eda.png
│   ├── task1_confusion_matrices.png
│   ├── task1_model_comparison.png
│   ├── task2_eda.png
│   ├── task2_evaluation.png
│   ├── task2_top_spam_words.png
│   ├── task3_feature_distributions.png
│   ├── task3_correlation_heatmap.png
│   ├── task3_top_feature_scatter.png
│   ├── task3_actual_vs_predicted.png
│   ├── task3_model_comparison.png
│   ├── task3_feature_importance.png
│   └── task3_residuals.png
│
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/qskill-aiml-internship.git
cd qskill-aiml-internship

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn
```

> Python 3.8 or higher is required.

---

## Task 1 — Iris Flower Classification

### Objective
Classify iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on four numeric measurements: sepal length, sepal width, petal length, and petal width.

### Dataset
- **Source:** `sklearn.datasets.load_iris()` (built-in, no download needed)
- **Size:** 150 samples × 4 features
- **Classes:** 3, perfectly balanced (50 samples each)

### Approach

| Step | What was done |
|------|--------------|
| Load | Loaded from sklearn, converted to pandas DataFrame |
| EDA | Scatter plots (sepal & petal), histograms, box plots |
| Split | 80% train / 20% test, stratified to preserve class ratio |
| Scale | StandardScaler — zero mean, unit variance |
| Train | KNN (k=5), Logistic Regression, Decision Tree |
| Evaluate | Accuracy, Precision, Recall, F1, Confusion Matrix |

### Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| K-Nearest Neighbors (k=5) | **93.33%** | 0.9444 | 0.9333 | 0.9327 |
| Logistic Regression | 93.33% | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 93.33% | 0.9333 | 0.9333 | 0.9333 |

**Best Model:** KNN (k=5) — 93.33% accuracy

#### Key Insight
- **Setosa** is perfectly separable from the other two classes (100% precision & recall)
- **Versicolor** and **Virginica** overlap slightly, causing the small error margin
- Petal measurements are far more discriminative than sepal measurements

### Output Files
- `task1_eda.png` — Scatter plots and distributions by species
- `task1_confusion_matrices.png` — Side-by-side confusion matrices for all 3 models
- `task1_model_comparison.png` — Bar chart comparing accuracy across models

### Run
```bash
python task1_iris_classification.py
```

---

## Task 2 — Spam Mail Detector

### Objective
Build an NLP classifier that detects whether an SMS/email message is **spam** or **ham (not spam)** using text feature extraction.

### Dataset
- **Source:** [SMS Spam Collection — UCI Repository](https://github.com/justmarkham/pycon-2016-tutorial/blob/master/data/sms.tsv) (loaded automatically via URL)
- **Size:** 5,572 messages
- **Class split:** 4,825 ham (86.6%) vs 747 spam (13.4%)

### Approach

| Step | What was done |
|------|--------------|
| Load | Fetched SMS Spam Collection via URL (TSV format) |
| Clean | Lowercased, removed URLs/numbers/punctuation with regex |
| Vectorize | TF-IDF (max 5000 features, unigrams + bigrams, English stopwords removed) |
| Split | 80% train / 20% test, stratified |
| Train | Multinomial Naive Bayes, Logistic Regression |
| Evaluate | Accuracy, Precision, Recall, F1, AUC-ROC |

### Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Multinomial Naive Bayes | **98.57%** | 0.9716 | 0.9195 | **0.9448** | 0.9918 |
| Logistic Regression | 97.67% | 0.9556 | 0.8658 | 0.9085 | 0.9883 |

**Best Model:** Multinomial Naive Bayes — 98.57% accuracy, F1 = 0.9448

#### Key Insights
- **Naive Bayes** is a natural fit for text classification — it works on word probability distributions which is exactly what spam detection needs
- **TF-IDF with bigrams** (e.g., "free prize", "click here") is more powerful than plain word counts because spam phrases are often two-word combos
- High **Precision (97%)** means very few legitimate emails get wrongly flagged as spam — critical for a real-world detector
- AUC of **0.9918** means the model almost perfectly separates spam from ham across all decision thresholds

#### Live Demo Predictions
```
🚨 SPAM  (98.3%)  | "Congratulations! You have won a free iPhone. Claim now!"
✅ HAM   (0.1%)   | "Hey, are we still on for lunch tomorrow?"
🚨 SPAM  (55.0%)  | "URGENT: Your bank account has been compromised. Click here."
✅ HAM   (12.3%)  | "Can you send me the homework assignment please?"
🚨 SPAM  (100.0%) | "Win £1000 cash prize! Text WIN to 80082 to claim your reward."
```

### Output Files
- `task2_eda.png` — Message length distribution + class balance pie chart
- `task2_evaluation.png` — Confusion matrices + ROC curves for both models
- `task2_top_spam_words.png` — Top 15 spam-indicating words (Naive Bayes log-prob analysis)

### Run
```bash
python task2_spam_detector.py
```

> The script auto-downloads the dataset. If offline, it falls back to a built-in representative dataset.

---

## Task 3 — House Price Prediction

### Objective
Predict the price of a house (in $thousands) based on structural and location-based features using regression models.

### Dataset
- **Type:** Realistic synthetic dataset (built from domain-known relationships)
- **Size:** 2,000 samples × 8 features
- **Target:** House price in $k (range: $37k – $391k, mean ~$218k)

#### Features

| Feature | Description |
|---------|-------------|
| `Area_sqft` | Total area of the house in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Garage_cars` | Garage capacity (number of cars) |
| `Age_years` | Age of the house in years |
| `Distance_km` | Distance from city center in km |
| `Has_pool` | Whether the house has a pool (0/1) |
| `School_rating` | Nearby school quality rating (3–10) |

### Approach

| Step | What was done |
|------|--------------|
| Generate | Realistic synthetic data with domain-known price weights + noise |
| EDA | Feature histograms, correlation heatmap, scatter of top predictor |
| Preprocess | StandardScaler normalization |
| Split | 80% train / 20% test |
| Train | Linear Regression, Ridge Regression, Decision Tree, Random Forest |
| Evaluate | MSE, RMSE, MAE, R² Score |
| Analyze | Feature importance (Random Forest), residual plots |

### Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Ridge Regression** | **0.9337** | $14.44k | $11.44k |
| Linear Regression | 0.9336 | $14.44k | $11.45k |
| Random Forest | 0.8902 | $18.58k | $14.39k |
| Decision Tree | 0.8026 | $24.91k | $19.76k |

**Best Model:** Ridge Regression — R² = 0.9337 (explains 93.4% of price variance)

#### Key Insights
- **Ridge** beat Random Forest here because the dataset was built using linear relationships — this is an important lesson: **always match model complexity to data structure**
- `Area_sqft` is the strongest predictor (highest correlation with price) — larger houses cost more
- `Distance_km` has a negative correlation — farther from city center = lower price
- `Age_years` is negatively correlated — older houses are cheaper
- **R² of 0.93** means the model predicts within ~$14k of the actual price on average, which is strong for a regression task
- The residual distribution is approximately normal and centered at 0 — this confirms the model is unbiased

### Sample Predictions (Ridge Regression)
```
#    Actual      Predicted    Error
1    $189.0k     $184.4k      -4.6k
2    $220.8k     $228.8k      +8.1k
3    $317.9k     $308.3k      -9.7k
4    $204.1k     $210.6k      +6.5k
5    $124.6k     $115.4k      -9.2k
```

### Output Files
- `task3_feature_distributions.png` — Histogram of all 8 features + target
- `task3_correlation_heatmap.png` — Heatmap showing feature-to-feature and feature-to-price correlations
- `task3_top_feature_scatter.png` — Area vs Price scatter (colored by price)
- `task3_actual_vs_predicted.png` — Actual vs predicted plots for all 4 models
- `task3_model_comparison.png` — R² and RMSE bar charts side by side
- `task3_feature_importance.png` — Random Forest feature importances
- `task3_residuals.png` — Residuals vs predicted + residual distribution

### Run
```bash
python task3_house_price_prediction.py
```

---

## 📊 Metrics Explained

| Metric | Used In | Meaning |
|--------|---------|---------|
| **Accuracy** | Task 1, 2 | % of total predictions that were correct |
| **Precision** | Task 1, 2 | Of all predicted positives, how many were actually positive |
| **Recall** | Task 1, 2 | Of all actual positives, how many did we correctly catch |
| **F1 Score** | Task 1, 2 | Harmonic mean of Precision and Recall — best single metric when classes are imbalanced |
| **AUC-ROC** | Task 2 | Area under ROC curve — measures model's ability to distinguish classes at all thresholds |
| **MSE** | Task 3 | Mean Squared Error — average of squared prediction errors |
| **RMSE** | Task 3 | Root MSE — same unit as target (dollars), easier to interpret |
| **MAE** | Task 3 | Mean Absolute Error — average absolute difference between actual and predicted |
| **R² Score** | Task 3 | Proportion of variance explained by the model (1.0 = perfect, 0 = no better than mean) |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Core language |
| scikit-learn | latest | ML models, preprocessing, metrics |
| pandas | latest | Data loading and manipulation |
| numpy | latest | Numerical operations |
| matplotlib | latest | Plotting and visualizations |
| seaborn | latest | Statistical visualizations (heatmaps, etc.) |

---

## 📚 Skills Gained

- **Task 1:** Numeric data analysis, multi-class classification, confusion matrix reading, model comparison
- **Task 2:** Text preprocessing with regex, TF-IDF vectorization, NLP basics, binary classification, ROC curves
- **Task 3:** Tabular data handling, feature engineering, regression modeling, MSE/R² interpretation, residual analysis, feature importance

---

## 👤 Author

**Goranshu**  
B.Tech CSE — Chandigarh Group of Colleges, Jhanjeri  
Web Developer Intern @ XC0MRADE Technologies  
QSkill AI/ML Internship — June 2026
