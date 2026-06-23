"""
============================================================
TASK 1 — Iris Flower Classification
QSkill AI/ML Internship | June 2026
============================================================
Objective : Classify iris flowers into Setosa, Versicolor,
            and Virginica based on sepal & petal measurements.
Model used : K-Nearest Neighbors (KNN) + comparison with
             Logistic Regression and Decision Tree
============================================================
"""
import os
os.makedirs("outputs", exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

# ── 1. LOAD DATASET ──────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
class_names = iris.target_names          # ['setosa', 'versicolor', 'virginica']

print("=" * 60)
print("  TASK 1 — Iris Flower Classification")
print("=" * 60)
print(f"\nDataset shape : {X.shape}")
print(f"Classes       : {list(class_names)}")
print(f"Class counts  :\n{y.value_counts().rename(index=dict(enumerate(class_names)))}\n")
print("First 5 rows:")
print(pd.concat([X, y.rename("species")], axis=1).head(), "\n")

# ── 2. EXPLORATORY DATA ANALYSIS (saved to PNG) ──────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Iris Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig)

colors = ["#e74c3c", "#2ecc71", "#3498db"]
palette = {i: colors[i] for i in range(3)}

# 2a. Pair-wise scatter: sepal length vs sepal width
ax1 = fig.add_subplot(gs[0, 0])
for cls in range(3):
    mask = y == cls
    ax1.scatter(X.loc[mask, "sepal length (cm)"],
                X.loc[mask, "sepal width (cm)"],
                label=class_names[cls], color=colors[cls], alpha=0.7, edgecolors="white", s=60)
ax1.set_xlabel("Sepal Length (cm)"); ax1.set_ylabel("Sepal Width (cm)")
ax1.set_title("Sepal: Length vs Width"); ax1.legend()

# 2b. Pair-wise scatter: petal length vs petal width
ax2 = fig.add_subplot(gs[0, 1])
for cls in range(3):
    mask = y == cls
    ax2.scatter(X.loc[mask, "petal length (cm)"],
                X.loc[mask, "petal width (cm)"],
                label=class_names[cls], color=colors[cls], alpha=0.7, edgecolors="white", s=60)
ax2.set_xlabel("Petal Length (cm)"); ax2.set_ylabel("Petal Width (cm)")
ax2.set_title("Petal: Length vs Width"); ax2.legend()

# 2c. Feature distribution histograms
ax3 = fig.add_subplot(gs[1, 0])
for feat in X.columns:
    ax3.hist(X[feat], bins=20, alpha=0.5, label=feat)
ax3.set_title("Feature Distributions"); ax3.set_xlabel("Value (cm)"); ax3.legend(fontsize=7)

# 2d. Box plot per feature
ax4 = fig.add_subplot(gs[1, 1])
X.boxplot(ax=ax4, grid=False, patch_artist=True)
ax4.set_title("Box Plot — All Features"); ax4.set_ylabel("Value (cm)")
plt.xticks(rotation=15, fontsize=8)

plt.tight_layout()
plt.savefig("outputs/task1_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] EDA plot saved → task1_eda.png")

# ── 3. SPLIT & SCALE ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain size : {len(X_train)} | Test size : {len(X_test)}")

# ── 4. TRAIN 3 MODELS & COMPARE ──────────────────────────
models = {
    "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression"      : LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree"            : DecisionTreeClassifier(random_state=42),
}

results = {}
best_model_name, best_acc, best_preds = "", 0, None

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec  = recall_score(y_test, preds, average="weighted")
    f1   = f1_score(y_test, preds, average="weighted")
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"\n▸ {name}")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    if acc > best_acc:
        best_acc, best_model_name, best_preds = acc, name, preds

print(f"\n🏆 Best model : {best_model_name} ({best_acc*100:.2f}% accuracy)")

# ── 5. DETAILED REPORT FOR BEST MODEL ────────────────────
print("\n" + "=" * 60)
print(f"  CLASSIFICATION REPORT — {best_model_name}")
print("=" * 60)
print(classification_report(y_test, best_preds, target_names=class_names))

# ── 6. CONFUSION MATRIX PLOT ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Iris Classification — Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X_test_s)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    acc = accuracy_score(y_test, preds)
    ax.set_title(f"{name}\nAccuracy: {acc*100:.1f}%", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/task1_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Confusion matrix plot saved → task1_confusion_matrices.png")

# ── 7. ACCURACY COMPARISON BAR CHART ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names  = list(results.keys())
accs   = [results[n]["Accuracy"] for n in names]
bars   = ax.bar(names, accs, color=["#3498db", "#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Accuracy"); ax.set_title("Model Accuracy Comparison — Iris Classification")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{acc*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
plt.xticks(wrap=True)
plt.tight_layout()
plt.savefig("outputs/task1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Model comparison chart saved → task1_model_comparison.png")
print("\n[✓] Task 1 Complete!\n")
