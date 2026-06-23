"""
============================================================
TASK 3 — House Price Prediction
QSkill AI/ML Internship | June 2026
============================================================
Objective : Predict house prices based on features like size,
            location, number of bedrooms, etc.
Dataset   : Realistic synthetic housing dataset (2000 samples)
            built from domain-known relationships so the model
            actually learns meaningful patterns.
Models    : Linear Regression, Ridge Regression,
            Decision Tree Regressor, Random Forest Regressor
Metrics   : MSE, RMSE, MAE, R² Score
============================================================
"""

import numpy as np
import os
os.makedirs('outputs', exist_ok=True)
import os
os.makedirs('outputs', exist_ok=True)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. BUILD REALISTIC HOUSING DATASET ───────────────────
print("=" * 60)
print("  TASK 3 — House Price Prediction")
print("=" * 60)

np.random.seed(42)
n = 2000

area_sqft      = np.random.normal(1800, 600, n).clip(500, 5000)
bedrooms       = np.random.randint(1, 6, n).astype(float)
bathrooms      = (bedrooms * 0.6 + np.random.normal(0, 0.5, n)).clip(1, 5).round()
garage_cars    = np.random.randint(0, 4, n).astype(float)
age_years      = np.random.randint(0, 50, n).astype(float)
distance_km    = np.abs(np.random.normal(15, 8, n)).clip(1, 50)   # dist to city center
has_pool       = (np.random.rand(n) > 0.75).astype(float)
school_rating  = np.random.uniform(3, 10, n)                      # 1-10

# Price formula with realistic weights + noise
noise = np.random.normal(0, 15000, n)
price = (
      80  * area_sqft
    + 8000 * bedrooms
    + 5000 * bathrooms
    + 4000 * garage_cars
    - 500  * age_years
    - 2000 * distance_km
    + 18000 * has_pool
    + 3000  * school_rating
    + 50000                  # base price
    + noise
).clip(30000, None)

X = pd.DataFrame({
    "Area_sqft"    : area_sqft.round(1),
    "Bedrooms"     : bedrooms,
    "Bathrooms"    : bathrooms,
    "Garage_cars"  : garage_cars,
    "Age_years"    : age_years,
    "Distance_km"  : distance_km.round(2),
    "Has_pool"     : has_pool,
    "School_rating": school_rating.round(2),
})
y = pd.Series((price / 1000).round(2), name="Price_$k")   # price in $thousands

print(f"\nDataset shape : {X.shape}")
print(f"Target range  : ${y.min():.0f}k – ${y.max():.0f}k  |  Mean: ${y.mean():.0f}k")
print(f"\nFirst 5 rows:")
print(pd.concat([X, y], axis=1).head().to_string(), "\n")
print("Missing values:", X.isnull().sum().sum(), "(dataset is clean ✓)\n")

# ── 2. EDA — FEATURE DISTRIBUTIONS ──────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("Housing Dataset — Feature Distributions", fontsize=14, fontweight="bold")
axes = axes.flatten()

colors = ["#3498db","#2ecc71","#e74c3c","#9b59b6","#f39c12","#1abc9c","#e67e22","#34495e"]
for i, col in enumerate(X.columns):
    axes[i].hist(X[col], bins=30, color=colors[i], edgecolor="white", alpha=0.85)
    axes[i].set_title(col); axes[i].set_xlabel("Value")

axes[-1].hist(y, bins=40, color="#c0392b", edgecolor="white", alpha=0.85)
axes[-1].set_title("Price ($k) — Target Distribution"); axes[-1].set_xlabel("Price ($k)")

plt.tight_layout()
plt.savefig("outputs/task3_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Feature distributions saved → task3_feature_distributions.png")

# ── 3. CORRELATION HEATMAP ───────────────────────────────
full_df = pd.concat([X, y], axis=1)
corr = full_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/task3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Correlation heatmap saved → task3_correlation_heatmap.png")

# ── 4. TOP FEATURE SCATTER ────────────────────────────────
top_feat = corr["Price_$k"].drop("Price_$k").abs().idxmax()
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X[top_feat], y, alpha=0.3, s=10, c=y, cmap="coolwarm")
plt.colorbar(sc, ax=ax, label="Price ($k)")
ax.set_xlabel(top_feat); ax.set_ylabel("Price ($k)")
ax.set_title(f"Strongest Predictor: {top_feat} vs House Price")
plt.tight_layout()
plt.savefig("outputs/task3_top_feature_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[✓] Scatter plot saved → task3_top_feature_scatter.png  (top feature: {top_feat})\n")

# ── 5. SPLIT & NORMALIZE ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── 6. TRAIN MODELS ──────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression" : Ridge(alpha=1.0),
    "Decision Tree"    : DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest"    : RandomForestRegressor(n_estimators=100, max_depth=10,
                                                random_state=42, n_jobs=-1),
}

results = {}
best_name, best_r2, best_preds = "", -np.inf, None

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mse   = mean_squared_error(y_test, preds)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}
    print(f"\n▸ {name}")
    print(f"  MSE={mse:.2f}  RMSE=${rmse:.2f}k  MAE=${mae:.2f}k  R²={r2:.4f}")
    if r2 > best_r2:
        best_r2, best_name, best_preds = r2, name, preds

best_model = models[best_name]
print(f"\n🏆 Best model : {best_name} (R²={best_r2:.4f})")

# ── 7. FEATURE IMPORTANCE (Random Forest) ─────────────────
importances = models["Random Forest"].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind="bar", color="#e67e22", edgecolor="white", ax=ax)
ax.set_title("Random Forest — Feature Importance")
ax.set_ylabel("Importance Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/task3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Feature importance saved → task3_feature_importance.png")

# ── 8. ACTUAL vs PREDICTED (all 4 models) ────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("House Price — Actual vs Predicted", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes.flatten(), models.items()):
    preds = model.predict(X_test_s)
    r2 = r2_score(y_test, preds)
    ax.scatter(y_test, preds, alpha=0.3, s=8, color="#3498db")
    lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual ($k)"); ax.set_ylabel("Predicted ($k)")
    ax.set_title(f"{name}\nR²={r2:.4f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/task3_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Actual vs Predicted plot saved → task3_actual_vs_predicted.png")

# ── 9. MODEL COMPARISON BAR CHARTS ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — House Price Prediction", fontsize=13, fontweight="bold")
names     = list(results.keys())
r2_vals   = [results[n]["R²"]   for n in names]
rmse_vals = [results[n]["RMSE"] for n in names]
palette   = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

bars = axes[0].bar(names, r2_vals, color=palette, edgecolor="white")
axes[0].set_title("R² Score (higher = better)"); axes[0].set_ylim(0, 1.05)
for bar, v in zip(bars, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
axes[0].tick_params(axis="x", rotation=15)

bars = axes[1].bar(names, rmse_vals, color=palette, edgecolor="white")
axes[1].set_title("RMSE — $k (lower = better)")
for bar, v in zip(bars, rmse_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"${v:.1f}k", ha="center", fontweight="bold", fontsize=9)
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("outputs/task3_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Model comparison chart saved → task3_model_comparison.png")

# ── 10. RESIDUAL ANALYSIS ────────────────────────────────
residuals = y_test - best_preds
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Residual Analysis — {best_name}", fontsize=13, fontweight="bold")

axes[0].scatter(best_preds, residuals, alpha=0.3, s=8, color="#e74c3c")
axes[0].axhline(0, color="black", lw=1.5, linestyle="--")
axes[0].set_xlabel("Predicted ($k)"); axes[0].set_ylabel("Residual ($k)")
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
axes[1].set_title("Residual Distribution"); axes[1].set_xlabel("Residual ($k)")

plt.tight_layout()
plt.savefig("outputs/task3_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Residual analysis saved → task3_residuals.png")

# ── 11. SAMPLE PREDICTIONS ───────────────────────────────
print("\n" + "=" * 60)
print(f"  SAMPLE PREDICTIONS — {best_name}")
print("=" * 60)
sample_s = X_test_s[:8]
pred_vals   = best_model.predict(sample_s)
actual_vals = y_test.iloc[:8].values

print(f"{'#':<4} {'Actual':>12} {'Predicted':>12} {'Error':>12}")
print("-" * 44)
for i, (act, pred) in enumerate(zip(actual_vals, pred_vals)):
    err = pred - act
    print(f"{i+1:<4} ${act:>8.1f}k   ${pred:>8.1f}k   {err:>+8.1f}k")

print(f"\n[✓] Task 3 Complete!\n")
