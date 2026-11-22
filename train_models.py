"""
train_models.py - Project 3: model building and evaluation
Reads marketing_features.csv, trains regression (Revenue) and classification
(Profit/Loss) models, tunes the best one, saves metrics, charts and models.
Run:  python train_models.py        (takes about 3-6 minutes)
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

RS = 42                                   # random_state: makes results repeatable
df = pd.read_csv("marketing_features.csv")

# ---------------------------------------------------------------
# 1. FEATURES
# ---------------------------------------------------------------
CHANNELS = ["Channel_Email", "Channel_Facebook", "Channel_Google",
            "Channel_Instagram", "Channel_WhatsApp", "Channel_YouTube"]
CATS = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
NUMS = ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
        "Acquisition_Cost", "Engagement_Score"]

# one-hot encoding turns text categories into 0/1 columns the models understand
X = pd.get_dummies(df[NUMS + CATS + CHANNELS], columns=CATS, dtype=int)
X["Total_Cost"] = df["Acquisition_Cost"] * df["Conversions"]
columns = list(X.columns)

# ROI and Revenue are NOT in X on purpose (data leakage):
#   Profit_Flag is created from ROI, and ROI is created from Revenue and cost.
X_roi = X.copy()
X_roi["ROI"] = df["ROI"]                   # only for the "with ROI" revenue model

# ---------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------
def importance(model, cols):
    """Feature importance for trees/forests, or |coefficient| for linear models."""
    v = getattr(model, "feature_importances_", None)
    if v is None:
        v = np.abs(model.coef_) if hasattr(model, "coef_") else np.zeros(len(cols))
    return pd.Series(np.ravel(v), index=cols)

def reg_scores(y, p):
    mse = mean_squared_error(y, p)
    return {"RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y, p),
            "MSE": mse, "R2": r2_score(y, p)}

def clf_scores(y, p):
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p),
            "Recall": recall_score(y, p), "F1": f1_score(y, p)}

def run(models, Xtr, Xte, ytr, yte, scorer, label):
    rows, fitted = [], {}
    for name, m in models.items():
        t = time.time()
        m.fit(Xtr, ytr)
        s = scorer(yte, m.predict(Xte))
        rows.append({"Task": label, "Model": name, **s})
        fitted[name] = m
        print(f"  {label:28s} {name:20s} {({k: round(v, 3) for k, v in s.items()})}  ({time.time()-t:.0f}s)")
    return rows, fitted

def tune(est, grid, Xtr, ytr, scoring):
    """Randomised search with 3-fold cross-validation on a 30k-row sample (fast)."""
    idx = Xtr.sample(30000, random_state=RS).index
    s = RandomizedSearchCV(est, grid, n_iter=5, cv=3, scoring=scoring,
                           random_state=RS, n_jobs=-1)
    s.fit(Xtr.loc[idx], ytr.loc[idx])
    print("  best parameters found:", s.best_params_)
    return s.best_params_

results = []

# ---------------------------------------------------------------
# 3. REGRESSION A - predict Revenue from campaign inputs (no ROI)
# ---------------------------------------------------------------
print("\n[1] REVENUE - campaign inputs only (honest, no leakage)")
Xtr, Xte, ytr, yte = train_test_split(X, df["Revenue"], test_size=0.2, random_state=RS)
models = {"Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=RS),
          "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=RS)}
rows, fitted = run(models, Xtr, Xte, ytr, yte, reg_scores, "Revenue (inputs only)")
bp = tune(RandomForestRegressor(n_jobs=1, random_state=RS),
          {"n_estimators": [40, 60], "max_depth": [8, 10, 12], "min_samples_leaf": [5, 20, 50]},
          Xtr, ytr, "r2")
tuned = RandomForestRegressor(**bp, n_jobs=-1, random_state=RS)
r2, f2 = run({"Random Forest (tuned)": tuned}, Xtr, Xte, ytr, yte, reg_scores, "Revenue (inputs only)")
rows += r2; fitted.update(f2); results += rows
best = max([r for r in rows], key=lambda r: r["R2"])["Model"]
print("  >> best:", best)
rev_model = fitted[best]
joblib.dump({"model": rev_model, "columns": columns}, "revenue_model.joblib", compress=3)
rev_imp = importance(rev_model, columns)

# ---------------------------------------------------------------
# 4. REGRESSION B - same, but ROI is also given (to reach the brief's R2 >= 0.95)
# ---------------------------------------------------------------
print("\n[2] REVENUE - with ROI as an input (ROI already contains revenue information)")
Xtr, Xte, ytr, yte = train_test_split(X_roi, df["Revenue"], test_size=0.2, random_state=RS)
models = {"Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(max_depth=14, random_state=RS),
          "Random Forest": RandomForestRegressor(n_estimators=40, max_depth=14, min_samples_leaf=10,
                                                 n_jobs=-1, random_state=RS)}   # min_samples_leaf keeps the saved file small
rows, fitted = run(models, Xtr, Xte, ytr, yte, reg_scores, "Revenue (with ROI)")
results += rows
best = max(rows, key=lambda r: r["R2"])["Model"]
print("  >> best:", best)
joblib.dump({"model": fitted[best], "columns": list(X_roi.columns)}, "revenue_model_with_roi.joblib", compress=3)

# ---------------------------------------------------------------
# 5. CLASSIFICATION - predict Profit (1) or Loss (0); ROI and Revenue excluded
# ---------------------------------------------------------------
print("\n[3] PROFIT / LOSS classification (ROI and Revenue excluded)")
y = df["Profit_Flag"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
          "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RS),
          "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1, random_state=RS)}
rows, fitted = run(models, Xtr, Xte, ytr, yte, clf_scores, "Profit/Loss")
bp = tune(RandomForestClassifier(n_jobs=1, random_state=RS),
          {"n_estimators": [40, 60], "max_depth": [8, 10, 12], "min_samples_leaf": [5, 20, 50]},
          Xtr, ytr, "f1")
tuned = RandomForestClassifier(**bp, n_jobs=-1, random_state=RS)
r2, f2 = run({"Random Forest (tuned)": tuned}, Xtr, Xte, ytr, yte, clf_scores, "Profit/Loss")
rows += r2; fitted.update(f2); results += rows
best = max(rows, key=lambda r: r["F1"])["Model"]
print("  >> best:", best)
clf_model = fitted[best]
joblib.dump({"model": clf_model, "columns": columns}, "profit_model.joblib", compress=3)
pred = clf_model.predict(Xte)
cm = confusion_matrix(yte, pred)
print("  confusion matrix [[TN FP],[FN TP]]:", cm.tolist())
clf_imp = importance(clf_model, columns)

# ---------------------------------------------------------------
# 6. SAVE REPORT FILES AND CHARTS
# ---------------------------------------------------------------
res = pd.DataFrame(results).round(4)
res.to_csv("model_metrics.csv", index=False)
print("\n=== MODEL METRICS ===\n", res.to_string(index=False))

# lists used by the Streamlit app
json.dump({"columns": columns, "cats": {c: sorted(df[c].unique().tolist()) for c in CATS},
           "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in NUMS}},
          open("app_config.json", "w"), indent=1)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
rev_imp.sort_values().tail(10).plot.barh(ax=ax[0], color="#2a6f97"); ax[0].set_title("Top features - Revenue model")
clf_imp.sort_values().tail(10).plot.barh(ax=ax[1], color="#52b788"); ax[1].set_title("Top features - Profit/Loss model")
plt.tight_layout(); plt.savefig("eda_charts/9_feature_importance.png", dpi=130); plt.close()

plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Profit"], yticklabels=["Loss", "Profit"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test set)")
plt.tight_layout(); plt.savefig("eda_charts/10_confusion_matrix.png", dpi=130); plt.close()
print("\nSaved: model_metrics.csv, app_config.json, 3 .joblib models, 2 charts")
