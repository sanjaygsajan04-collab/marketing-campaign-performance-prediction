"""
predict_utils.py - applies the SAME preprocessing as training to one new campaign.
Training turned text columns into 0/1 columns (one-hot) and channels into 6 flags
(multi-label). The app must rebuild exactly the same columns, in the same order,
otherwise the model would read the wrong numbers.
"""
import json
import joblib
import pandas as pd

CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]
CATS = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]


def load_assets():
    config = json.load(open("app_config.json"))
    rev = joblib.load("revenue_model.joblib")
    rev_roi = joblib.load("revenue_model_with_roi.joblib")
    clf = joblib.load("profit_model.joblib")
    return config, rev, rev_roi, clf


def build_row(inp, columns, roi=None):
    """inp: dict of the user's inputs. Returns a 1-row DataFrame with `columns`."""
    row = {c: 0 for c in columns}
    for c in ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
              "Acquisition_Cost", "Engagement_Score"]:
        row[c] = inp[c]
    for cat in CATS:                                   # one-hot encoding
        row[f"{cat}_{inp[cat]}"] = 1
    for ch in CHANNELS:                                # multi-label encoding
        row[f"Channel_{ch}"] = int(ch in inp["Channels"])
    row["Total_Cost"] = inp["Acquisition_Cost"] * inp["Conversions"]
    if roi is not None and "ROI" in columns:
        row["ROI"] = roi
    return pd.DataFrame([row])[columns]


def predict(inp, assets, known_roi=None):
    config, rev, rev_roi, clf = assets
    X = build_row(inp, rev["columns"])
    total_cost = inp["Acquisition_Cost"] * inp["Conversions"]
    if known_roi is None:
        revenue = float(rev["model"].predict(X)[0])
    else:
        Xr = build_row(inp, rev_roi["columns"], roi=known_roi)
        revenue = float(rev_roi["model"].predict(Xr)[0])
    Xc = build_row(inp, clf["columns"])
    p_profit = float(clf["model"].predict_proba(Xc)[0][1])
    return {"revenue": revenue, "total_cost": total_cost,
            "implied_roi": (revenue - total_cost) / total_cost if total_cost else 0.0,
            "p_profit": p_profit, "is_profit": p_profit >= 0.5}
