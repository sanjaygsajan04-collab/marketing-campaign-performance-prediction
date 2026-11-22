"""
clean_data.py  -  Project 3: Marketing Campaign Performance Prediction
Step 1: load the 3 brand CSVs, clean them, validate ROI, create features.

Run:  python clean_data.py
Outputs: marketing_raw_combined.csv, marketing_clean.csv, marketing_features.csv
"""
import glob
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1. DATA COLLECTION - read every *campaign_data_with_nulls.csv
# ---------------------------------------------------------------
files = sorted(glob.glob("*campaign_data_with_nulls.csv"))
if not files:
    raise SystemExit("No CSV files found. Run this script inside the project_3 folder.")

frames = []
for f in files:
    df = pd.read_csv(f)
    # brand name is only in the file name, so we create a Brand column
    name = os.path.basename(f).lower()
    brand = "Nykaa" if "nykaa" in name else "Purplle" if "purplle" in name else "Tira"
    df["Brand"] = brand
    frames.append(df)
    print(f"Loaded {f}: {df.shape[0]} rows")

data = pd.concat(frames, ignore_index=True)
data.to_csv("marketing_raw_combined.csv", index=False)
print("\nCombined shape:", data.shape)

# ---------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------
print("\nRows with at least one null:", int(data.isna().any(axis=1).sum()))

# 2a. Clean text columns (remove extra spaces)
text_cols = ["Campaign_Type", "Target_Audience", "Channel_Used", "Language",
             "Customer_Segment"]
for c in text_cols:
    data[c] = data[c].str.strip()

# 2b. Data type conversion: Date text -> real date (day first, e.g. 29-04-2025)
data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")

# 2c. Remove duplicates. Null IDs are NOT duplicates, so check only real IDs.
full_dups = data.duplicated().sum()
data = data.drop_duplicates()
has_id = data["Campaign_ID"].notna()
id_dups = data[has_id].duplicated(subset="Campaign_ID").sum()
data = pd.concat([data[has_id].drop_duplicates(subset="Campaign_ID"), data[~has_id]])
data = data.sort_index().reset_index(drop=True)
print(f"Duplicate rows removed: full={full_dups}, repeated Campaign_ID={id_dups}")

# 2d. Missing Campaign_ID: rebuild a unique ID from brand + row number
prefix = {"Nykaa": "NY", "Purplle": "PU", "Tira": "TI"}
missing_id = data["Campaign_ID"].isna()
data.loc[missing_id, "Campaign_ID"] = [
    f"{prefix[b]}-CMP-FILL{i}" for i, b in zip(data.index[missing_id], data.loc[missing_id, "Brand"])]

# 2e. ROI validation. Hidden rule found in the data:
#     Total cost = Acquisition_Cost * Conversions
#     ROI        = (Revenue - Total cost) / Total cost
# Use the rule to REBUILD missing Revenue / Acquisition_Cost / ROI when the
# other values are present (better than guessing with an average).
cost = data["Acquisition_Cost"] * data["Conversions"]
roi_calc = (data["Revenue"] - cost) / cost
ok = data["ROI"].notna() & roi_calc.notna()
mismatch = (abs(roi_calc[ok] - data.loc[ok, "ROI"]) > 0.01).sum()
print(f"ROI check: {ok.sum()} rows comparable, {mismatch} wrong ROI values")

# rebuild Revenue:  Revenue = cost * (1 + ROI)
m = data["Revenue"].isna() & cost.notna() & data["ROI"].notna()
data.loc[m, "Revenue"] = (cost * (1 + data["ROI"]))[m].round(0)
# rebuild Acquisition_Cost:  cost = Revenue / (1 + ROI);  AC = cost / Conversions
m = (data["Acquisition_Cost"].isna() & data["Revenue"].notna()
     & data["ROI"].notna() & data["Conversions"].notna())
data.loc[m, "Acquisition_Cost"] = (data["Revenue"] / (1 + data["ROI"]) / data["Conversions"])[m].round(2)

# ---------------------------------------------------------------
# 3. FILL THE REMAINING NULLS
#    numbers -> median of that brand (median is not hurt by outliers)
#    categories -> most frequent value (mode)
#    Channel_Used -> "Unknown" (we should not invent a channel)
# ---------------------------------------------------------------
num_cols = ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
            "Revenue", "Acquisition_Cost", "Engagement_Score"]
for c in num_cols:
    data[c] = data[c].fillna(data.groupby("Brand")[c].transform("median"))

for c in ["Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]:
    data[c] = data[c].fillna(data[c].mode()[0])
data["Channel_Used"] = data["Channel_Used"].fillna("Unknown")
# Date is left empty when missing (a made-up date would distort the monthly trend)

# Recalculate ROI for EVERY row with the rule, so ROI is consistent everywhere
data["Total_Cost"] = data["Acquisition_Cost"] * data["Conversions"]
data["ROI"] = ((data["Revenue"] - data["Total_Cost"]) / data["Total_Cost"]).round(2)

print("\nNulls left after cleaning (only Date, left empty on purpose):", int(data.isna().sum().sum()))
data.to_csv("marketing_clean.csv", index=False)

# ---------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------
# 4a. Profit/Loss flag: 1 = Profit (ROI > 0), 0 = Loss
data["Profit_Flag"] = (data["ROI"] > 0).astype(int)

# 4b. Multi-label encoding of Channel_Used ("YouTube, Instagram" -> two 1s)
channels = sorted({ch for v in data["Channel_Used"] for ch in v.split(", ") if ch != "Unknown"})
for ch in channels:
    data[f"Channel_{ch}"] = data["Channel_Used"].apply(lambda v: int(ch in v.split(", ")))
data["Channel_Count"] = data[[f"Channel_{c}" for c in channels]].sum(axis=1)

# 4c. Useful date features
data["Month"] = data["Date"].dt.month.astype("Int64")   # stays empty where Date is missing
data["Year"] = data["Date"].dt.year.astype("Int64")

# 4d. Ratios (rates) that explain performance
data["CTR"] = (data["Clicks"] / data["Impressions"]).round(4)          # click-through rate
data["Conversion_Rate"] = (data["Conversions"] / data["Clicks"]).round(4)

data.to_csv("marketing_features.csv", index=False)

print("\nChannels found:", channels)
print("Profit/Loss split:\n", data["Profit_Flag"].map({1: "Profit", 0: "Loss"}).value_counts(normalize=True).round(3).to_string())
print("\nFinal shape:", data.shape)
print("Saved: marketing_raw_combined.csv, marketing_clean.csv, marketing_features.csv")
