"""
eda.py - Project 3: Exploratory Data Analysis
Reads marketing_features.csv, prints key insights, saves charts to eda_charts/
Run:  python eda.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save charts as files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("eda_charts", exist_ok=True)
df = pd.read_csv("marketing_features.csv", parse_dates=["Date"])
sns.set_theme(style="whitegrid", palette="viridis")
channels = ["Channel_Email", "Channel_Facebook", "Channel_Google", "Channel_Instagram", "Channel_WhatsApp", "Channel_YouTube"]

def save(name):
    plt.tight_layout()
    plt.savefig(f"eda_charts/{name}.png", dpi=130)
    plt.close()

# ---------- 1. Brand performance ----------
brand = df.groupby("Brand").agg(
    Campaigns=("Campaign_ID", "count"),
    Total_Revenue=("Revenue", "sum"),
    Avg_Revenue=("Revenue", "mean"),
    Avg_ROI=("ROI", "mean"),
    Profit_Rate=("Profit_Flag", "mean")).round(2)
print("=== 1. BRAND PERFORMANCE ===\n", brand.to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sns.barplot(x=brand.index, y=brand["Avg_Revenue"], ax=ax[0]); ax[0].set_title("Average revenue per campaign")
sns.barplot(x=brand.index, y=brand["Avg_ROI"], ax=ax[1]); ax[1].set_title("Average ROI")
sns.barplot(x=brand.index, y=brand["Profit_Rate"] * 100, ax=ax[2]); ax[2].set_title("% campaigns in profit")
save("1_brand_performance")

# ---------- 2. Campaign type ----------
ctype = df.groupby("Campaign_Type").agg(Avg_Revenue=("Revenue", "mean"),
        Avg_ROI=("ROI", "mean"), Profit_Rate=("Profit_Flag", "mean")).round(2).sort_values("Avg_ROI", ascending=False)
print("\n=== 2. CAMPAIGN TYPE ===\n", ctype.to_string())
plt.figure(figsize=(8, 4.5))
sns.barplot(x=ctype["Avg_ROI"], y=ctype.index)
plt.title("Average ROI by campaign type")
save("2_campaign_type_roi")

# ---------- 3. Channel effectiveness ----------
rows = []
for c in channels:
    s = df[df[c] == 1]
    rows.append([c.replace("Channel_", ""), len(s), s["ROI"].mean(), s["Revenue"].mean(), s["Profit_Flag"].mean()])
ch = pd.DataFrame(rows, columns=["Channel", "Campaigns", "Avg_ROI", "Avg_Revenue", "Profit_Rate"]).round(2).sort_values("Avg_ROI", ascending=False)
print("\n=== 3. CHANNEL EFFECTIVENESS ===\n", ch.to_string(index=False))
plt.figure(figsize=(8, 4.5))
sns.barplot(data=ch, x="Avg_ROI", y="Channel")
plt.title("Average ROI by channel (campaigns using that channel)")
save("3_channel_roi")

# ---------- 4. Top / bottom campaigns ----------
cols = ["Campaign_ID", "Brand", "Campaign_Type", "Revenue", "Total_Cost", "ROI"]
print("\n=== 4. TOP 5 CAMPAIGNS BY ROI ===\n", df.nlargest(5, "ROI")[cols].to_string(index=False))
print("\n=== 4. BOTTOM 5 CAMPAIGNS BY ROI ===\n", df.nsmallest(5, "ROI")[cols].to_string(index=False))
plt.figure(figsize=(8, 4.5))
sns.histplot(df["ROI"].clip(upper=15), bins=60)
plt.axvline(0, color="red", linestyle="--"); plt.title("ROI distribution (red line = break-even)")
save("4_roi_distribution")

# ---------- 5. Relationships ----------
num = ["Impressions", "Clicks", "Leads", "Conversions", "Revenue", "Acquisition_Cost",
       "Total_Cost", "ROI", "Engagement_Score", "Duration"]
corr = df[num].corr()
print("\n=== 5. CORRELATION WITH REVENUE ===\n", corr["Revenue"].round(2).sort_values(ascending=False).to_string())
print("\n=== 5. CORRELATION WITH ROI ===\n", corr["ROI"].round(2).sort_values(ascending=False).to_string())
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
save("5_correlation_heatmap")

s = df.sample(4000, random_state=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.scatterplot(data=s, x="Conversions", y="Revenue", hue="Profit_Flag", alpha=.5, ax=ax[0]); ax[0].set_title("Conversions vs Revenue")
sns.scatterplot(data=s, x="Total_Cost", y="Revenue", hue="Profit_Flag", alpha=.5, ax=ax[1]); ax[1].set_title("Total cost vs Revenue")
save("6_spend_vs_revenue")

# ---------- 6. Audience, language, segment ----------
for col in ["Target_Audience", "Language", "Customer_Segment"]:
    t = df.groupby(col).agg(Avg_ROI=("ROI", "mean"), Profit_Rate=("Profit_Flag", "mean")).round(2)
    print(f"\n=== 6. {col.upper()} ===\n", t.to_string())
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["Target_Audience", "Language", "Customer_Segment"]):
    d = df.groupby(col)["ROI"].mean().sort_values()
    sns.barplot(x=d.values, y=d.index, ax=a); a.set_title(f"Average ROI by {col}")
save("7_audience_language_segment")

# ---------- 7. Trend over time ----------
m = df.dropna(subset=["Date"]).groupby(df["Date"].dt.to_period("M")).agg(Revenue=("Revenue", "sum"), Profit_Rate=("Profit_Flag", "mean"))
m.index = m.index.astype(str)
print("\n=== 7. MONTHLY TREND ===\n", m.round(2).to_string())
plt.figure(figsize=(11, 4.5))
sns.lineplot(x=m.index, y=m["Revenue"] / 1e9, marker="o")
plt.xticks(rotation=60); plt.ylabel("Revenue (billions)"); plt.title("Monthly revenue")
save("8_monthly_revenue")

print("\nSaved", len(os.listdir("eda_charts")), "charts in eda_charts/")
