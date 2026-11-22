"""make_sample.py - creates two small files for the app (the full 33 MB file is
too big to upload to GitHub through the web page):
  campaign_sample.csv   - random 15,000 campaigns, used by the charts
  campaign_extremes.csv - the 300 best and 300 worst campaigns by net profit (full data)
"""
import pandas as pd
df = pd.read_csv("marketing_features.csv")
df["Net_Profit"] = df["Revenue"] - df["Total_Cost"]

cols = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment", "Date",
        "Impressions", "Clicks", "Leads", "Conversions", "Revenue", "Acquisition_Cost",
        "Total_Cost", "Net_Profit", "ROI", "Engagement_Score", "Duration", "Profit_Flag",
        "Channel_Email", "Channel_Facebook", "Channel_Google", "Channel_Instagram",
        "Channel_WhatsApp", "Channel_YouTube"]
df[cols].sample(15000, random_state=42).to_csv("campaign_sample.csv", index=False)
print("Saved campaign_sample.csv (15,000 rows)")

keep = ["Campaign_ID", "Brand", "Campaign_Type", "Target_Audience", "Customer_Segment",
        "Channel_Used", "Conversions", "Revenue", "Total_Cost", "Net_Profit", "ROI"]
top = df.nlargest(300, "Net_Profit")[keep].assign(Group="Top")
bottom = df.nsmallest(300, "Net_Profit")[keep].assign(Group="Bottom")
pd.concat([top, bottom]).to_csv("campaign_extremes.csv", index=False)
print("Saved campaign_extremes.csv (300 best + 300 worst campaigns)")
