# Multi-Brand Marketing Campaign Performance Analysis and Prediction

An end-to-end machine learning project on 166,665 marketing campaigns from **Nykaa, Purplle and Tira**.
It cleans the data, explores what drives performance, predicts **campaign revenue** (regression) and
**profit or loss** (classification), and serves both models in an interactive **Streamlit** app.

GUVI x HCL Capstone Project 3 | Domain: Marketing Analytics and Business Intelligence

## What the project does
1. **Data collection** - three brand CSVs combined with Pandas; a `Brand` column is added.
2. **Cleaning** - nulls, duplicates, data types, and **ROI validation** (hidden rule found: `ROI = (Revenue - Cost) / Cost`, where `Cost = Acquisition_Cost x Conversions`).
3. **Feature engineering** - `Profit_Flag` from ROI, **multi-label encoding** of `Channel_Used`, one-hot encoding, rates such as CTR.
4. **EDA** - brands, campaign types, channels, audiences, top and bottom campaigns, relationships, trends.
5. **Models** - Linear/Logistic Regression, Decision Tree, Random Forest, tuned with randomised search and cross-validation.
6. **Evaluation** - RMSE, MAE, MSE, R2, Accuracy, Precision, Recall, F1, with **no data leakage**.
7. **Streamlit app** - live predictions, what-if scenarios, data explorer, model report.

## Results
| Task | Best model | Result |
|---|---|---|
| Revenue (campaign inputs only) | Random Forest (tuned) | R2 0.758, RMSE about Rs 239k, MAE about Rs 158k |
| Revenue (ROI also known) | Random Forest | R2 0.997, RMSE about Rs 28k |
| Profit / Loss | Random Forest | Accuracy 0.910, Precision 0.925, Recall 0.959, F1 0.942 |

**Why some results are below the 0.95 target:** in this dataset revenue per conversion is random between Rs 200 and Rs 800,
which caps an honest revenue model at about R2 0.78. The Profit/Loss label is built from ROI, which is built from Revenue,
so ROI and Revenue are excluded from the classifier to avoid data leakage. Full explanation in `docs/Project_Documentation.docx`.

## Key insights
- **Conversions drive revenue** (correlation 0.85).
- **Cost per conversion decides profit.** For a typical campaign the chance of profit falls below 50% once it passes roughly Rs 330.
- About **24% of campaigns lose money**.
- Brand, channel, language, audience and duration have very little effect.

## Project structure
```
clean_data.py          cleaning, ROI validation, feature engineering
eda.py                 EDA tables and charts  ->  eda_charts/
train_models.py        training, tuning, evaluation, saves models
predict_utils.py       applies the training preprocessing to new inputs
make_sample.py         creates the small data files used by the app
app.py                 Streamlit application
model_metrics.csv      all evaluation metrics
app_config.json        column list and category options for the app
*.joblib               saved models (revenue, revenue with ROI, profit)
campaign_sample.csv    15,000-row sample for the app charts
campaign_extremes.csv  300 best and 300 worst campaigns
data/                  raw brand CSVs and zipped cleaned / feature-engineered datasets
docs/                  Project_Documentation.docx (full report)
eda_charts/            charts used in the report
```

## How to run
```bash
pip install -r requirements.txt

# 1. put the three raw brand CSVs in the project folder, then:
python clean_data.py        # creates marketing_clean.csv and marketing_features.csv
python eda.py               # EDA output and charts
python train_models.py      # trains and saves the models (3-6 minutes)
python make_sample.py       # small files for the app

# 2. launch the app
streamlit run app.py
```
The cleaned and feature-engineered datasets are in `data/` as zip files; unzip them into the project folder
to skip step 1.

## Streamlit app pages
- **Predict** - enter campaign details; see predicted revenue, cost, profit/loss, chance of profit and the funnel.
- **What-if lab** - compare two scenarios and see a sensitivity curve with the break-even cost per conversion.
- **Explore data** - filters, group-by, scatter plot, trend, top and bottom campaigns.
- **Model report** - metrics, feature importance, confusion matrix and the leakage explanation.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Joblib.
