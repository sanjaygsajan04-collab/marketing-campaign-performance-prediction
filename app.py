"""
app.py - Project 3: Marketing Campaign Performance Prediction System  (version 2)
Pages (sidebar menu):
  1. Predict          - live prediction of Revenue and Profit/Loss
  2. What-if lab      - compare two scenarios + sensitivity curve + break-even
  3. Explore data     - filters, group-by, scatter, trend, top/bottom campaigns
  4. Model report     - metrics, feature importance, leakage explanation
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict_utils import CHANNELS, load_assets, predict

st.set_page_config(page_title="Marketing Campaign Performance Prediction System", page_icon="📈", layout="wide")

# ---------- look and feel: warm cream + berry, serif headings ----------
INK, BERRY, TEAL, CREAM, GOLD, PLUM = "#2B2330", "#C2185B", "#1B998B", "#FFF8F2", "#E9A23B", "#6A1B9A"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Poppins:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Poppins', sans-serif; color: {INK}; }}
.stApp {{ background: {CREAM}; }}
h1, h2, h3 {{ font-family: 'DM Serif Display', serif !important; color: {INK}; }}
section[data-testid="stSidebar"] {{ background: #2B2330; }}
section[data-testid="stSidebar"] * {{ color: #FBEFF3 !important; }}
.hero {{ background: linear-gradient(120deg, {BERRY} 0%, {PLUM} 100%); color: white;
        padding: 24px 32px; border-radius: 20px; margin-bottom: 16px; }}
.hero h1 {{ color: white !important; margin: 0; font-size: 1.8rem; }}
.hero p {{ margin: 6px 0 0 0; opacity: .92; }}
.card {{ background: white; border-radius: 16px; padding: 16px 18px; border: 1px solid #F0E3DA;
        box-shadow: 0 2px 10px rgba(43,35,48,.05); }}
.big {{ font-family: 'DM Serif Display', serif; font-size: 2rem; line-height: 1.1; }}
.label {{ font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; color: #8a7d86; }}
.pill-profit {{ background:#DFF5EE; color:#0b6b5f; padding:6px 16px; border-radius:99px; font-weight:600; }}
.pill-loss {{ background:#FCE4EC; color:#a3123f; padding:6px 16px; border-radius:99px; font-weight:600; }}
</style>
""", unsafe_allow_html=True)


def inr(v):
    """Format rupees the Indian way: lakh (L) and crore (Cr)."""
    s = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e7:
        return f"{s}₹{v/1e7:.2f} Cr"
    if v >= 1e5:
        return f"{s}₹{v/1e5:.2f} L"
    return f"{s}₹{v:,.0f}"


def card(label, value_html):
    return f'<div class="card"><div class="label">{label}</div><div class="big">{value_html}</div></div>'


def pill(is_profit):
    return '<span class="pill-profit">PROFIT</span>' if is_profit else '<span class="pill-loss">LOSS</span>'


@st.cache_resource
def get_assets():
    return load_assets()


@st.cache_data
def get_sample():
    return pd.read_csv("campaign_sample.csv", parse_dates=["Date"])


@st.cache_data
def get_extremes():
    return pd.read_csv("campaign_extremes.csv")


@st.cache_data
def get_metrics():
    return pd.read_csv("model_metrics.csv")


assets = get_assets()
config = assets[0]
CATS = config["cats"]

# ---------------------------------------------------------------
# SESSION STATE: remembers every widget value while you change pages
# ---------------------------------------------------------------
PRESETS = {
    "Typical campaign": dict(impressions=55000, clicks=3900, leads=1500, conversions=775, acq_cost=208.0, engagement=13.6, duration=17),
    "Efficient (low cost per conversion)": dict(impressions=70000, clicks=6000, leads=2600, conversions=1500, acq_cost=90.0, engagement=18.0, duration=20),
    "Risky (high cost per conversion)": dict(impressions=40000, clicks=2500, leads=900, conversions=400, acq_cost=900.0, engagement=9.0, duration=12),
    "Big reach, weak conversion": dict(impressions=95000, clicks=9000, leads=1200, conversions=250, acq_cost=350.0, engagement=11.0, duration=25),
}
DEFAULTS = dict(
    preset="Typical campaign", brand="Nykaa", ctype="Paid Ads", audience="Youth", segment="Youth",
    language="Hindi", channels=["Instagram"], knows_roi=False, roi_in=1.0,
    wi_impr=0, wi_clicks=0, wi_leads=0, wi_conv=20, wi_cost=-30, wi_sweep="Cost per conversion",
    ex_brand=CATS["Brand"], ex_type=CATS["Campaign_Type"], ex_segment=CATS["Customer_Segment"],
    ex_channel=CHANNELS, ex_outcome="All", ex_group="Brand", ex_metric="Average ROI",
    ex_x="Conversions", ex_y="Revenue", ex_rank="Top 10",
    **PRESETS["Typical campaign"])
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
    else:
        st.session_state[k] = st.session_state[k]   # keeps value when a page is not shown


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


def current_inputs():
    s = st.session_state
    return dict(Duration=s["duration"], Impressions=s["impressions"], Clicks=s["clicks"], Leads=s["leads"],
                Conversions=s["conversions"], Acquisition_Cost=s["acq_cost"], Engagement_Score=s["engagement"],
                Brand=s["brand"], Campaign_Type=s["ctype"], Target_Audience=s["audience"],
                Language=s["language"], Customer_Segment=s["segment"], Channels=s["channels"])


# ---------------------------------------------------------------
# SIDEBAR + HEADER
# ---------------------------------------------------------------
st.sidebar.markdown("## Campaign Analytics")
page = st.sidebar.radio("Go to", ["Predict", "What-if lab", "Explore data", "Model report"])
st.sidebar.markdown("---")
st.sidebar.caption("Marketing campaign analysis for Nykaa, Purplle and Tira. "
                   "166,665 campaigns, Jul 2024 - Jun 2025.")
st.markdown("""<div class="hero"><h1>Marketing Campaign Performance Prediction System</h1>
<p>Revenue and profitability prediction, scenario analysis and campaign insights for Nykaa, Purplle and Tira.</p></div>""",
            unsafe_allow_html=True)

# =====================================================================
# PAGE 1 - PREDICT  (updates live, no button needed)
# =====================================================================
if page == "Predict":
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Campaign details")
        st.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=apply_preset)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.selectbox("Brand", CATS["Brand"], key="brand")
            c2.selectbox("Campaign type", CATS["Campaign_Type"], key="ctype")
            c1.selectbox("Target audience", CATS["Target_Audience"], key="audience")
            c2.selectbox("Customer segment", CATS["Customer_Segment"], key="segment")
            c1.selectbox("Language", CATS["Language"], key="language")
            c2.slider("Duration (days)", 5, 30, key="duration")
            st.multiselect("Channels used (pick one or more)", CHANNELS, key="channels")
            n1, n2 = st.columns(2)
            n1.number_input("Impressions", 0, 200000, step=1000, key="impressions")
            n2.number_input("Clicks", 0, 30000, step=100, key="clicks")
            n1.number_input("Leads", 0, 20000, step=50, key="leads")
            n2.number_input("Conversions", 0, 15000, step=25, key="conversions")
            n1.number_input("Cost per conversion (₹)", 1.0, 20000.0, step=10.0, key="acq_cost")
            n2.number_input("Engagement score", 0.0, 40.0, step=0.5, key="engagement")
            st.checkbox("I already know the ROI (uses the high-accuracy revenue model)", key="knows_roi")
            if st.session_state["knows_roi"]:
                st.number_input("Known ROI (1.5 means a 150% return)", -1.0, 100.0, step=0.1, key="roi_in")

    inp = current_inputs()
    with right:
        st.subheader("Prediction")
        problems = []
        if inp["Clicks"] > inp["Impressions"]:
            problems.append("Clicks are higher than impressions.")
        if inp["Leads"] > inp["Clicks"]:
            problems.append("Leads are higher than clicks.")
        if inp["Conversions"] > inp["Leads"]:
            problems.append("Conversions are higher than leads.")
        if problems:
            st.warning("Please check the numbers: " + " ".join(problems) +
                       " The models were trained on campaigns where each funnel step is smaller than the one before.")

        known = st.session_state["roi_in"] if st.session_state["knows_roi"] else None
        res = predict(inp, assets, known_roi=known)

        k1, k2, k3 = st.columns(3)
        k1.markdown(card("Predicted revenue", inr(res["revenue"])), unsafe_allow_html=True)
        k2.markdown(card("Total cost", inr(res["total_cost"])), unsafe_allow_html=True)
        k3.markdown(card("Predicted outcome", f'<div style="margin-top:6px">{pill(res["is_profit"])}</div>'),
                    unsafe_allow_html=True)
        st.caption(
            f"Revenue minus cost gives an implied ROI of **{res['implied_roi']*100:.0f}%**. "
            + ("Revenue came from the model that was given your ROI. " if known is not None else
               "Revenue came from the model that uses campaign inputs only. ")
            + "The profit/loss class comes from a separate classifier, so the two can disagree "
              "when a campaign is close to break-even.")

        g1, g2 = st.columns(2)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=res["p_profit"] * 100, number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Chance of profit"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": TEAL if res["is_profit"] else BERRY},
                   "steps": [{"range": [0, 50], "color": "#FCE4EC"}, {"range": [50, 100], "color": "#DFF5EE"}]}))
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
        g1.plotly_chart(gauge)

        bars = go.Figure(go.Bar(x=["Predicted revenue", "Total cost"], y=[res["revenue"], res["total_cost"]],
                                marker_color=[TEAL, BERRY], text=[inr(res["revenue"]), inr(res["total_cost"])],
                                textposition="outside"))
        bars.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), title="Revenue vs cost",
                           yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        g2.plotly_chart(bars)

        funnel = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Leads", "Conversions"],
                                     x=[inp["Impressions"], inp["Clicks"], inp["Leads"], inp["Conversions"]],
                                     marker={"color": [GOLD, BERRY, PLUM, TEAL]}))
        funnel.update_layout(height=270, margin=dict(l=10, r=10, t=40, b=10), title="Your campaign funnel",
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(funnel)

        sample = get_sample()
        st.info(f"For comparison, the typical campaign in the data earns about **{inr(sample['Revenue'].median())}** "
                f"and {sample['Profit_Flag'].mean()*100:.0f}% of campaigns make a profit. "
                "Open **What-if lab** in the sidebar to test changes to this campaign.")

# =====================================================================
# PAGE 2 - WHAT-IF LAB
# =====================================================================
elif page == "What-if lab":
    base = current_inputs()
    st.subheader("What-if lab")
    st.caption("Scenario A is the campaign from the Predict page. Move the sliders to build Scenario B.")

    sl1, sl2 = st.columns([1, 1.6], gap="large")
    with sl1:
        with st.container(border=True):
            st.markdown("**Change Scenario B by...**")
            st.slider("Conversions (%)", -50, 100, key="wi_conv")
            st.slider("Cost per conversion (%)", -60, 100, key="wi_cost")
            st.slider("Leads (%)", -50, 100, key="wi_leads")
            st.slider("Clicks (%)", -50, 100, key="wi_clicks")
            st.slider("Impressions (%)", -50, 100, key="wi_impr")

    def scale(v, pct):
        return v * (1 + pct / 100)

    scen_b = dict(base,
                  Conversions=int(scale(base["Conversions"], st.session_state["wi_conv"])),
                  Acquisition_Cost=scale(base["Acquisition_Cost"], st.session_state["wi_cost"]),
                  Leads=int(scale(base["Leads"], st.session_state["wi_leads"])),
                  Clicks=int(scale(base["Clicks"], st.session_state["wi_clicks"])),
                  Impressions=int(scale(base["Impressions"], st.session_state["wi_impr"])))
    ra, rb = predict(base, assets), predict(scen_b, assets)

    with sl2:
        table = pd.DataFrame({
            "": ["Conversions", "Cost per conversion", "Predicted revenue", "Total cost", "Implied ROI",
                 "Chance of profit", "Outcome"],
            "Scenario A": [f"{base['Conversions']:,}", inr(base["Acquisition_Cost"]), inr(ra["revenue"]),
                           inr(ra["total_cost"]), f"{ra['implied_roi']*100:.0f}%", f"{ra['p_profit']*100:.0f}%",
                           "Profit" if ra["is_profit"] else "Loss"],
            "Scenario B": [f"{scen_b['Conversions']:,}", inr(scen_b["Acquisition_Cost"]), inr(rb["revenue"]),
                           inr(rb["total_cost"]), f"{rb['implied_roi']*100:.0f}%", f"{rb['p_profit']*100:.0f}%",
                           "Profit" if rb["is_profit"] else "Loss"]})
        st.dataframe(table, hide_index=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Revenue change", inr(rb["revenue"] - ra["revenue"]),
                  f"{(rb['revenue']/ra['revenue']-1)*100:.0f}%" if ra["revenue"] else None)
        m2.metric("Cost change", inr(rb["total_cost"] - ra["total_cost"]))
        m3.metric("Chance of profit", f"{rb['p_profit']*100:.0f}%", f"{(rb['p_profit']-ra['p_profit'])*100:+.0f} pts")

    fig = go.Figure()
    fig.add_bar(name="Predicted revenue", x=["Scenario A", "Scenario B"], y=[ra["revenue"], rb["revenue"]], marker_color=TEAL)
    fig.add_bar(name="Total cost", x=["Scenario A", "Scenario B"], y=[ra["total_cost"], rb["total_cost"]], marker_color=BERRY)
    fig.update_layout(barmode="group", height=320, title="Scenario A vs Scenario B",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)

    # ---- sensitivity curve: change ONE input across a range, keep the rest ----
    st.markdown("### Sensitivity: change one number and watch the result")
    st.selectbox("What to vary", ["Cost per conversion", "Conversions", "Leads"], key="wi_sweep")
    what = st.session_state["wi_sweep"]
    field = {"Cost per conversion": "Acquisition_Cost", "Conversions": "Conversions", "Leads": "Leads"}[what]
    xs = {"Cost per conversion": np.linspace(25, 1500, 25), "Conversions": np.linspace(100, 4000, 25),
          "Leads": np.linspace(200, 6000, 25)}[what]
    rows = []
    for x in xs:
        r = predict(dict(base, **{field: float(x) if field == "Acquisition_Cost" else int(x)}), assets)
        rows.append(dict(x=float(x), Revenue=r["revenue"], Cost=r["total_cost"], p=r["p_profit"]))
    sw = pd.DataFrame(rows)

    s1, s2 = st.columns(2)
    f1 = go.Figure()
    f1.add_scatter(x=sw["x"], y=sw["Revenue"], name="Predicted revenue", line=dict(color=TEAL, width=3))
    f1.add_scatter(x=sw["x"], y=sw["Cost"], name="Total cost", line=dict(color=BERRY, width=3))
    f1.update_layout(height=330, title="Revenue and cost", xaxis_title=what, paper_bgcolor="rgba(0,0,0,0)")
    s1.plotly_chart(f1)
    f2 = go.Figure(go.Scatter(x=sw["x"], y=sw["p"] * 100, mode="lines+markers", line=dict(color=PLUM, width=3)))
    f2.add_hline(y=50, line_dash="dash", line_color="grey")
    f2.update_layout(height=330, title="Chance of profit (%)", xaxis_title=what, yaxis_range=[0, 100],
                     paper_bgcolor="rgba(0,0,0,0)")
    s2.plotly_chart(f2)

    below = sw[sw["p"] < 0.5]
    above = sw[sw["p"] >= 0.5]
    if what == "Cost per conversion" and len(below) and len(above):
        st.success(f"Break-even: with the rest of your campaign unchanged, it stops being a likely profit "
                   f"once cost per conversion passes roughly **₹{above['x'].max():,.0f}**.")
    elif len(below) == 0:
        st.success(f"Across this whole range of {what.lower()}, the campaign stays on the profit side.")
    elif len(above) == 0:
        st.error(f"Across this whole range of {what.lower()}, the campaign is predicted as a loss.")

# =====================================================================
# PAGE 3 - EXPLORE DATA
# =====================================================================
elif page == "Explore data":
    s = get_sample()
    st.subheader("Explore the campaign data")
    st.caption("Charts use a random sample of 15,000 of the 166,665 campaigns. Change the filters and every chart updates.")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        f1.multiselect("Brand", CATS["Brand"], key="ex_brand")
        f2.multiselect("Campaign type", CATS["Campaign_Type"], key="ex_type")
        f3.multiselect("Customer segment", CATS["Customer_Segment"], key="ex_segment")
        g1, g2 = st.columns([2, 1])
        g1.multiselect("Channels (campaign uses at least one of these)", CHANNELS, key="ex_channel")
        g2.radio("Outcome", ["All", "Profit", "Loss"], horizontal=True, key="ex_outcome")

    d = s[s["Brand"].isin(st.session_state["ex_brand"]) & s["Campaign_Type"].isin(st.session_state["ex_type"])
          & s["Customer_Segment"].isin(st.session_state["ex_segment"])]
    if st.session_state["ex_channel"]:
        d = d[d[[f"Channel_{c}" for c in st.session_state["ex_channel"]]].sum(axis=1) > 0]
    if st.session_state["ex_outcome"] == "Profit":
        d = d[d["Profit_Flag"] == 1]
    elif st.session_state["ex_outcome"] == "Loss":
        d = d[d["Profit_Flag"] == 0]

    if d.empty:
        st.warning("No campaigns match these filters. Add more options above.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(card("Campaigns", f"{len(d):,}"), unsafe_allow_html=True)
        k2.markdown(card("Avg revenue", inr(d["Revenue"].mean())), unsafe_allow_html=True)
        k3.markdown(card("Avg ROI", f"{d['ROI'].mean()*100:.0f}%"), unsafe_allow_html=True)
        k4.markdown(card("In profit", f"{d['Profit_Flag'].mean()*100:.0f}%"), unsafe_allow_html=True)

        st.markdown("#### Compare groups")
        c1, c2 = st.columns([1, 3])
        c1.selectbox("Group by", ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"], key="ex_group")
        c1.selectbox("Measure", ["Average ROI", "Average revenue", "Profit rate (%)", "Average cost per conversion"], key="ex_metric")
        grp, met = st.session_state["ex_group"], st.session_state["ex_metric"]
        agg = {"Average ROI": ("ROI", "mean"), "Average revenue": ("Revenue", "mean"),
               "Profit rate (%)": ("Profit_Flag", "mean"), "Average cost per conversion": ("Acquisition_Cost", "mean")}[met]
        gd = d.groupby(grp, as_index=False).agg(Value=agg)
        if met == "Profit rate (%)":
            gd["Value"] = gd["Value"] * 100
        fig = px.bar(gd.sort_values("Value"), x="Value", y=grp, orientation="h", color="Value",
                     color_continuous_scale=["#F8BBD0", BERRY, PLUM], title=f"{met} by {grp}")
        fig.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", height=340)
        c2.plotly_chart(fig)

        st.markdown("#### Relationships")
        c1, c2 = st.columns([1, 3])
        nums = ["Conversions", "Leads", "Clicks", "Impressions", "Revenue", "Total_Cost", "Acquisition_Cost",
                "Engagement_Score", "ROI"]
        c1.selectbox("X axis", nums, key="ex_x")
        c1.selectbox("Y axis", nums, key="ex_y")
        sc = d.sample(min(3000, len(d)), random_state=1).assign(
            Outcome=lambda t: t["Profit_Flag"].map({1: "Profit", 0: "Loss"}))
        fig = px.scatter(sc, x=st.session_state["ex_x"], y=st.session_state["ex_y"], color="Outcome", opacity=.55,
                         color_discrete_map={"Profit": TEAL, "Loss": BERRY},
                         title=f"{st.session_state['ex_y']} vs {st.session_state['ex_x']}")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
        c2.plotly_chart(fig)

        st.markdown("#### Trend over time")
        m = (d.dropna(subset=["Date"]).assign(Month=lambda t: t["Date"].dt.to_period("M").astype(str))
             .groupby(["Month", "Brand"], as_index=False)["Revenue"].mean())
        fig = px.line(m, x="Month", y="Revenue", color="Brand", markers=True,
                      color_discrete_sequence=[BERRY, PLUM, TEAL], title="Average revenue per campaign by month")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=340)
        st.plotly_chart(fig)

    st.markdown("#### Top and bottom campaigns (ranked by net profit = revenue - cost)")
    st.caption("Taken from all 166,665 campaigns (not the sample), then narrowed by the Brand, Campaign type and Segment filters above.")
    ex = get_extremes()
    ex = ex[ex["Brand"].isin(st.session_state["ex_brand"]) & ex["Campaign_Type"].isin(st.session_state["ex_type"])
            & ex["Customer_Segment"].isin(st.session_state["ex_segment"])]
    st.radio("Show", ["Top 10", "Bottom 10"], horizontal=True, key="ex_rank")
    if st.session_state["ex_rank"] == "Top 10":
        show = ex[ex["Group"] == "Top"].sort_values("Net_Profit", ascending=False).head(10)
    else:
        show = ex[ex["Group"] == "Bottom"].sort_values("Net_Profit").head(10)
    if show.empty:
        st.info("No top/bottom campaigns match the current filters.")
    else:
        show = show.drop(columns="Group").assign(
            Revenue=show["Revenue"].map(inr), Total_Cost=show["Total_Cost"].map(inr),
            Net_Profit=show["Net_Profit"].map(inr), ROI=(show["ROI"] * 100).round(0).astype(int).astype(str) + "%")
        st.dataframe(show, hide_index=True)

    st.markdown("""
**Key findings and recommendations**
- **Conversions are the engine of revenue.** Revenue rises almost in a straight line with conversions, so effort should go into turning leads into buyers.
- **Cost per conversion decides profit.** Loss-making campaigns have a high cost per conversion. Keep it near the typical ₹200 before scaling spend.
- **Brand, channel, language and audience hardly matter here.** Profit rates are about 75-76% everywhere, so changing channel alone will not rescue a weak campaign.
- **About 1 in 4 campaigns loses money.** Check cost per conversion before launch and use the predictor to screen risky plans.
""")

# =====================================================================
# PAGE 4 - MODEL REPORT
# =====================================================================
else:
    st.subheader("How the models were built and judged")
    mt = get_metrics()
    reg = mt[mt["Task"].str.startswith("Revenue")].dropna(axis=1, how="all")
    clf = mt[mt["Task"] == "Profit/Loss"].dropna(axis=1, how="all")
    st.markdown("**Regression - predicting Revenue** (R² closer to 1 and lower RMSE/MAE/MSE is better)")
    st.dataframe(reg, hide_index=True)
    st.markdown("**Classification - predicting Profit or Loss**")
    st.dataframe(clf, hide_index=True)

    i1, i2 = st.columns(2)
    i1.image("eda_charts/9_feature_importance.png", caption="What each model relies on")
    i2.image("eda_charts/10_confusion_matrix.png", caption="Profit/Loss predictions on unseen data")

    st.markdown("""
**Why the results are below 0.95 and what we did about it**
- *Revenue (inputs only), R² about 0.76.* In this dataset revenue per conversion varies randomly between ₹200 and ₹800, which no model can predict. This is the honest forecast you can use before a campaign runs.
- *Revenue (with ROI), R² about 0.997.* ROI is calculated from revenue, so this model reaches the 0.95 target only because it already knows part of the answer. Use it when ROI is known.
- *Profit/Loss, accuracy about 0.91.* The label comes from ROI, so **ROI and Revenue are left out** of the features to avoid data leakage. Adding Revenue would push accuracy near 100% but would be cheating.
- Data was split 80% training and 20% testing, and parameters were tuned with randomised search and 3-fold cross-validation.
""")
