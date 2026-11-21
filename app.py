"""
Multi-Unit Hospitality: Daily Close & Demand Forecast  (pitch demo)

Run:  pip install -r requirements.txt && streamlit run app.py
All data is synthetic and generated on the fly.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Daily Close & Demand Forecast",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ palette
BG, CARD, BORDER = "#0B1220", "#121B30", "#1F2B47"
TEXT, MUTED = "#E6EDF7", "#8EA0BF"
TEAL, BLUE, AMBER, ROSE, GREEN = "#2DD4BF", "#60A5FA", "#FBBF24", "#FB7185", "#34D399"

SUSPENSE = "9999 - Suspense (Unmapped)"
CARD_FEE_RATE = 0.024
PRIME_TARGET = 62.0

CSS = """
<style>
#MainMenu, footer, .stAppDeployButton, [data-testid="stToolbar"] {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}

.hero {
  background: radial-gradient(1200px 300px at 85% -40%, rgba(45,212,191,.28), transparent 60%),
              linear-gradient(135deg, #10203d 0%, #0e1a33 55%, #0B1220 100%);
  border: 1px solid #1F2B47; border-radius: 18px; padding: 26px 30px 22px 30px; margin-bottom: 18px;
}
.hero .eyebrow {color:#2DD4BF; font-size:12px; letter-spacing:.14em; text-transform:uppercase; font-weight:600;}
.hero h1 {margin:4px 0 6px 0; font-size:30px; line-height:1.15; font-weight:700; color:#E6EDF7; padding:0;}
.hero p {color:#8EA0BF; margin:0 0 18px 0; font-size:15px; max-width:860px;}
.badge {display:inline-block; font-size:11px; padding:3px 9px; border-radius:999px; border:1px solid #2b3a5f;
        color:#8EA0BF; margin-left:8px; vertical-align:middle; letter-spacing:.04em;}

.pipe {display:flex; gap:10px; flex-wrap:wrap;}
.step {flex:1 1 150px; background:rgba(18,27,48,.75); border:1px solid #1F2B47; border-radius:12px; padding:11px 14px;}
.step .n {font-size:11px; color:#8EA0BF; text-transform:uppercase; letter-spacing:.08em;}
.step .t {font-size:15px; font-weight:600; color:#E6EDF7; margin-top:2px;}
.step .s {font-size:12px; margin-top:2px;}
.step.ok {border-color: rgba(52,211,153,.45);} .step.ok .s {color:#34D399;}
.step.warn {border-color: rgba(251,191,36,.55);} .step.warn .s {color:#FBBF24;}
.step.bad {border-color: rgba(251,113,133,.6);} .step.bad .s {color:#FB7185;}

.kpi {background:#121B30; border:1px solid #1F2B47; border-radius:14px; padding:16px 18px; height:100%;}
.kpi .l {font-size:12px; color:#8EA0BF; text-transform:uppercase; letter-spacing:.08em;}
.kpi .v {font-size:30px; font-weight:700; color:#E6EDF7; margin:4px 0 2px 0; line-height:1.1;}
.kpi .d {font-size:12.5px; color:#8EA0BF;}
.kpi .d.good {color:#34D399;} .kpi .d.bad {color:#FB7185;} .kpi .d.warn {color:#FBBF24;}
.kpi.accent-bad {border-color: rgba(251,113,133,.55); box-shadow: 0 0 0 1px rgba(251,113,133,.12) inset;}
.kpi.accent-good {border-color: rgba(52,211,153,.35);}

.sec {margin: 26px 0 8px 0;}
.sec .h {font-size:18px; font-weight:650; color:#E6EDF7;}
.sec .s {font-size:13px; color:#8EA0BF; margin-top:1px;}

.ins {background:#121B30; border:1px solid #1F2B47; border-left:3px solid #2DD4BF; border-radius:10px;
      padding:10px 14px; margin-bottom:8px; font-size:14px; color:#E6EDF7;}
.ins.warn {border-left-color:#FBBF24;} .ins.bad {border-left-color:#FB7185;} .ins.good {border-left-color:#34D399;}

.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600;}
.pill.ok {background:rgba(52,211,153,.14); color:#34D399;}
.pill.warn {background:rgba(251,191,36,.14); color:#FBBF24;}
.pill.bad {background:rgba(251,113,133,.15); color:#FB7185;}
.pill.todo {background:rgba(142,160,191,.14); color:#8EA0BF;}

.chk {display:flex; align-items:center; gap:14px; background:#121B30; border:1px solid #1F2B47; border-radius:12px;
      padding:12px 16px; margin-bottom:8px;}
.chk .i {width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center;
         font-weight:700; font-size:14px; flex:0 0 30px;}
.chk .i.ok {background:rgba(52,211,153,.16); color:#34D399;} .chk .i.warn {background:rgba(251,191,36,.16); color:#FBBF24;}
.chk .i.bad {background:rgba(251,113,133,.16); color:#FB7185;} .chk .i.todo {background:rgba(142,160,191,.16); color:#8EA0BF;}
.chk .b {flex:1;} .chk .t {font-weight:600; color:#E6EDF7; font-size:14.5px;} .chk .d {color:#8EA0BF; font-size:12.5px;}

.card2 {background:#121B30; border:1px solid #1F2B47; border-radius:14px; padding:16px 18px; height:100%;}
.card2 .t {font-weight:650; color:#E6EDF7; margin-bottom:4px;} .card2 .d {color:#8EA0BF; font-size:13.5px;}

.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid #1F2B47;}
.stTabs [data-baseweb="tab"] {height: 44px; padding: 0 16px; border-radius: 10px 10px 0 0; color:#8EA0BF;}
.stTabs [aria-selected="true"] {color:#E6EDF7 !important;}
[data-testid="stSidebar"] {border-right: 1px solid #1F2B47;}
div[data-testid="stDataFrame"] {border: 1px solid #1F2B47; border-radius: 12px; overflow: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def usd(x: float, dec: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{dec}f}"


def section(title: str, sub: str = "") -> None:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sec"><div class="h">{title}</div>{sub_html}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str = "", tone: str = "", accent: str = "") -> str:
    d = f'<div class="d {tone}">{delta}</div>' if delta else ""
    return f'<div class="kpi {accent}"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False, linecolor=BORDER)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ------------------------------------------------------------------ synthetic data
LOCATIONS = [
    dict(location="Downtown Bistro", concept="Casual Dining", segment="F&B", base=240, check=34.0, cps=28, cash=0.18, card=0.82, room=0.00),
    dict(location="Harbor Steakhouse", concept="Fine Dining", segment="F&B", base=145, check=88.0, cps=18, cash=0.10, card=0.90, room=0.00),
    dict(location="Market Burger Bar", concept="Fast Casual", segment="F&B", base=390, check=21.0, cps=55, cash=0.30, card=0.70, room=0.00),
    dict(location="Boutique Hotel Lobby & Bar", concept="Hotel F&B", segment="F&B", base=195, check=42.0, cps=28, cash=0.08, card=0.62, room=0.30),
    dict(location="Hotel Gift Shop", concept="Hotel Retail", segment="Retail", base=260, check=27.0, cps=70, cash=0.22, card=0.60, room=0.18),
]
CATEGORIES = {
    "F&B": [("Food - Entrees", 0.56), ("Beverage - Alcohol", 0.24), ("Beverage - Non-Alc", 0.08), ("Private Events & Catering", 0.12)],
    "Retail": [("Retail - Merchandise", 0.78), ("Retail - Packaged Food", 0.22)],
}


@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)
    end_date = datetime.today().date() - timedelta(days=1)
    dates = [end_date - timedelta(days=i) for i in range(59, -1, -1)]
    ops_rows, sales_rows = [], []
    for d in dates:
        dow = d.weekday()
        weekend = 1.38 if dow in (4, 5) else (1.12 if dow == 6 else 0.90)
        rain = bool(rng.choice([0, 1], p=[0.8, 0.2]))
        weather = 0.88 if rain else 1.02
        for loc in LOCATIONS:
            covers = int(loc["base"] * weekend * weather * rng.uniform(0.92, 1.08))
            gross_target = covers * loc["check"]
            labor_pct = rng.uniform(0.26, 0.33) if loc["segment"] == "F&B" else rng.uniform(0.14, 0.19)
            cogs_pct = rng.uniform(0.27, 0.32) if loc["segment"] == "F&B" else rng.uniform(0.38, 0.46)
            # Intentional anomalies on the latest day so the audit story is visible.
            if d == end_date and loc["location"] == "Harbor Steakhouse":
                variance = -215.50
            elif d == end_date and loc["location"] == "Market Burger Bar":
                variance = -18.25
            else:
                variance = round(float(rng.normal(0, 8.0)), 2)
            gross = 0.0
            for cat, share in CATEGORIES[loc["segment"]]:
                amt = round(gross_target * share, 2)
                gross += amt
                sales_rows.append(dict(date=d, location=loc["location"], concept=loc["concept"],
                                       segment=loc["segment"], pos_category=cat, gross_sales=amt))
            ops_rows.append(dict(
                date=d, location=loc["location"], concept=loc["concept"], segment=loc["segment"],
                covers=covers, gross_sales=round(gross, 2),
                labor_cost=round(gross * labor_pct, 2), cogs=round(gross * cogs_pct, 2),
                variance=variance, rain="Rain" if rain else "Clear",
                cash_pct=loc["cash"], card_pct=loc["card"], room_pct=loc["room"],
            ))
    return pd.DataFrame(ops_rows), pd.DataFrame(sales_rows)


ops_all, sales_all = generate_data()

DEFAULT_RULES = pd.DataFrame([
    {"pos_category": "Food - Entrees", "gl_account": "4010 - Food Revenue", "tax_rule": "Standard F&B", "tax_rate": 7.0, "active": True},
    {"pos_category": "Beverage - Alcohol", "gl_account": "4020 - Liquor & Wine Revenue", "tax_rule": "Alcohol Tax", "tax_rate": 9.0, "active": True},
    {"pos_category": "Beverage - Non-Alc", "gl_account": "4015 - Non-Alc Beverage Revenue", "tax_rule": "Standard F&B", "tax_rate": 7.0, "active": True},
    {"pos_category": "Private Events & Catering", "gl_account": "4050 - Banquet & Catering Revenue", "tax_rule": "Event Taxable", "tax_rate": 7.5, "active": True},
    {"pos_category": "Retail - Merchandise", "gl_account": "4110 - Retail Merchandise Revenue", "tax_rule": "Standard Retail", "tax_rate": 8.0, "active": True},
    {"pos_category": "Retail - Packaged Food", "gl_account": "4115 - Retail Packaged Food Revenue", "tax_rule": "Standard F&B", "tax_rate": 7.0, "active": True},
])
if "gl_rules" not in st.session_state:
    st.session_state.gl_rules = DEFAULT_RULES.copy()


def active_rules(rules: pd.DataFrame) -> pd.DataFrame:
    r = rules.copy()
    r = r[r["active"].fillna(False).astype(bool) & r["gl_account"].notna() & r["pos_category"].notna()]
    r = r[r["gl_account"].astype(str).str.strip() != ""]
    return r.drop_duplicates("pos_category").set_index("pos_category")


def build_journal(sales_day: pd.DataFrame, ops_day: pd.DataFrame, rules: pd.DataFrame):
    """Balanced double-entry journal for one business day. Returns (lines_df, unmapped_categories)."""
    r = active_rules(rules)
    s = sales_day.copy()
    s["gl_account"] = s["pos_category"].map(r["gl_account"]).fillna(SUSPENSE)
    unmapped = sorted(s.loc[s["gl_account"] == SUSPENSE, "pos_category"].unique())
    s["tax"] = (s["gross_sales"] * s["pos_category"].map(r["tax_rate"]).fillna(0.0) / 100).round(2)

    loc_tax = s.groupby("location")["tax"].sum()
    o = ops_day.set_index("location")
    receipts = (o["gross_sales"] + loc_tax.reindex(o.index).fillna(0)).round(2)
    cash_exp = (receipts * o["cash_pct"]).round(2)
    card = (receipts * o["card_pct"]).round(2)
    room = (receipts - cash_exp - card).round(2)
    fees = (card * CARD_FEE_RATE).round(2)
    variance = o["variance"]

    lines = []
    cash_actual = round(float((cash_exp + variance).sum()), 2)
    card_net = round(float((card - fees).sum()), 2)
    fee_total = round(float(fees.sum()), 2)
    room_total = round(float(room.sum()), 2)
    lines.append(("1110 - Cash Clearing (Bank Deposit)", "Actual cash deposited", cash_actual, 0.0))
    lines.append(("1120 - Card Processor Clearing", "Card sales net of fees", card_net, 0.0))
    lines.append(("6150 - Card Processing Fees", f"{CARD_FEE_RATE:.1%} of card volume", fee_total, 0.0))
    if room_total:
        lines.append(("1130 - Room Charge Receivable (PMS)", "Charged to guest folios", room_total, 0.0))
    net_short = round(float(-variance.sum()), 2)
    if net_short > 0:
        lines.append(("6990 - Cash Over/Short", "Net cash shortage", net_short, 0.0))
    elif net_short < 0:
        lines.append(("6990 - Cash Over/Short", "Net cash overage", 0.0, -net_short))
    for acct, amt in s.groupby("gl_account")["gross_sales"].sum().round(2).items():
        lines.append((acct, "Revenue by POS category", 0.0, float(amt)))
    lines.append(("2150 - Sales Tax Payable", "Tax collected per mapping rules", 0.0, round(float(s["tax"].sum()), 2)))

    df = pd.DataFrame(lines, columns=["GL Account", "Description", "Debit", "Credit"])
    return df, unmapped


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ◆ Daily Close Studio")
    st.caption("Synthetic demo data · 5 units · 60 days")
    all_locs = list(ops_all["location"].unique())
    selected = st.multiselect("Locations", options=all_locs, default=all_locs)
    threshold = st.slider("Over/short tolerance ($)", 10, 200, 50, 5,
                          help="Variances above this are routed to the exception queue.")
    gl_target = st.selectbox("Target ledger", ["QuickBooks Online", "Business Central", "Zoho Books"],
                             help="Where the approved journal is posted in production.")
    st.divider()
    st.caption("Every number on screen is computed from the synthetic POS feed and the mapping rules, "
               "so editing a rule changes the journal live.")

if not selected:
    st.warning("Select at least one location in the sidebar.")
    st.stop()

ops = ops_all[ops_all["location"].isin(selected)].copy()
sales = sales_all[sales_all["location"].isin(selected)].copy()
latest = ops["date"].max()
prior = latest - timedelta(days=7)
ops_day = ops[ops["date"] == latest]
ops_prior = ops[ops["date"] == prior]
sales_day = sales[sales["date"] == latest]

journal, unmapped = build_journal(sales_day, ops_day, st.session_state.gl_rules)
j_debit, j_credit = round(journal["Debit"].sum(), 2), round(journal["Credit"].sum(), 2)
balanced = abs(j_debit - j_credit) < 0.005

# exception candidates (computed before the tabs so the hero can show them)
cash_exp_loc = {}
for _, row in ops_day.iterrows():
    tax = 0.0
    r_ = active_rules(st.session_state.gl_rules)
    sd = sales_day[sales_day["location"] == row["location"]]
    tax = (sd["gross_sales"] * sd["pos_category"].map(r_["tax_rate"]).fillna(0) / 100).round(2).sum()
    cash_exp_loc[row["location"]] = round((row["gross_sales"] + tax) * row["cash_pct"], 2)
exc_base = ops_day[ops_day["variance"].abs() > threshold][["location", "variance"]].copy()
n_exc = len(exc_base)

# ------------------------------------------------------------------ hero
n_loc = len(selected)
n_cat = sales_day["pos_category"].nunique()
mapped_n = n_cat - len(unmapped)
posted = st.session_state.get("posted_key") == f"{latest}|{'|'.join(sorted(selected))}"


def step(n, title, sub, tone):
    return f'<div class="step {tone}"><div class="n">{n}</div><div class="t">{title}</div><div class="s">{sub}</div></div>'


pipe = "".join([
    step("01", "Ingest", f"{n_loc}/{n_loc} POS feeds received", "ok"),
    step("02", "Validate", "Totals foot to tenders" if balanced else "Journal out of balance", "ok" if balanced else "bad"),
    step("03", "Map to GL", f"{mapped_n}/{n_cat} categories mapped" if not unmapped else f"{len(unmapped)} to suspense", "ok" if not unmapped else "bad"),
    step("04", "Reconcile", f"{n_exc} exception{'s' if n_exc != 1 else ''} to review" if n_exc else "No exceptions", "warn" if n_exc else "ok"),
    step("05", "Post", f"Posted to {gl_target}" if posted else f"Ready for {gl_target}", "ok" if posted else "warn"),
])
st.markdown(
    f'<div class="hero"><div class="eyebrow">Multi-unit hospitality &amp; retail</div>'
    f'<h1>Daily close &amp; demand forecast<span class="badge">DEMO · SYNTHETIC DATA</span></h1>'
    f'<p>POS sales flow to the general ledger without re-keying. Cash and card variances are flagged the morning after, '
    f'F&amp;B and retail close on one process, and the next seven days of covers and labor are forecast.</p>'
    f'<div class="pipe">{pipe}</div></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Morning flash", "Over / short audit", "GL mapping & journal",
    "Close checklist", "7-day forecast", "Impact",
])

# ================================================================== TAB 1
with tab1:
    day_sales, day_prior = ops_day["gross_sales"].sum(), ops_prior["gross_sales"].sum()
    day_prime = ops_day["labor_cost"].sum() + ops_day["cogs"].sum()
    prime_pct = day_prime / day_sales * 100 if day_sales else 0
    prior_prime_pct = ((ops_prior["labor_cost"].sum() + ops_prior["cogs"].sum()) / day_prior * 100) if day_prior else 0
    covers, covers_prior = int(ops_day["covers"].sum()), int(ops_prior["covers"].sum())
    net_var = ops_day["variance"].sum()
    wd = latest.strftime("%A")

    sales_chg = (day_sales / day_prior - 1) * 100 if day_prior else 0
    cov_chg = (covers / covers_prior - 1) * 100 if covers_prior else 0
    prime_chg = prime_pct - prior_prime_pct

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Gross sales", usd(day_sales), f"{'▲' if sales_chg >= 0 else '▼'} {abs(sales_chg):.1f}% vs last {wd}",
                    "good" if sales_chg >= 0 else "bad"), unsafe_allow_html=True)
    c2.markdown(kpi("Prime cost %", f"{prime_pct:.1f}%",
                    f"{'▼' if prime_chg <= 0 else '▲'} {abs(prime_chg):.1f} pts vs last {wd} · target {PRIME_TARGET:.0f}%",
                    "good" if prime_pct <= PRIME_TARGET else "warn"), unsafe_allow_html=True)
    c3.markdown(kpi("Covers + transactions", f"{covers:,}", f"{'▲' if cov_chg >= 0 else '▼'} {abs(cov_chg):.1f}% vs last {wd}",
                    "good" if cov_chg >= 0 else "bad"), unsafe_allow_html=True)
    c4.markdown(kpi("Net cash over / short", usd(net_var, 2),
                    f"{n_exc} unit{'s' if n_exc != 1 else ''} over ${threshold} tolerance" if n_exc else "All within tolerance",
                    "bad" if n_exc else "good", "accent-bad" if n_exc else "accent-good"), unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        section("Sales, last 14 days", "F&B and retail on one view")
        t14 = ops[ops["date"] > latest - timedelta(days=14)].groupby(["date", "segment"], as_index=False)["gross_sales"].sum()
        fig = go.Figure()
        for seg, col in (("F&B", TEAL), ("Retail", BLUE)):
            d = t14[t14["segment"] == seg]
            fig.add_bar(x=d["date"], y=d["gross_sales"], name=seg, marker_color=col,
                        hovertemplate="%{x|%a %b %d}<br>%{y:$,.0f}<extra>" + seg + "</extra>")
        fig.update_layout(barmode="stack", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        show(style_fig(fig, 320))
    with right:
        section("Prime cost by unit", f"Food/retail cost + labor as % of sales · target {PRIME_TARGET:.0f}%")
        loc = ops_day.groupby("location", as_index=False).agg(
            gross=("gross_sales", "sum"), labor=("labor_cost", "sum"), cogs=("cogs", "sum"))
        loc["prime"] = (loc["labor"] + loc["cogs"]) / loc["gross"] * 100
        loc = loc.sort_values("prime")
        fig = go.Figure(go.Bar(
            x=loc["prime"], y=loc["location"], orientation="h",
            marker_color=[TEAL if v <= PRIME_TARGET else AMBER for v in loc["prime"]],
            text=[f"{v:.1f}%" for v in loc["prime"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>"))
        fig.add_vline(x=PRIME_TARGET, line_dash="dot", line_color=MUTED)
        fig.update_xaxes(range=[0, max(75, loc["prime"].max() + 8)], ticksuffix="%")
        show(style_fig(fig, 320))

    section("Automated insights", "Generated from the day's data and the rules, with no manual review needed to produce them")
    ins = []
    ins.append(("good" if sales_chg >= 0 else "warn",
                f"Sales were <b>{usd(day_sales)}</b>, {abs(sales_chg):.1f}% {'above' if sales_chg >= 0 else 'below'} last {wd}."))
    worst = loc.sort_values("prime").iloc[-1]
    ins.append(("warn" if worst["prime"] > PRIME_TARGET else "good",
                f"Highest prime cost: <b>{worst['location']}</b> at {worst['prime']:.1f}% "
                f"({'above' if worst['prime'] > PRIME_TARGET else 'within'} the {PRIME_TARGET:.0f}% target)."))
    for _, e in exc_base.iterrows():
        ins.append(("bad", f"<b>{e['location']}</b> cash is {usd(abs(e['variance']), 2)} {'short' if e['variance'] < 0 else 'over'} "
                           f"({abs(e['variance']) / threshold:.1f}x tolerance), so it is in the exception queue."))
    if unmapped:
        ins.append(("bad", f"{len(unmapped)} POS categor{'y' if len(unmapped) == 1 else 'ies'} had no active GL rule "
                           f"({', '.join(unmapped)}). Sales were held in suspense."))
    else:
        ins.append(("good", f"All {n_cat} POS categories mapped to GL accounts. Journal is balanced at {usd(j_debit, 2)}."))
    for tone, html in ins:
        st.markdown(f'<div class="ins {tone}">{html}</div>', unsafe_allow_html=True)

    section(f"Location roll-up · {latest:%A, %b %d}")
    tbl = ops_day.groupby(["location", "concept"], as_index=False).agg(
        covers=("covers", "sum"), gross_sales=("gross_sales", "sum"), cogs=("cogs", "sum"),
        labor_cost=("labor_cost", "sum"), variance=("variance", "sum"))
    tbl["prime_pct"] = (tbl["cogs"] + tbl["labor_cost"]) / tbl["gross_sales"] * 100
    tbl["status"] = np.where(tbl["variance"].abs() > threshold, "Review", "OK")
    tbl = tbl[["location", "concept", "covers", "gross_sales", "prime_pct", "variance", "status"]]
    st.dataframe(
        tbl, width="stretch", hide_index=True,
        column_config={
            "location": "Location", "concept": "Concept",
            "covers": st.column_config.NumberColumn("Covers / txns", format="%d"),
            "gross_sales": st.column_config.NumberColumn("Gross sales", format="$%.0f"),
            "prime_pct": st.column_config.ProgressColumn("Prime cost %", min_value=0, max_value=100, format="%.1f%%"),
            "variance": st.column_config.NumberColumn("Over / short", format="$%.2f"),
            "status": "Audit",
        },
    )

# ================================================================== TAB 2
pending_exc = 0
with tab2:
    section("Cash over / short · last 14 days",
            "Counted deposit versus POS expected cash. Red is short, teal is over.")
    hm = ops[ops["date"] > latest - timedelta(days=14)].pivot_table(index="location", columns="date", values="variance", aggfunc="sum")
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=[d.strftime("%a %d") for d in hm.columns], y=list(hm.index),
        zmin=-100, zmax=100, colorscale=[[0, ROSE], [0.5, "#16213A"], [1, TEAL]],
        texttemplate="%{z:.0f}", textfont=dict(size=11, color=TEXT), xgap=3, ygap=3,
        colorbar=dict(title="$", thickness=10, len=0.8),
        hovertemplate="%{y}<br>%{x}<br>%{z:$.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, 80 + 52 * len(hm)))

    section("Exception queue", f"Variances above ${threshold}. Choose a resolution so the day can be approved for close.")
    if exc_base.empty:
        st.markdown('<div class="ins good">No variances above tolerance. This day clears without review.</div>', unsafe_allow_html=True)
    else:
        def likely(v: float) -> str:
            if abs(v) >= 150:
                return "Possible missed cash drop or count error"
            return "Check voids, refunds and tip payouts"
        exc = exc_base.copy()
        exc["Expected cash"] = exc["location"].map(cash_exp_loc)
        exc["Actual deposit"] = (exc["Expected cash"] + exc["variance"]).round(2)
        exc["Likely cause"] = exc["variance"].map(likely)
        exc["Resolution"] = "Pending"
        exc = exc.rename(columns={"location": "Location", "variance": "Variance"})
        exc = exc[["Location", "Expected cash", "Actual deposit", "Variance", "Likely cause", "Resolution"]]
        edited = st.data_editor(
            exc, hide_index=True, width="stretch",
            key=f"exc_{latest}_{threshold}_{'|'.join(sorted(selected))}",
            disabled=["Location", "Expected cash", "Actual deposit", "Variance", "Likely cause"],
            column_config={
                "Expected cash": st.column_config.NumberColumn(format="$%.2f"),
                "Actual deposit": st.column_config.NumberColumn(format="$%.2f"),
                "Variance": st.column_config.NumberColumn(format="$%.2f"),
                "Resolution": st.column_config.SelectboxColumn(
                    options=["Pending", "Counting error corrected", "Void / refund timing",
                             "Employee follow-up", "Written off to Over/Short"], required=True),
            },
        )
        pending_exc = int((edited["Resolution"] == "Pending").sum())
        st.caption(f"{len(edited) - pending_exc} of {len(edited)} resolved")

    section("30-day trend", "Total daily over / short across selected units")
    tr = ops[ops["date"] > latest - timedelta(days=30)].groupby("date", as_index=False)["variance"].sum()
    fig = go.Figure(go.Bar(x=tr["date"], y=tr["variance"],
                           marker_color=[ROSE if v < 0 else TEAL for v in tr["variance"]],
                           hovertemplate="%{x|%a %b %d}<br>%{y:$,.2f}<extra></extra>"))
    fig.add_hline(y=-threshold, line_dash="dot", line_color=AMBER, annotation_text=f"-${threshold}", annotation_font_color=AMBER)
    fig.add_hline(y=threshold, line_dash="dot", line_color=AMBER)
    fig.update_yaxes(tickprefix="$")
    show(style_fig(fig, 260))

# ================================================================== TAB 3
with tab3:
    section("Self-service mapping rules",
            "Finance edits accounts and tax rates here. The journal below rebuilds instantly, with no ticket to IT.")
    edited_rules = st.data_editor(
        st.session_state.gl_rules, num_rows="dynamic", width="stretch", hide_index=True, key="rule_editor",
        column_config={
            "pos_category": "POS category", "gl_account": "GL account", "tax_rule": "Tax rule",
            "tax_rate": st.column_config.NumberColumn("Tax rate %", min_value=0.0, max_value=30.0, step=0.25, format="%.2f"),
            "active": st.column_config.CheckboxColumn("Active"),
        },
    )
    b1, b2, _ = st.columns([1, 1, 5])
    if b1.button("Apply rules", type="primary"):
        st.session_state.gl_rules = edited_rules.reset_index(drop=True)
        st.rerun()
    if b2.button("Reset"):
        st.session_state.gl_rules = DEFAULT_RULES.copy()
        st.rerun()
    st.caption("Tip for the demo: untick *Active* on a category and apply. Its sales move to suspense and the close is blocked.")

    section(f"Journal preview · {latest:%b %d, %Y}", "Balanced double entry, one summary journal for all selected units")
    if unmapped:
        st.markdown(f'<div class="ins bad">Unmapped: {", ".join(unmapped)}. Posting is blocked until a rule exists.</div>',
                    unsafe_allow_html=True)
    jc1, jc2, jc3 = st.columns(3)
    jc1.markdown(kpi("Total debits", usd(j_debit, 2)), unsafe_allow_html=True)
    jc2.markdown(kpi("Total credits", usd(j_credit, 2)), unsafe_allow_html=True)
    jc3.markdown(kpi("Balance check", "Balanced" if balanced else "OUT OF BALANCE",
                     "Debits equal credits" if balanced else f"Difference {usd(j_debit - j_credit, 2)}",
                     "good" if balanced else "bad", "accent-good" if balanced else "accent-bad"), unsafe_allow_html=True)
    st.write("")
    st.dataframe(
        journal, width="stretch", hide_index=True,
        column_config={
            "Debit": st.column_config.NumberColumn(format="$%.2f"),
            "Credit": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    can_post = balanced and not unmapped and pending_exc == 0
    key = f"{latest}|{'|'.join(sorted(selected))}"
    p1, p2, p3 = st.columns([1.4, 1.2, 4])
    if p1.button(f"Approve & post to {gl_target}", type="primary", disabled=not can_post or posted):
        st.session_state.posted_key = key
        st.session_state.posted_ref = f"JE-{latest:%Y%m%d}-POS-SUMMARY"
        st.toast(f"Posted {st.session_state.posted_ref} to {gl_target}", icon="✅")
        st.rerun()
    p2.download_button("Download CSV", journal.to_csv(index=False), file_name=f"journal_{latest}.csv", mime="text/csv")
    if posted:
        p3.markdown(f'<span class="pill ok">Posted · {st.session_state.get("posted_ref")}</span>', unsafe_allow_html=True)
    elif not can_post:
        why = []
        if unmapped:
            why.append("unmapped categories")
        if not balanced:
            why.append("journal out of balance")
        if pending_exc:
            why.append(f"{pending_exc} open cash exception{'s' if pending_exc != 1 else ''}")
        p3.markdown(f'<span class="pill warn">Blocked: {", ".join(why)}</span>', unsafe_allow_html=True)

# ================================================================== TAB 4
with tab4:
    total_tax = float(journal.loc[journal["GL Account"].str.startswith("2150"), "Credit"].sum())
    items = [
        ("ok", "POS data received", f"{n_loc} of {n_loc} units reported for {latest:%a %b %d}"),
        ("ok" if balanced else "bad", "Totals validated", "Sales foot to tenders; journal balances" if balanced else "Journal is out of balance"),
        ("ok" if not unmapped else "bad", "GL mapping coverage",
         f"{mapped_n} of {n_cat} categories mapped" if not unmapped else f"{len(unmapped)} category(ies) in suspense"),
        ("ok" if pending_exc == 0 else "warn", "Cash over / short reviewed",
         "No open exceptions" if pending_exc == 0 else f"{pending_exc} exception(s) awaiting resolution"),
        ("ok", "Card settlement matched (simulated)", "Processor batches tie to POS card tenders within tolerance"),
        ("ok", "Sales tax tie-out", f"{usd(total_tax, 2)} accrued to Sales Tax Payable"),
        ("ok" if posted else ("todo" if (balanced and not unmapped and pending_exc == 0) else "todo"),
         f"Journal posted to {gl_target}",
         f"Posted as {st.session_state.get('posted_ref')}" if posted else "Ready once the steps above are clear"),
    ]
    done = sum(1 for s_, *_ in items if s_ == "ok")
    section("Close checklist", f"{latest:%A, %b %d} · {done} of {len(items)} steps complete")
    st.progress(done / len(items))
    icons = {"ok": "✓", "warn": "!", "bad": "✕", "todo": "·"}
    for s_, title, detail in items:
        st.markdown(
            f'<div class="chk"><div class="i {s_}">{icons[s_]}</div><div class="b"><div class="t">{title}</div>'
            f'<div class="d">{detail}</div></div></div>', unsafe_allow_html=True)
    if done == len(items):
        st.markdown('<div class="ins good">Day closed. The audit trail links each ledger line to its POS source and rule version.</div>',
                    unsafe_allow_html=True)

# ================================================================== TAB 5
with tab5:
    section("7-day cover and staffing forecast", "Same-weekday baseline, with weather and event effects learned from history")
    f1, f2, f3 = st.columns([1.4, 1.2, 1])
    target = f1.selectbox("Location", options=selected)
    weekend_weather = f2.radio("Weekend weather", ["Clear / mild", "Heavy rain / storm"], horizontal=True)
    event = f3.checkbox("Local event nearby (+15% Fri/Sat)", value=False)

    hist = ops[ops["location"] == target].copy()
    hist["dow"] = pd.to_datetime(hist["date"]).dt.dayofweek
    dow_clear = hist[hist["rain"] == "Clear"].groupby("dow")["covers"].mean()
    dow_clear = dow_clear.reindex(range(7)).fillna(hist["covers"].mean())
    rain_days = hist[hist["rain"] == "Rain"]
    rain_factor = float((rain_days["covers"] / rain_days["dow"].map(dow_clear)).mean()) if len(rain_days) else 0.9
    avg_check = hist["gross_sales"].sum() / hist["covers"].sum()
    labor_pct = hist["labor_cost"].sum() / hist["gross_sales"].sum()
    cps = next(l["cps"] for l in LOCATIONS if l["location"] == target)
    st.caption(f"Learned from 60 days: rain lowers traffic by {(1 - rain_factor) * 100:.0f}%. "
               f"Average check {usd(avg_check, 2)}. Labor {labor_pct * 100:.1f}% of sales.")

    rows = []
    start = ops_all["date"].max() + timedelta(days=1)
    baseline_avg = float(dow_clear.mean())
    for i in range(7):
        d = start + timedelta(days=i)
        dw = d.weekday()
        pred = float(dow_clear[dw])
        if dw in (4, 5, 6) and weekend_weather.startswith("Heavy"):
            pred *= rain_factor
        if dw in (4, 5) and event:
            pred *= 1.15
        pc = int(round(pred))
        rows.append(dict(
            Date=d.strftime("%a %b %d"), Baseline=int(round(float(dow_clear[dw]))), Predicted=pc,
            Prep=pc / baseline_avg * 100, Staff=max(3, int(np.ceil(pc / cps))),
            Sales=pc * avg_check, Labor=pc * avg_check * labor_pct))
    fc = pd.DataFrame(rows)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Week covers", f"{fc['Predicted'].sum():,}"), unsafe_allow_html=True)
    k2.markdown(kpi("Forecast sales", usd(fc["Sales"].sum())), unsafe_allow_html=True)
    k3.markdown(kpi("Labor budget", usd(fc["Labor"].sum())), unsafe_allow_html=True)
    peak = fc.loc[fc["Predicted"].idxmax()]
    k4.markdown(kpi("Peak day", peak["Date"], f"{peak['Predicted']:,} covers · {peak['Staff']} staff"), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_bar(x=fc["Date"], y=fc["Predicted"], name="Forecast", marker_color=TEAL,
                text=fc["Predicted"], textposition="outside", cliponaxis=False)
    fig.add_scatter(x=fc["Date"], y=fc["Baseline"], name="Typical (clear-day avg)", mode="lines+markers",
                    line=dict(color=AMBER, dash="dot"), marker=dict(size=7))
    fig.update_yaxes(range=[0, fc[["Predicted", "Baseline"]].max().max() * 1.2])
    st.write("")
    show(style_fig(fig, 320))

    st.dataframe(
        fc.rename(columns={"Predicted": "Predicted covers", "Prep": "Prep volume index", "Staff": "Recommended floor staff",
                           "Sales": "Forecast sales", "Labor": "Labor budget"}).drop(columns=["Baseline"]),
        width="stretch", hide_index=True,
        column_config={
            "Prep volume index": st.column_config.ProgressColumn(min_value=0, max_value=160, format="%.0f%%"),
            "Forecast sales": st.column_config.NumberColumn(format="$%.0f"),
            "Labor budget": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

# ================================================================== TAB 6
with tab6:
    section("Impact calculator", "Illustrative assumptions. Replace them with the prospect's own numbers.")
    a1, a2, a3, a4 = st.columns(4)
    sites = a1.slider("Units", 1, 60, 12)
    manual_min = a2.slider("Manual min / unit / day", 10, 120, 45, help="Pull reports, re-key sales, chase variances")
    auto_min = a3.slider("With automation", 2, 30, 8, help="Reviewing exceptions only")
    rate = a4.slider("Loaded cost / hour ($)", 15, 120, 38)
    days = 365
    hrs_before = sites * manual_min / 60 * days
    hrs_after = sites * auto_min / 60 * days
    saved = max(hrs_before - hrs_after, 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi("Hours returned / year", f"{saved:,.0f}", f"{hrs_before:,.0f} before → {hrs_after:,.0f} after", "good"), unsafe_allow_html=True)
    m2.markdown(kpi("Labor value / year", usd(saved * rate), f"at {usd(rate)}/hr loaded", "good"), unsafe_allow_html=True)
    m3.markdown(kpi("FTE equivalent", f"{saved / 1800:.1f}", "at 1,800 productive hrs/yr"), unsafe_allow_html=True)
    m4.markdown(kpi("Variance visibility", "Next morning", "instead of at month-end review"), unsafe_allow_html=True)

    section("How it connects in production")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card2"><div class="t">1 · Ingest &amp; validate</div><div class="d">POS, payment processor, '
                'bank and PMS feeds land in a staging layer. Duplicate, missing-day and footing checks run before anything touches the ledger.</div></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="card2"><div class="t">2 · Rules as data</div><div class="d">Account mapping, tax rates and '
                'tolerances live in a table the finance team owns, with effective dates, approvals and a change log.</div></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="card2"><div class="t">3 · Approve &amp; post</div><div class="d">Clean days post automatically; '
                f'exceptions wait for an approver. Journals go to {gl_target} by API with an idempotency key, '
                f'and every line traces back to its source.</div></div>', unsafe_allow_html=True)
    st.caption("Demo note: figures above are illustrative. Card settlement matching is simulated in this prototype.")
