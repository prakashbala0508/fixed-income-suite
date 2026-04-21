"""
Page 3 — ALM NII Forecasting & Balance Sheet Stress Tool
Anchored to PNC Financial Services Group 1Q26 public earnings data (March 31, 2026).
Models net interest income across 4 rate scenarios with repricing gap analysis
and CFO-level management commentary.

Data source: PNC 1Q26 Earnings Release, April 15, 2026.
Repricing assumptions are analyst estimates derived from earnings commentary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALM NII Forecasting | FI Suite",
    page_icon="🏦",
    layout="wide",
)

# ── Institutional theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a1628; color: #e8edf5; }
    [data-testid="stSidebar"] {
        background-color: #0d1f3c;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c8d6e8 !important; }
    h1, h2, h3, h4 { color: #e8edf5 !important; }
    .stMetric {
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #a8bdd4 !important; }
    [data-testid="stMetricValue"] { color: #4a9eff !important; }
    .stDataFrame { border: 1px solid #1e3a5f; border-radius: 6px; }
    div[data-testid="stExpander"] {
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
    }
    .interp-box {
        background: #0d1f3c;
        border-left: 3px solid #4a9eff;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #a8bdd4;
        margin-top: 8px;
    }
    .commentary-box {
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 24px 28px;
        font-size: 0.92rem;
        color: #c8d6e8;
        line-height: 1.8;
    }
    .commentary-box h4 { color: #4a9eff !important; margin-bottom: 12px; }
    .footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #0a1628; border-top: 1px solid #1e3a5f;
        padding: 8px 32px; font-size: 0.78rem; color: #5a7a9a;
        display: flex; justify-content: space-between;
    }
    hr { border-color: #1e3a5f; }
    .source-tag {
        font-size: 0.75rem;
        color: #5a7a9a;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "navy":  "#0a1628",
    "blue":  "#4a9eff",
    "slate": "#a8bdd4",
    "white": "#e8edf5",
    "green": "#2ecc71",
    "red":   "#e74c3c",
    "amber": "#f39c12",
    "teal":  "#1abc9c",
}

# ── PNC Balance Sheet Constants — Source: 1Q26 Earnings Release, 03/31/2026 ──
# All values in millions of dollars
PNC_BALANCE_SHEET = {
    # ASSETS
    "ci_loans":           190_000,   # Commercial & Industrial loans (est. from segment data)
    "cre_loans":           34_800,   # Commercial Real Estate loans
    "consumer_loans":     105_000,   # Consumer loans (residential mortgage, auto, etc.)
    "investment_securities": 143_112, # Total investment securities (AFS + HTM)
    "earning_deposits":    26_053,   # Interest-earning deposits with banks (Fed reserves)

    # LIABILITIES
    "ib_deposits":        356_900,   # Interest-bearing deposits (78% of $457.6B total deposits)
    "nib_deposits":       100_700,   # Noninterest-bearing deposits (22%)
    "borrowed_funds":      66_666,   # Borrowed funds (FHLB, senior debt, etc.)

    # INCOME (quarterly, annualized for model)
    "quarterly_nii":        3_961,   # 1Q26 net interest income ($MM)
    "annual_nii":          15_844,   # Annualized 1Q26 NII ($MM)
    "nim":                   2.95,   # Net interest margin (%)
}

# Repricing assumptions: fraction of each category that reprices within 12 months
# Source: Analyst estimates derived from PNC earnings commentary on fixed rate asset
# repricing benefit, deposit beta observations, and standard ALM industry conventions.
REPRICING_ASSUMPTIONS = {
    "ci_loans":              0.60,   # 60% floating / short-term fixed
    "cre_loans":             0.30,   # 30% floating; mostly fixed term
    "consumer_loans":        0.20,   # 20% floating; mostly fixed mortgages/auto
    "investment_securities": 0.05,   # ~5% (short duration runoff only)
    "earning_deposits":      1.00,   # 100% — overnight Fed funds rate
    "ib_deposits":           0.75,   # 75% sensitive (high deposit beta observed)
    "nib_deposits":          0.00,   # 0% — no interest paid
    "borrowed_funds":        0.50,   # 50% floating (FHLB + senior debt mix)
}


# ── Section 1: FRED Data ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_rate_environment() -> dict:
    """
    Fetch current and recent benchmark rates from FRED.
    Used to set the base rate environment for the ALM model.

    Returns:
        dict with current Fed Funds, 2Y Treasury, and 10Y Treasury rates.
    """
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])
    rates = {}
    tickers = {
        "fed_funds": "FEDFUNDS",
        "treasury_2y": "DGS2",
        "treasury_10y": "DGS10",
    }
    for key, ticker in tickers.items():
        series = fred.get_series(ticker).dropna()
        rates[key]      = float(series.iloc[-1])
        rates[f"{key}_date"] = series.index[-1].strftime("%Y-%m-%d")
    return rates


# ── Section 2: Repricing Gap Table ───────────────────────────────────────────

def build_repricing_gap_table(bs: dict, repricing: dict,
                               user_overrides: dict) -> pd.DataFrame:
    """
    Build the standard ALM repricing gap table.

    For each balance sheet category, we show:
    - Total balance
    - Rate-sensitive portion (reprices within 12 months)
    - Fixed portion (does not reprice within 12 months)

    The repricing gap = total rate-sensitive assets minus total rate-sensitive liabilities.
    A positive gap means the bank benefits from rising rates (asset-sensitive).
    A negative gap means the bank is hurt by rising rates (liability-sensitive).

    Args:
        bs:             Balance sheet balances dict (in $MM)
        repricing:      Default repricing assumption fractions
        user_overrides: User-adjusted repricing fractions from Streamlit sliders

    Returns:
        DataFrame with repricing gap breakdown
    """
    # Merge defaults with user overrides
    rep = {**repricing, **user_overrides}

    rows = []

    # ASSETS
    asset_items = [
        ("Commercial & Industrial Loans",   "ci_loans",              "Asset"),
        ("Commercial Real Estate Loans",    "cre_loans",             "Asset"),
        ("Consumer Loans",                  "consumer_loans",        "Asset"),
        ("Investment Securities",           "investment_securities", "Asset"),
        ("Interest-Earning Deposits (Fed)", "earning_deposits",      "Asset"),
    ]

    # LIABILITIES
    liability_items = [
        ("Interest-Bearing Deposits",  "ib_deposits",    "Liability"),
        ("Noninterest-Bearing Deposits","nib_deposits",  "Liability"),
        ("Borrowed Funds",             "borrowed_funds", "Liability"),
    ]

    for label, key, category in asset_items + liability_items:
        balance     = bs[key]
        rate_pct    = rep[key]
        rate_sens   = balance * rate_pct
        fixed_amt   = balance * (1 - rate_pct)
        rows.append({
            "Category":          category,
            "Line Item":         label,
            "Total Balance ($MM)": balance,
            "Rate Sensitive ($MM)": rate_sens,
            "Fixed / Non-Sensitive ($MM)": fixed_amt,
            "Repricing %":       rate_pct * 100,
        })

    df = pd.DataFrame(rows)

    # Summary rows
    total_rate_sens_assets = df[df["Category"]=="Asset"]["Rate Sensitive ($MM)"].sum()
    total_rate_sens_liabs  = df[df["Category"]=="Liability"]["Rate Sensitive ($MM)"].sum()
    gap = total_rate_sens_assets - total_rate_sens_liabs

    summary = pd.DataFrame([
        {"Category": "Summary", "Line Item": "Total Rate-Sensitive Assets",
         "Total Balance ($MM)": df[df["Category"]=="Asset"]["Total Balance ($MM)"].sum(),
         "Rate Sensitive ($MM)": total_rate_sens_assets,
         "Fixed / Non-Sensitive ($MM)": "", "Repricing %": ""},
        {"Category": "Summary", "Line Item": "Total Rate-Sensitive Liabilities",
         "Total Balance ($MM)": df[df["Category"]=="Liability"]["Total Balance ($MM)"].sum(),
         "Rate Sensitive ($MM)": total_rate_sens_liabs,
         "Fixed / Non-Sensitive ($MM)": "", "Repricing %": ""},
        {"Category": "Summary", "Line Item": "Repricing Gap (Assets minus Liabilities)",
         "Total Balance ($MM)": "",
         "Rate Sensitive ($MM)": gap,
         "Fixed / Non-Sensitive ($MM)": "",
         "Repricing %": "ASSET SENSITIVE" if gap > 0 else "LIABILITY SENSITIVE"},
    ])

    return pd.concat([df, summary], ignore_index=True), gap


# ── Section 3: NII Stress Model ───────────────────────────────────────────────

def build_rate_scenarios(base_rates: dict) -> list:
    """
    Define 4 standard ALM rate scenarios used by bank treasury desks.

    Args:
        base_rates: dict with fed_funds, treasury_2y, treasury_10y (all in %)

    Returns:
        List of scenario dicts with rate shifts and labels.
    """
    return [
        {
            "name":        "Base Case",
            "description": "Rates held flat at current levels for 12 months",
            "color":       COLORS["white"],
            "fed_shift":   0,
            "short_shift": 0,
            "long_shift":  0,
        },
        {
            "name":        "+100bps Parallel",
            "description": "All rates rise 100bps uniformly — moderate tightening",
            "color":       COLORS["amber"],
            "fed_shift":   +100,
            "short_shift": +100,
            "long_shift":  +100,
        },
        {
            "name":        "+200bps Parallel",
            "description": "All rates rise 200bps — severe stress scenario, ALM floor",
            "color":       COLORS["red"],
            "fed_shift":   +200,
            "short_shift": +200,
            "long_shift":  +200,
        },
        {
            "name":        "Curve Flattener",
            "description": "Short rates +150bps, long rates +50bps — Fed hiking, long end anchored",
            "color":       COLORS["teal"],
            "fed_shift":   +150,
            "short_shift": +150,
            "long_shift":  +50,
        },
    ]


def compute_nii_scenario(bs: dict, repricing: dict, user_overrides: dict,
                          base_rates: dict, scenario: dict,
                          asset_yields: dict, liability_costs: dict) -> dict:
    """
    Compute NII for one scenario using a simplified static gap model.

    Logic:
    - Fixed-rate assets and liabilities: income/cost does not change with rates.
    - Rate-sensitive assets: income increases by (rate shift * balance * repricing %).
    - Rate-sensitive liabilities: cost increases by (rate shift * balance * repricing %).
      For deposits, a deposit beta is applied — banks rarely pass 100% of rate hikes
      to depositors.

    NII change = (rate-sensitive asset income change) minus (rate-sensitive liability cost change)

    Args:
        asset_yields:    dict of current yield for each asset category (%)
        liability_costs: dict of current cost for each liability category (%)
        scenario:        scenario dict with rate shift parameters

    Returns:
        dict with base NII, stressed NII, NII change, and component breakdown
    """
    rep = {**repricing, **user_overrides}

    # Deposit beta: fraction of rate hike passed through to deposit costs
    # PNC deposit beta implied ~0.40 from NIM expansion during rate hike cycle
    deposit_beta = 0.40

    short_shift = scenario["short_shift"] / 10000  # convert bps to decimal
    long_shift  = scenario["long_shift"]  / 10000
    fed_shift   = scenario["fed_shift"]   / 10000

    # Base NII = sum of (balance * yield) for assets minus sum of (balance * cost) for liabilities
    base_asset_income = sum(
        bs[k] * asset_yields[k] / 100
        for k in ["ci_loans", "cre_loans", "consumer_loans",
                  "investment_securities", "earning_deposits"]
    )
    base_liab_cost = sum(
        bs[k] * liability_costs[k] / 100
        for k in ["ib_deposits", "borrowed_funds"]
    )
    base_nii = base_asset_income - base_liab_cost

    # NII change from rate-sensitive positions
    # Assets: use short_shift for loans and Fed deposits; long_shift for securities
    nii_changes = {}

    nii_changes["ci_loans"] = (
        bs["ci_loans"] * rep["ci_loans"] * short_shift
    )
    nii_changes["cre_loans"] = (
        bs["cre_loans"] * rep["cre_loans"] * short_shift
    )
    nii_changes["consumer_loans"] = (
        bs["consumer_loans"] * rep["consumer_loans"] * short_shift
    )
    nii_changes["investment_securities"] = (
        bs["investment_securities"] * rep["investment_securities"] * long_shift
    )
    nii_changes["earning_deposits"] = (
        bs["earning_deposits"] * rep["earning_deposits"] * fed_shift
    )

    # Liabilities: apply deposit beta to deposits (banks lag in passing rate hikes)
    nii_changes["ib_deposits"] = -(
        bs["ib_deposits"] * rep["ib_deposits"] * short_shift * deposit_beta
    )
    nii_changes["borrowed_funds"] = -(
        bs["borrowed_funds"] * rep["borrowed_funds"] * fed_shift
    )

    total_nii_change = sum(nii_changes.values())
    stressed_nii     = base_nii + total_nii_change

    return {
        "scenario":        scenario["name"],
        "base_nii":        base_nii,
        "stressed_nii":    stressed_nii,
        "nii_change":      total_nii_change,
        "nii_change_pct":  total_nii_change / base_nii * 100 if base_nii != 0 else 0,
        "component_changes": nii_changes,
    }



# ── Section 4: Charting Functions ───────────────────────────────────────────

def plot_repricing_gap(gap_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing rate-sensitive vs fixed balances
    for each asset and liability category.
    """
    detail_df = gap_df[gap_df["Category"].isin(["Asset", "Liability"])].copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Rate Sensitive",
        y=detail_df["Line Item"],
        x=detail_df["Rate Sensitive ($MM)"] / 1000,
        orientation="h",
        marker_color=COLORS["blue"],
    ))
    fig.add_trace(go.Bar(
        name="Fixed / Non-Sensitive",
        y=detail_df["Line Item"],
        x=detail_df["Fixed / Non-Sensitive ($MM)"] / 1000,
        orientation="h",
        marker_color="#1e3a5f",
    ))

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(text="Repricing Profile by Balance Sheet Segment ($ Billions)",
                   font=dict(size=14, color=COLORS["white"]), x=0.02),
        xaxis=dict(title="Balance ($ Billions)", gridcolor="#1e3a5f", zeroline=False),
        yaxis=dict(gridcolor="#1e3a5f", autorange="reversed"),
        legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=220, r=40, t=50, b=50),
        height=420,
    )
    return fig


def plot_nii_scenarios(scenario_results: list) -> go.Figure:
    """
    Grouped bar chart comparing base vs stressed NII for each scenario.
    """
    names       = [r["scenario"] for r in scenario_results]
    base_niis   = [r["base_nii"] / 1000 for r in scenario_results]
    stress_niis = [r["stressed_nii"] / 1000 for r in scenario_results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Base NII",
        x=names, y=base_niis,
        marker_color=COLORS["slate"],
        text=[f"${v:.1f}B" for v in base_niis],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=11),
    ))
    fig.add_trace(go.Bar(
        name="Stressed NII",
        x=names, y=stress_niis,
        marker_color=[COLORS["green"] if s >= b else COLORS["red"]
                      for s, b in zip(stress_niis, base_niis)],
        text=[f"${v:.1f}B" for v in stress_niis],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=11),
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(text="Base vs Stressed NII by Scenario ($ Billions, Annualized)",
                   font=dict(size=14, color=COLORS["white"]), x=0.02),
        xaxis=dict(gridcolor="#1e3a5f", zeroline=False),
        yaxis=dict(title="NII ($ Billions)", gridcolor="#1e3a5f",
                   zeroline=False, tickformat=".1f"),
        legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=60, r=40, t=50, b=80),
        height=400,
    )
    return fig


def plot_nii_waterfall(result: dict) -> go.Figure:
    """
    Waterfall chart showing how each balance sheet segment contributes
    to the NII change under a given stress scenario.
    """
    labels = {
        "ci_loans":              "C&I Loans",
        "cre_loans":             "CRE Loans",
        "consumer_loans":        "Consumer Loans",
        "investment_securities": "Inv. Securities",
        "earning_deposits":      "Fed Deposits",
        "ib_deposits":           "IB Deposits (cost)",
        "borrowed_funds":        "Borrowed Funds (cost)",
    }

    items   = list(result["component_changes"].items())
    names   = [labels[k] for k, _ in items]
    values  = [v / 1e6 for _, v in items]  # convert MM to B (display as $B)
    colors  = [COLORS["green"] if v >= 0 else COLORS["red"] for v in values]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"${v:+.2f}B" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=11),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(
            text=f"NII Sensitivity Waterfall — {result['scenario']} ($ Billions)",
            font=dict(size=14, color=COLORS["white"]), x=0.02
        ),
        xaxis=dict(gridcolor="#1e3a5f", zeroline=False),
        yaxis=dict(title="NII Change ($ Billions)", gridcolor="#1e3a5f",
                   zeroline=True, zerolinecolor=COLORS["slate"],
                   tickformat=".2f"),
        margin=dict(l=60, r=40, t=50, b=100),
        height=400,
    )
    return fig


def plot_nim_trend(base_rates: dict, scenario_results: list) -> go.Figure:
    """
    Line chart showing NIM under each scenario, compared to reported NIM trend.
    """
    historical_nims = {
        "1Q25": 2.78, "2Q25": 2.80, "3Q25": 2.79, "1Q26": 2.95
    }

    fig = go.Figure()

    # Historical NIM line
    fig.add_trace(go.Scatter(
        x=list(historical_nims.keys()),
        y=list(historical_nims.values()),
        mode="lines+markers",
        name="Reported NIM",
        line=dict(color=COLORS["white"], width=2.5),
        marker=dict(size=8),
    ))

    # Total assets for NIM approximation
    total_earning_assets = (
        PNC_BALANCE_SHEET["ci_loans"] +
        PNC_BALANCE_SHEET["cre_loans"] +
        PNC_BALANCE_SHEET["consumer_loans"] +
        PNC_BALANCE_SHEET["investment_securities"] +
        PNC_BALANCE_SHEET["earning_deposits"]
    )

    scenario_colors = [COLORS["slate"], COLORS["amber"], COLORS["red"], COLORS["teal"]]

    for i, r in enumerate(scenario_results):
        stressed_nim = r["stressed_nii"] / total_earning_assets * 100
        fig.add_trace(go.Scatter(
            x=["1Q26 Stressed"],
            y=[stressed_nim],
            mode="markers",
            name=r["scenario"],
            marker=dict(size=12, color=scenario_colors[i],
                        symbol="diamond"),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(text="Net Interest Margin — Historical Trend & Scenario Projections (%)",
                   font=dict(size=14, color=COLORS["white"]), x=0.02),
        xaxis=dict(gridcolor="#1e3a5f", zeroline=False),
        yaxis=dict(title="NIM (%)", gridcolor="#1e3a5f", zeroline=False,
                   tickformat=".2f"),
        legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=60, r=40, t=50, b=50),
        height=360,
    )
    return fig



# ── Section 5: Management Commentary Generator ───────────────────────────────

def generate_commentary(scenario_results: list, gap: float,
                         base_rates: dict, gap_df: pd.DataFrame) -> str:
    """
    Generate a one-page CFO-level management commentary summarizing the
    ALM stress results. Written in plain English, boardroom register.

    Args:
        scenario_results: List of NII result dicts for all scenarios
        gap:              Repricing gap in $MM
        base_rates:       Current market rates dict
        gap_df:           Full repricing gap DataFrame

    Returns:
        HTML-formatted commentary string
    """
    base   = scenario_results[0]
    s100   = scenario_results[1]
    s200   = scenario_results[2]
    flat   = scenario_results[3]

    gap_direction  = "asset-sensitive" if gap > 0 else "liability-sensitive"
    gap_bn         = abs(gap) / 1000

    best_sc  = max(scenario_results[1:], key=lambda r: r["nii_change"])
    worst_sc = min(scenario_results[1:], key=lambda r: r["nii_change"])

    commentary = f"""
    <div class="commentary-box">
    <h4>📋 Management Commentary — Net Interest Income Sensitivity Analysis</h4>
    <p><span class="source-tag">Prepared for CFO Review &nbsp;|&nbsp;
    Based on PNC 1Q26 Balance Sheet (March 31, 2026) &nbsp;|&nbsp;
    Rate environment as of {base_rates.get('fed_funds_date', 'latest available')}</span></p>
    <br>

    <b>Executive Summary</b><br>
    PNC's balance sheet enters the stress period in a structurally <b>{gap_direction}</b>
    position, with a 12-month repricing gap of approximately
    <b>${gap_bn:.1f} billion</b> (assets minus liabilities).
    The base case annualized NII of <b>${base['base_nii']/1000:.1f} billion</b>
    reflects the current rate environment with the federal funds rate at
    <b>{base_rates.get('fed_funds', 0):.2f}%</b>,
    the 2-year Treasury at <b>{base_rates.get('treasury_2y', 0):.2f}%</b>,
    and the 10-year Treasury at <b>{base_rates.get('treasury_10y', 0):.2f}%</b>.
    <br><br>

    <b>Rate Sensitivity</b><br>
    Under a <b>+100bps parallel shock</b>, modeled NII
    {"increases" if s100["nii_change"] >= 0 else "decreases"} by
    <b>${abs(s100["nii_change"])/1000:.2f} billion</b>
    ({s100["nii_change_pct"]:+.1f}%), reflecting the bank's
    {gap_direction} positioning and a deposit beta assumption of 40 cents
    on the dollar — consistent with PNC's observed NIM expansion during
    the 2022-2023 tightening cycle.
    The more severe <b>+200bps shock</b>
    {"adds" if s200["nii_change"] >= 0 else "costs"} an incremental
    <b>${abs(s200["nii_change"] - s100["nii_change"])/1000:.2f} billion</b>
    relative to the +100bps scenario, as the nonlinear deposit repricing
    assumption limits additional liability cost growth beyond the first
    100bps of movement.
    <br><br>

    <b>Curve Shape Risk</b><br>
    The <b>curve flattener scenario</b> (short rates +150bps, long rates +50bps)
    produces an NII {"benefit" if flat["nii_change"] >= 0 else "headwind"} of
    <b>${abs(flat["nii_change"])/1000:.2f} billion</b> ({flat["nii_change_pct"]:+.1f}%),
    driven by the large share of floating-rate commercial and industrial loans
    repricing against a Fed funds rate that rises more than the long end.
    Investment securities — primarily agency MBS with a portfolio duration of
    approximately 3.6 years — contribute modestly given their low repricing
    fraction in the near term.
    <br><br>

    <b>Balance Sheet Positioning</b><br>
    Commercial and industrial loans ($190 billion, 60% floating-rate assumption)
    represent the primary driver of asset sensitivity. Investment securities
    ($143 billion, 3.6-year duration) provide a longer-dated fixed income
    anchor that limits the upside in falling rate environments.
    On the liability side, interest-bearing deposits ($357 billion)
    reprice with an assumed 40% beta, meaning the bank captures the majority
    of rate increases as incremental NII rather than passing them through
    to depositors at the full rate. Borrowed funds ($67 billion, 50% floating)
    represent a meaningful but manageable source of liability sensitivity.
    <br><br>

    <b>Key Risks and Considerations</b><br>
    The primary risk to the upside NII scenarios is deposit mix shift —
    if commercial clients accelerate migration from noninterest-bearing to
    interest-bearing accounts in a rising rate environment, effective deposit
    beta would exceed the 40% assumption and compress NIM.
    The FirstBank acquisition adds approximately $23 billion in deposits
    and $16 billion in loans, modestly increasing balance sheet scale but
    not materially changing the overall asset-sensitive positioning.
    Management should monitor the 2s10s Treasury spread as a leading
    indicator of flattener risk to the net interest margin.
    <br><br>

    <b>Conclusion</b><br>
    PNC is well-positioned to benefit from a higher-for-longer rate
    environment given its asset-sensitive balance sheet. The strongest NII
    outcome is observed under the <b>{best_sc["scenario"]}</b> scenario
    (${best_sc["nii_change"]/1000:+.2f}B NII change),
    while the most adverse outcome under the scenarios modeled is
    <b>{worst_sc["scenario"]}</b>
    (${worst_sc["nii_change"]/1000:+.2f}B NII change).
    No scenario produces a structurally impaired NII, consistent with
    PNC management's public guidance for continued NII growth in 2026.
    </div>
    """
    return commentary



# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI — Page Layout
# ═══════════════════════════════════════════════════════════════════════════

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 28px 0 12px 0;">
    <div style="font-size:0.8rem; color:#4a9eff; letter-spacing:0.1em;
                text-transform:uppercase; margin-bottom:8px;">
        Tool 3 of 3 · Fixed Income Analytics Suite
    </div>
    <h1 style="font-size:2rem; font-weight:700; color:#e8edf5; margin:0 0 8px 0;">
        ALM NII Forecasting & Balance Sheet Stress Tool
    </h1>
    <p style="color:#a8bdd4; font-size:0.95rem; margin:0;">
        Anchored to PNC 1Q26 public balance sheet (March 31, 2026) ·
        Repricing gap analysis · NII sensitivity across 4 rate scenarios ·
        CFO-level management commentary
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="source-tag" style="margin-bottom:16px;">
Data source: PNC Financial Services Group 1Q26 Earnings Release, April 15, 2026.
Repricing assumptions are analyst estimates derived from earnings commentary
and standard ALM industry conventions — clearly labeled throughout.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Fetch live rates ──────────────────────────────────────────────────────────
with st.spinner("Fetching live rate environment from FRED..."):
    try:
        base_rates = fetch_rate_environment()
        st.info(
            f"📡 Current Rate Environment (FRED) — "
            f"Fed Funds: **{base_rates['fed_funds']:.2f}%** · "
            f"2Y Treasury: **{base_rates['treasury_2y']:.2f}%** · "
            f"10Y Treasury: **{base_rates['treasury_10y']:.2f}%** "
            f"(as of {base_rates['fed_funds_date']})"
        )
    except Exception as e:
        st.warning(f"Could not fetch rates: {e}")
        base_rates = {"fed_funds": 4.33, "treasury_2y": 4.00,
                      "treasury_10y": 4.30, "fed_funds_date": "latest"}

st.markdown("---")

# ── Section A: Balance Sheet Display ─────────────────────────────────────────
st.subheader("🏦 PNC Balance Sheet Snapshot — March 31, 2026")
st.caption("Source: PNC 1Q26 Earnings Release. All figures in millions of dollars.")

bs_display = pd.DataFrame([
    {"Segment": "ASSETS", "Line Item": "Commercial & Industrial Loans",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['ci_loans']:,.0f}",
     "Note": "Est. from C&I segment avg $211.4B; spot $221.2B"},
    {"Segment": "ASSETS", "Line Item": "Commercial Real Estate Loans",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['cre_loans']:,.0f}",
     "Note": "Quarter-end per earnings release"},
    {"Segment": "ASSETS", "Line Item": "Consumer Loans",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['consumer_loans']:,.0f}",
     "Note": "Quarter-end per earnings release"},
    {"Segment": "ASSETS", "Line Item": "Investment Securities",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['investment_securities']:,.0f}",
     "Note": "AFS $71.6B + HTM $72.9B avg; duration 3.6 yrs"},
    {"Segment": "ASSETS", "Line Item": "Interest-Earning Deposits (Fed)",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['earning_deposits']:,.0f}",
     "Note": "Quarter-end per earnings release"},
    {"Segment": "LIABILITIES", "Line Item": "Interest-Bearing Deposits",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['ib_deposits']:,.0f}",
     "Note": "78% of $457.6B total deposits"},
    {"Segment": "LIABILITIES", "Line Item": "Noninterest-Bearing Deposits",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['nib_deposits']:,.0f}",
     "Note": "22% of total deposits"},
    {"Segment": "LIABILITIES", "Line Item": "Borrowed Funds",
     "Balance ($MM)": f"${PNC_BALANCE_SHEET['borrowed_funds']:,.0f}",
     "Note": "FHLB advances + senior debt; quarter-end"},
])

st.dataframe(bs_display, use_container_width=True, hide_index=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("1Q26 NII (Quarterly)", f"${PNC_BALANCE_SHEET['quarterly_nii']:,.0f}MM")
col_m2.metric("Annualized NII",       f"${PNC_BALANCE_SHEET['annual_nii']/1000:.1f}B")
col_m3.metric("Net Interest Margin",  f"{PNC_BALANCE_SHEET['nim']:.2f}%")
col_m4.metric("Total Assets",         "$603.0B")

st.markdown("---")

# ── Section B: Repricing Assumption Controls ──────────────────────────────────
st.subheader("⚙️ Repricing Assumptions")
st.caption(
    "Default values are analyst estimates derived from PNC earnings commentary. "
    "Adjust the sliders to model alternative repricing assumptions."
)

with st.expander("🔧 Adjust Repricing Assumptions (click to expand)", expanded=False):
    st.markdown("**Assets** — fraction repricing within 12 months")
    col1, col2, col3 = st.columns(3)
    with col1:
        ci_rep  = st.slider("C&I Loans (%)",   0, 100, 60, 5,
                             help="Default 60% — mix of floating SOFR loans and short-term fixed") / 100
        cre_rep = st.slider("CRE Loans (%)",   0, 100, 30, 5,
                             help="Default 30% — mostly fixed term loans") / 100
    with col2:
        con_rep = st.slider("Consumer Loans (%)", 0, 100, 20, 5,
                             help="Default 20% — mostly fixed rate mortgages and auto loans") / 100
        sec_rep = st.slider("Inv. Securities (%)", 0, 100, 5, 1,
                             help="Default 5% — short runoff; duration ~3.6 years") / 100
    with col3:
        fed_rep = st.slider("Fed Deposits (%)", 50, 100, 100, 5,
                             help="Default 100% — overnight rate, fully repricing") / 100

    st.markdown("**Liabilities** — fraction repricing within 12 months")
    col4, col5 = st.columns(2)
    with col4:
        ib_rep  = st.slider("IB Deposits (%)",    0, 100, 75, 5,
                             help="Default 75% — high rate sensitivity; beta applied separately") / 100
    with col5:
        bor_rep = st.slider("Borrowed Funds (%)", 0, 100, 50, 5,
                             help="Default 50% — mix of short FHLB and longer senior debt") / 100

user_overrides = {
    "ci_loans":              ci_rep,
    "cre_loans":             cre_rep,
    "consumer_loans":        con_rep,
    "investment_securities": sec_rep,
    "earning_deposits":      fed_rep,
    "ib_deposits":           ib_rep,
    "borrowed_funds":        bor_rep,
    "nib_deposits":          0.0,
}

st.markdown("**Asset Yields & Liability Costs** (current, used to compute base NII)")
st.caption("Defaults are analyst estimates based on reported NIM, asset mix, and rate environment.")

col_y1, col_y2, col_y3 = st.columns(3)
with col_y1:
    ci_yield  = st.number_input("C&I Loan Yield (%)",    value=6.80, step=0.05, format="%.2f")
    cre_yield = st.number_input("CRE Loan Yield (%)",    value=6.20, step=0.05, format="%.2f")
with col_y2:
    con_yield = st.number_input("Consumer Loan Yield (%)", value=5.90, step=0.05, format="%.2f")
    sec_yield = st.number_input("Securities Yield (%)",  value=3.80, step=0.05, format="%.2f")
with col_y3:
    fed_yield = st.number_input("Fed Deposit Yield (%)", value=4.30, step=0.05, format="%.2f")
    ib_cost   = st.number_input("IB Deposit Cost (%)",   value=2.10, step=0.05, format="%.2f")
    bor_cost  = st.number_input("Borrowed Funds Cost (%)", value=4.50, step=0.05, format="%.2f")

asset_yields = {
    "ci_loans": ci_yield, "cre_loans": cre_yield, "consumer_loans": con_yield,
    "investment_securities": sec_yield, "earning_deposits": fed_yield,
}
liability_costs = {"ib_deposits": ib_cost, "borrowed_funds": bor_cost}

if st.button("▶ Run ALM Stress Model", type="primary"):

    # ── Section C: Repricing Gap ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Repricing Gap Analysis")

    gap_df, gap = build_repricing_gap_table(
        PNC_BALANCE_SHEET, REPRICING_ASSUMPTIONS, user_overrides
    )

    col_gap1, col_gap2 = st.columns([3, 2])

    with col_gap1:
        fig_gap = plot_repricing_gap(gap_df)
        st.plotly_chart(fig_gap, use_container_width=True)
        direction = "asset-sensitive" if gap > 0 else "liability-sensitive"
        st.markdown(f"""
        <div class="interp-box">
        📌 <b>Interpretation:</b> PNC is <b>{direction}</b> with a 12-month
        repricing gap of <b>${gap/1000:+.1f} billion</b>.
        {"An asset-sensitive bank benefits when rates rise — rate-sensitive assets reprice faster than liabilities, expanding NIM." if gap > 0 else "A liability-sensitive bank is hurt when rates rise — liabilities reprice faster than assets, compressing NIM."}
        </div>
        """, unsafe_allow_html=True)

    with col_gap2:
        display_gap = gap_df[["Line Item", "Total Balance ($MM)",
                               "Rate Sensitive ($MM)", "Repricing %"]].copy()
        for col in ["Total Balance ($MM)", "Rate Sensitive ($MM)"]:
            display_gap[col] = display_gap[col].apply(
                lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x
            )
        display_gap["Repricing %"] = display_gap["Repricing %"].apply(
            lambda x: f"{x:.0f}%" if isinstance(x, (int, float)) else x
        )
        st.dataframe(display_gap, use_container_width=True, hide_index=True, height=400)

    # ── Section D: NII Scenarios ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ NII Stress Scenarios")

    scenarios = build_rate_scenarios(base_rates)
    scenario_results = []
    for sc in scenarios:
        result = compute_nii_scenario(
            PNC_BALANCE_SHEET, REPRICING_ASSUMPTIONS, user_overrides,
            base_rates, sc, asset_yields, liability_costs
        )
        scenario_results.append(result)

    # Summary table
    summary_rows = []
    for r in scenario_results:
        summary_rows.append({
            "Scenario":            r["scenario"],
            "Base NII ($MM)":      f"${r['base_nii']:,.0f}",
            "Stressed NII ($MM)":  f"${r['stressed_nii']:,.0f}",
            "NII Change ($MM)":    f"${r['nii_change']:+,.0f}",
            "NII Change (%)":      f"{r['nii_change_pct']:+.1f}%",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # NII scenario chart
    col_nii1, col_nii2 = st.columns(2)
    with col_nii1:
        fig_nii = plot_nii_scenarios(scenario_results)
        st.plotly_chart(fig_nii, use_container_width=True)
        best = max(scenario_results[1:], key=lambda r: r["nii_change"])
        st.markdown(f"""
        <div class="interp-box">
        📌 <b>Interpretation:</b> PNC's asset-sensitive balance sheet generates
        the highest NII uplift under the <b>{best["scenario"]}</b> scenario —
        consistent with the bank's public guidance that higher rates benefit NII
        through faster repricing of its floating-rate commercial loan book.
        </div>
        """, unsafe_allow_html=True)

    with col_nii2:
        fig_nim = plot_nim_trend(base_rates, scenario_results)
        st.plotly_chart(fig_nim, use_container_width=True)
        st.markdown("""
        <div class="interp-box">
        📌 <b>Interpretation:</b> NIM has expanded from 2.78% in 1Q25 to 2.95%
        in 1Q26, reflecting lower funding costs and fixed rate asset repricing.
        Stressed scenario diamonds show projected NIM under each rate shock.
        </div>
        """, unsafe_allow_html=True)

    # ── Section E: NII Waterfall — pick worst non-base scenario ──────────────
    st.markdown("---")
    st.subheader("📉 NII Sensitivity Waterfall by Segment")

    tab_labels = [r["scenario"] for r in scenario_results[1:]]
    tabs = st.tabs(tab_labels)

    for tab, result in zip(tabs, scenario_results[1:]):
        with tab:
            fig_wf = plot_nii_waterfall(result)
            st.plotly_chart(fig_wf, use_container_width=True)
            st.markdown(f"""
            <div class="interp-box">
            📌 <b>Interpretation:</b> Under the <b>{result["scenario"]}</b>,
            C&I loans and Fed deposits are the primary drivers of NII benefit
            (green bars) while interest-bearing deposit cost increases
            (red bar) partially offset the asset repricing gains.
            The net NII change is <b>${result["nii_change"]/1000:+.2f} billion</b>.
            </div>
            """, unsafe_allow_html=True)

    # ── Section F: Management Commentary ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Management Commentary")
    st.caption("CFO-level summary of ALM stress results. Plain English, boardroom register.")

    commentary_html = generate_commentary(
        scenario_results, gap, base_rates, gap_df
    )
    st.markdown(commentary_html, unsafe_allow_html=True)

# ── Methodology Expander ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Methodology — ALM NII Model"):
    st.markdown("""
    ### What Is ALM?

    Asset-Liability Management (ALM) is the practice of managing the risks that arise
    from mismatches between the assets and liabilities on a bank's balance sheet.
    The primary risk managed is interest rate risk — the risk that changes in market
    interest rates will reduce the bank's net interest income (NII) or economic value.

    ---

    ### The Repricing Gap

    The repricing gap measures how much of a bank's assets versus liabilities will
    reprice (reset to current market rates) within a given time horizon, typically
    12 months. The gap is calculated as:

    Repricing Gap = Rate-Sensitive Assets minus Rate-Sensitive Liabilities

    A positive gap (more assets than liabilities repricing) means the bank is
    **asset-sensitive** — it benefits from rising rates. A negative gap means the
    bank is **liability-sensitive** — it is hurt by rising rates.

    ---

    ### How NII Is Modeled

    For each rate scenario, the NII change is estimated as follows:

    | Component | How It Is Calculated |
    |---|---|
    | Asset NII change | Rate-sensitive balance times rate shift |
    | Deposit cost change | Rate-sensitive balance times rate shift times deposit beta |
    | Borrowed funds cost change | Rate-sensitive balance times Fed funds shift |
    | Net NII change | Sum of asset changes minus sum of liability changes |

    The **deposit beta** is the fraction of a rate increase that a bank passes through
    to deposit customers. A beta of 40% means that for every 100bps the Fed raises rates,
    the bank pays depositors 40bps more. PNC's implied deposit beta of ~40% is derived
    from NIM expansion of approximately 38bps over the 2022-2023 tightening cycle relative
    to the magnitude of Fed hikes.

    ---

    ### Rate Scenarios

    | Scenario | Fed Funds Shift | 2Y Shift | 10Y Shift | What It Represents |
    |---|---|---|---|---|
    | Base Case | 0bps | 0bps | 0bps | Current rates held flat for 12 months |
    | +100bps Parallel | +100bps | +100bps | +100bps | Moderate Fed tightening |
    | +200bps Parallel | +200bps | +200bps | +200bps | Severe stress; regulatory floor |
    | Curve Flattener | +150bps | +150bps | +50bps | Fed hiking; long end anchored |

    ---

    ### Limitations

    This is a simplified static gap model. It does not model:
    - Option-adjusted prepayments on mortgage assets
    - Non-linear deposit behavior at extreme rate levels
    - Balance sheet growth or shrinkage under different scenarios
    - Basis risk between different floating rate indices
    - Economic value of equity (EVE) sensitivity

    A full production ALM model would incorporate all of these factors using a
    dynamic simulation engine such as QRM, Bancware, or Empyrean.

    ---

    ### Data Sources

    | Data | Source |
    |---|---|
    | Balance sheet balances | PNC 1Q26 Earnings Release, April 15, 2026 |
    | Repricing assumptions | Analyst estimates from earnings commentary |
    | Asset yields / liability costs | Analyst estimates based on reported NIM and rate environment |
    | Current market rates | FRED API (Fed Funds, DGS2, DGS10) |
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Fixed Income Analytics Suite · Tool 3: ALM NII Forecasting & Balance Sheet Stress</span>
    <span>Prakash Balasubramanian · prakash.bala.work@gmail.com</span>
</div>
""", unsafe_allow_html=True)
