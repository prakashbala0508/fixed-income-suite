"""
Page 2 — Agency MBS Cash Flow & Prepayment Model
Models monthly MBS pool cash flows using a from-scratch PSA prepayment framework.
Computes WAL, yield, and effective duration across user-defined PSA speed scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MBS Prepayment Model | FI Suite",
    page_icon="🏠",
    layout="wide",
)

# ── Institutional theme (matches app.py) ─────────────────────────────────────
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
    .footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #0a1628; border-top: 1px solid #1e3a5f;
        padding: 8px 32px; font-size: 0.78rem; color: #5a7a9a;
        display: flex; justify-content: space-between;
    }
    hr { border-color: #1e3a5f; }
    .scenario-card {
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .scenario-card h4 { color: #4a9eff !important; margin: 0 0 8px 0; font-size: 0.95rem; }
    .scenario-card p  { color: #a8bdd4; font-size: 0.85rem; margin: 4px 0; }
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
    "purple":"#9b59b6",
}

SCENARIO_COLORS = [COLORS["blue"], COLORS["green"], COLORS["amber"], COLORS["red"], COLORS["purple"]]


# ── Section 1: FRED Data ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_mortgage_rate() -> tuple:
    """
    Fetch the current 30-year fixed mortgage rate from FRED (MORTGAGE30US).
    Used to contextualize the pool coupon input — if WAC is well below the
    current mortgage rate, prepayments will be slow (no refinancing incentive).

    Returns:
        (rate_pct, as_of_date) tuple
    """
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])
    series = fred.get_series("MORTGAGE30US").dropna()
    rate   = float(series.iloc[-1])
    date   = series.index[-1].strftime("%Y-%m-%d")
    return rate, date


# ── Section 2: PSA Prepayment Model ─────────────────────────────────────────

def cpr_from_psa(month: int, psa_speed: float) -> float:
    """
    Compute the Conditional Prepayment Rate (CPR) for a given loan age
    and PSA speed using the standard PSA benchmark curve.

    PSA convention:
      - 100 PSA = CPR ramps linearly from 0% at month 1 to 6% at month 30,
        then stays flat at 6% for the remaining life.
      - Any PSA speed scales this linearly:
        CPR = PSA_speed/100 * min(month/30, 1) * 6%

    Args:
        month:     Loan age in months (1-indexed)
        psa_speed: PSA speed as a percentage (e.g., 100 = 100 PSA)

    Returns:
        Annual CPR as a decimal (e.g., 0.06 = 6% CPR)
    """
    base_cpr = min(month / 30.0, 1.0) * 0.06
    return (psa_speed / 100.0) * base_cpr


def smm_from_cpr(cpr: float) -> float:
    """
    Convert annual CPR to monthly SMM (Single Monthly Mortality).

    The SMM is the fraction of the remaining pool balance that prepays
    in a given month. The relationship to CPR is:

        SMM = 1 - (1 - CPR)^(1/12)

    Args:
        cpr: Annual CPR as a decimal

    Returns:
        SMM as a decimal
    """
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def build_cash_flows(original_balance: float, wac: float, wam: int,
                     psa_speed: float) -> pd.DataFrame:
    """
    Build the complete monthly cash flow waterfall for an agency MBS pool.

    Each month computes:
      1. Beginning balance
      2. Scheduled interest (WAC/12 * balance)
      3. Scheduled principal (from standard amortization)
      4. Prepayment principal (SMM * remaining balance after scheduled principal)
      5. Total principal = scheduled + prepayment
      6. Ending balance

    Note: The pass-through rate (what investors receive) equals WAC for simplicity.
    In real agency MBS, the investor coupon = WAC minus a servicing/guarantee fee (~25-50bps).

    Args:
        original_balance: Starting pool balance in dollars
        wac:              Weighted Average Coupon (annual, as decimal)
        wam:              Weighted Average Maturity in months
        psa_speed:        PSA prepayment speed

    Returns:
        DataFrame with one row per month containing all cash flow components
    """
    monthly_rate = wac / 12.0
    balance      = original_balance

    # Scheduled monthly payment for a fully-amortizing pool
    # Standard mortgage payment formula: P * r / (1 - (1+r)^-n)
    if monthly_rate > 0:
        scheduled_payment = balance * monthly_rate / (1 - (1 + monthly_rate) ** -wam)
    else:
        scheduled_payment = balance / wam

    rows = []
    for month in range(1, wam + 1):
        if balance <= 0.01:
            break

        cpr   = cpr_from_psa(month, psa_speed)
        smm   = smm_from_cpr(cpr)

        # Interest payment (on beginning balance)
        interest = balance * monthly_rate

        # Scheduled principal (from standard amortization formula)
        sched_principal = min(scheduled_payment - interest, balance)
        sched_principal = max(sched_principal, 0.0)

        # Prepayment principal (SMM applied to balance after scheduled principal)
        remaining_after_sched = balance - sched_principal
        prepay_principal      = smm * remaining_after_sched
        prepay_principal      = max(prepay_principal, 0.0)

        total_principal = sched_principal + prepay_principal
        total_cf        = interest + total_principal

        ending_balance  = balance - total_principal
        ending_balance  = max(ending_balance, 0.0)

        rows.append({
            "Month":                 month,
            "Beginning Balance ($)": balance,
            "CPR (%)":               cpr * 100,
            "SMM (%)":               smm * 100,
            "Scheduled Interest ($)":interest,
            "Scheduled Principal ($)":sched_principal,
            "Prepay Principal ($)":  prepay_principal,
            "Total Principal ($)":   total_principal,
            "Total Cash Flow ($)":   total_cf,
            "Ending Balance ($)":    ending_balance,
        })

        # Recalculate scheduled payment on remaining balance for next period
        remaining_months = wam - month
        if remaining_months > 0 and ending_balance > 0.01:
            if monthly_rate > 0:
                scheduled_payment = ending_balance * monthly_rate / (
                    1 - (1 + monthly_rate) ** -remaining_months
                )
            else:
                scheduled_payment = ending_balance / remaining_months

        balance = ending_balance

    return pd.DataFrame(rows)


# ── Section 3: Analytics ─────────────────────────────────────────────────────

def compute_wal(cf_df: pd.DataFrame) -> float:
    """
    Compute Weighted Average Life (WAL) — the weighted average time in years
    until each dollar of principal is returned to the investor.

    WAL = sum(principal_t * t) / sum(principal_t)
    where t is in years.

    WAL is the primary duration measure used for MBS on a trading desk.
    It tells the investor how long their principal is effectively at risk.

    Args:
        cf_df: Cash flow DataFrame from build_cash_flows()

    Returns:
        WAL in years
    """
    total_principal = cf_df["Total Principal ($)"].sum()
    if total_principal == 0:
        return 0.0
    weighted = (cf_df["Total Principal ($)"] * cf_df["Month"] / 12.0).sum()
    return weighted / total_principal


def compute_yield(cf_df: pd.DataFrame, price_pct: float = 100.0) -> float:
    """
    Compute the MBS yield (monthly IRR * 12) given a price as % of par.

    Uses Newton-Raphson iteration to solve for the discount rate r such that:
        sum(CF_t / (1 + r/12)^t) = Price

    Args:
        cf_df:     Cash flow DataFrame
        price_pct: Price as % of face value (default 100 = par)

    Returns:
        Annual yield as a decimal
    """
    price       = cf_df["Beginning Balance ($)"].iloc[0] * price_pct / 100.0
    cash_flows  = cf_df["Total Cash Flow ($)"].values
    months      = cf_df["Month"].values

    # Newton-Raphson: solve NPV(r) = 0
    r = 0.005  # initial guess: ~6% annual
    for _ in range(200):
        npv  = sum(cf / (1 + r) ** t for cf, t in zip(cash_flows, months)) - price
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for cf, t in zip(cash_flows, months))
        if abs(dnpv) < 1e-12:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < 1e-10:
            r = r_new
            break
        r = max(r_new, 1e-6)

    return r * 12  # annualize


def compute_effective_duration(cf_df: pd.DataFrame, wac: float,
                               original_balance: float, wam: int,
                               psa_speed: float, shock_bps: float = 100.0) -> float:
    """
    Compute effective duration by repricing the MBS under parallel yield shocks.
    Unlike modified duration, effective duration accounts for the change in
    prepayment behavior when rates shift (refinancing incentive changes).

    Effective Duration = -(P_up - P_down) / (2 * P_base * dy)

    For MBS, a rate increase reduces prepayments (extension risk),
    and a rate decrease accelerates them (contraction risk).
    We apply a simplified prepayment response: ±50 PSA per 100bps shock.

    Args:
        shock_bps: Yield shock in basis points for the numerical derivative

    Returns:
        Effective duration in years
    """
    dy = shock_bps / 10000.0

    # Adjust PSA speed for rate sensitivity (simplified convexity adjustment)
    psa_up   = max(psa_speed - 50 * (shock_bps / 100), 50)   # rates up → slower prepay
    psa_down = psa_speed + 50 * (shock_bps / 100)              # rates down → faster prepay

    wac_up   = wac + dy
    wac_down = max(wac - dy, 0.001)

    cf_up   = build_cash_flows(original_balance, wac_up,   wam, psa_up)
    cf_down = build_cash_flows(original_balance, wac_down, wam, psa_down)

    # Price = NPV of cash flows discounted at shifted yield
    def npv_at_rate(cf_df_local, rate):
        return sum(
            cf / (1 + rate / 12) ** t
            for cf, t in zip(cf_df_local["Total Cash Flow ($)"], cf_df_local["Month"])
        )

    p_base = npv_at_rate(cf_df, wac)
    p_up   = npv_at_rate(cf_up,   wac + dy)
    p_down = npv_at_rate(cf_down, wac - dy)

    if p_base == 0:
        return 0.0

    eff_dur = -(p_up - p_down) / (2 * p_base * dy)
    return eff_dur



# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI — Page Layout
# ═══════════════════════════════════════════════════════════════════════════

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 28px 0 12px 0;">
    <div style="font-size:0.8rem; color:#4a9eff; letter-spacing:0.1em;
                text-transform:uppercase; margin-bottom:8px;">
        Tool 2 of 3 · Fixed Income Analytics Suite
    </div>
    <h1 style="font-size:2rem; font-weight:700; color:#e8edf5; margin:0 0 8px 0;">
        Agency MBS Cash Flow & Prepayment Model
    </h1>
    <p style="color:#a8bdd4; font-size:0.95rem; margin:0;">
        PSA prepayment framework built from scratch · Full monthly cash flow waterfall ·
        WAL, yield & effective duration · Multi-scenario comparison
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Fetch live mortgage rate from FRED ───────────────────────────────────────
with st.spinner("Fetching current mortgage rate from FRED..."):
    try:
        mortgage_rate, mort_date = fetch_mortgage_rate()
        st.info(
            f"📡 Current 30-Year Fixed Mortgage Rate (FRED MORTGAGE30US): "
            f"**{mortgage_rate:.2f}%** as of {mort_date} — "
            f"use this as a reference when setting your pool WAC below."
        )
    except Exception as e:
        st.warning(f"Could not fetch mortgage rate: {e}")
        mortgage_rate = 7.0

st.markdown("---")

# ── Section A: Pool Parameters ───────────────────────────────────────────────
st.subheader("🏠 Pool Parameters")
st.caption("Define the characteristics of your generic agency MBS pool.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    original_balance = st.number_input(
        "Original Pool Balance ($)",
        min_value=1_000_000.0,
        max_value=500_000_000.0,
        value=100_000_000.0,
        step=1_000_000.0,
        format="%.0f",
        help="Total face value of all mortgages in the pool at origination.",
    )

with col2:
    wac_pct = st.number_input(
        "WAC — Weighted Avg Coupon (%)",
        min_value=1.0,
        max_value=15.0,
        value=round(mortgage_rate, 3),
        step=0.125,
        format="%.3f",
        help="The weighted average interest rate on all mortgages in the pool.",
    )
    wac = wac_pct / 100.0

with col3:
    wam = st.number_input(
        "WAM — Weighted Avg Maturity (months)",
        min_value=60,
        max_value=360,
        value=360,
        step=12,
        help="The weighted average remaining term of loans in the pool. 360 = 30 years.",
    )

with col4:
    price_pct = st.number_input(
        "Pool Price (% of Par)",
        min_value=80.0,
        max_value=120.0,
        value=100.0,
        step=0.25,
        format="%.2f",
        help="Current market price as % of face value. 100 = par.",
    )

st.markdown("---")

# ── Section B: PSA Scenario Selection ───────────────────────────────────────
st.subheader("📊 PSA Speed Scenarios")
st.caption(
    "Select up to 3 PSA speeds to compare side by side. "
    "100 PSA is the benchmark. 200 PSA = twice the benchmark prepayment speed."
)

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    psa1 = st.slider("Scenario 1 — PSA Speed", 50, 600, 100, step=25,
                     help="100 PSA is the industry benchmark.")
with col_s2:
    psa2 = st.slider("Scenario 2 — PSA Speed", 50, 600, 200, step=25)
with col_s3:
    psa3 = st.slider("Scenario 3 — PSA Speed", 50, 600, 400, step=25)

psa_speeds = [psa1, psa2, psa3]

# Remove duplicates while preserving order
seen = set()
psa_speeds_unique = []
for s in psa_speeds:
    if s not in seen:
        psa_speeds_unique.append(s)
        seen.add(s)

if st.button("▶ Run Prepayment Model", type="primary"):

    # ── Build cash flows for each scenario ───────────────────────────────────
    with st.spinner("Running PSA model..."):
        scenarios = []
        for speed in psa_speeds_unique:
            cf_df  = build_cash_flows(original_balance, wac, int(wam), speed)
            wal    = compute_wal(cf_df)
            yld    = compute_yield(cf_df, price_pct)
            eff_dur = compute_effective_duration(cf_df, wac, original_balance, int(wam), speed)
            scenarios.append({
                "psa":     speed,
                "cf_df":   cf_df,
                "wal":     wal,
                "yield":   yld,
                "eff_dur": eff_dur,
            })

    # ── Section C: PSA Curve Visualization ───────────────────────────────────
    st.markdown("---")
    st.subheader("📈 PSA Prepayment Ramp")

    fig_psa = plot_psa_curve(psa_speeds_unique)
    st.plotly_chart(fig_psa, use_container_width=True)
    st.markdown("""
    <div class="interp-box">
    📌 <b>Interpretation:</b> The PSA ramp reflects the empirical observation that 
    new mortgage pools have low initial prepayment rates as borrowers are unlikely 
    to refinance immediately after origination — prepayments accelerate over the 
    first 30 months as the pool seasons, then stabilize.
    </div>
    """, unsafe_allow_html=True)

    # ── Section D: Summary Metrics Table ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Scenario Summary Metrics")

    metric_rows = []
    for sc in scenarios:
        metric_rows.append({
            "PSA Speed": f"{sc['psa']} PSA",
            "Weighted Avg Life (Yrs)": f"{sc['wal']:.3f}",
            "Yield (%)": f"{sc['yield']*100:.3f}",
            "Effective Duration (Yrs)": f"{sc['eff_dur']:.3f}",
            "Total Principal ($MM)": f"{sc['cf_df']['Total Principal ($)'].sum()/1e6:.2f}",
            "Total Interest ($MM)": f"{sc['cf_df']['Scheduled Interest ($)'].sum()/1e6:.2f}",
            "Months to 50% Paydown": str(
                int(sc['cf_df'].loc[
                    sc['cf_df']["Ending Balance ($)"] <= original_balance * 0.50,
                    "Month"
                ].min()) if (sc['cf_df']["Ending Balance ($)"] <= original_balance * 0.50).any()
                else sc['cf_df']["Month"].max()
            ),
        })

    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Metric cards for the first scenario
    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WAL — Base Scenario",  f"{scenarios[0]['wal']:.2f} yrs",
              help="Weighted average time until principal is returned.")
    m2.metric("Yield — Base Scenario", f"{scenarios[0]['yield']*100:.3f}%",
              help="IRR of cash flows at the entered price.")
    m3.metric("Eff. Duration",          f"{scenarios[0]['eff_dur']:.2f} yrs",
              help="Price sensitivity accounting for prepayment response to rate changes.")
    m4.metric("Current Mortgage Rate",  f"{mortgage_rate:.2f}%",
              help="30Y fixed rate from FRED as of latest available date.")

    fastest = min(scenarios, key=lambda s: s["wal"])
    slowest = max(scenarios, key=lambda s: s["wal"])
    st.markdown(f"""
    <div class="interp-box">
    📌 <b>Interpretation:</b> At {fastest['psa']} PSA, the pool returns principal 
    in <b>{fastest['wal']:.2f} years</b> on a weighted average basis — nearly 
    {slowest['wal']/fastest['wal']:.1f}x faster than the {slowest['psa']} PSA scenario 
    ({slowest['wal']:.2f} years). Higher prepayment speeds compress WAL and reduce 
    duration, exposing the investor to reinvestment risk when rates are falling.
    </div>
    """, unsafe_allow_html=True)

    # ── Section E: WAL Comparison Chart ──────────────────────────────────────
    st.markdown("---")
    col_wal, col_decay = st.columns(2)

    with col_wal:
        fig_wal = plot_wal_comparison(scenarios)
        st.plotly_chart(fig_wal, use_container_width=True)
        st.markdown("""
        <div class="interp-box">
        📌 <b>Interpretation:</b> WAL compresses significantly at higher PSA speeds — 
        a core feature of MBS negative convexity. When rates fall and homeowners 
        refinance, the investor gets principal back faster but must reinvest at lower yields.
        </div>
        """, unsafe_allow_html=True)

    with col_decay:
        fig_decay = plot_balance_decay(scenarios)
        st.plotly_chart(fig_decay, use_container_width=True)
        st.markdown("""
        <div class="interp-box">
        📌 <b>Interpretation:</b> Steeper balance decay curves reflect faster prepayment 
        speeds. A pool at 400 PSA may be nearly fully paid down in half the time of a 
        100 PSA pool — drastically shortening the investor's effective hold period.
        </div>
        """, unsafe_allow_html=True)

    # ── Section F: Cash Flow & Principal Charts ───────────────────────────────
    st.markdown("---")
    col_cf, col_prin = st.columns(2)

    with col_cf:
        fig_cf = plot_cash_flows(scenarios)
        st.plotly_chart(fig_cf, use_container_width=True)
        st.markdown("""
        <div class="interp-box">
        📌 <b>Interpretation:</b> Higher PSA speeds produce a larger early cash flow 
        hump as prepayments spike in the first few years, followed by a sharper 
        dropoff as the pool balance is rapidly retired.
        </div>
        """, unsafe_allow_html=True)

    with col_prin:
        # Show principal breakdown for the middle scenario
        mid_sc = scenarios[len(scenarios) // 2]
        fig_prin = plot_principal_breakdown(mid_sc["cf_df"], mid_sc["psa"])
        st.plotly_chart(fig_prin, use_container_width=True)
        st.markdown(f"""
        <div class="interp-box">
        📌 <b>Interpretation:</b> At {mid_sc['psa']} PSA, prepayment principal 
        dominates total principal return from the first year — 
        scheduled amortization represents only a small fraction of 
        total principal paydown, which is typical for high-PSA scenarios.
        </div>
        """, unsafe_allow_html=True)

    # ── Section G: Detailed Cash Flow Table ──────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Detailed Cash Flow Table")

    tab_labels = [f"{sc['psa']} PSA" for sc in scenarios]
    tabs = st.tabs(tab_labels)

    for i, (tab, sc) in enumerate(zip(tabs, scenarios)):
        with tab:
            display_cf = sc["cf_df"].copy()
            # Format for display
            for col in ["Beginning Balance ($)", "Ending Balance ($)",
                        "Scheduled Interest ($)", "Scheduled Principal ($)",
                        "Prepay Principal ($)", "Total Principal ($)", "Total Cash Flow ($)"]:
                display_cf[col] = display_cf[col].map("${:,.0f}".format)
            display_cf["CPR (%)"] = display_cf["CPR (%)"].map("{:.3f}%".format)
            display_cf["SMM (%)"] = display_cf["SMM (%)"].map("{:.4f}%".format)

            st.dataframe(display_cf, use_container_width=True, hide_index=True, height=350)
            st.caption(f"Showing all {len(sc['cf_df'])} months for {sc['psa']} PSA scenario.")

# ── Methodology Expander ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Methodology — PSA Prepayment Model"):
    st.markdown("""
    ### What Is Prepayment Risk?

    Agency MBS investors receive monthly cash flows from pools of residential mortgages.
    Unlike a corporate bond, the timing of those cash flows is uncertain — homeowners
    can pay off their mortgage early by refinancing or selling their home. This is
    **prepayment risk**, and it is the defining risk of MBS investing.

    ---

    ### The PSA Benchmark

    The **Public Securities Association (PSA)** benchmark is the industry-standard
    framework for describing prepayment speeds. It is a ramp, not a fixed rate:

    - **100 PSA** means CPR ramps linearly from 0% at month 1 up to 6% at month 30,
      then stays flat at 6% for the remaining life of the pool.
    - Any other PSA speed scales this proportionally. For example:
      - 200 PSA = twice as fast; CPR reaches 12% by month 30 and stays there.
      - 50 PSA = half speed; CPR reaches only 3% by month 30.

    **Formula in plain terms:**
    CPR at month T = (PSA Speed / 100) times the lesser of (T / 30) or 1, times 6%

    The ramp reflects empirical seasoning — newly originated loans have low prepayment
    rates because borrowers do not refinance immediately after taking out a mortgage.

    ---

    ### From CPR to SMM

    **CPR (Conditional Prepayment Rate)** is an annual rate. To apply it monthly,
    we convert it to **SMM (Single Monthly Mortality)**:

    SMM = 1 minus (1 minus CPR) raised to the power of (1/12)

    The SMM is the fraction of the remaining pool balance that prepays in a given month.

    **Example:** At 100 PSA by month 30, CPR = 6%.
    SMM = 1 minus (1 minus 0.06)^(1/12) = approximately 0.514% per month.
    That means about 0.514% of the remaining pool balance prepays every month.

    ---

    ### Monthly Cash Flow Waterfall

    For each month, the cash flow components are calculated in this order:

    | Step | Component | How It Is Calculated |
    |---|---|---|
    | 1 | Interest | Beginning balance times (WAC divided by 12) |
    | 2 | Scheduled Principal | Standard mortgage amortization payment minus interest |
    | 3 | Prepayment Principal | SMM times (balance remaining after scheduled principal) |
    | 4 | Total Principal | Scheduled principal plus prepayment principal |
    | 5 | Total Cash Flow | Interest plus total principal |
    | 6 | Ending Balance | Beginning balance minus total principal |

    The ending balance becomes next month's beginning balance. This repeats until the
    pool is fully paid down.

    ---

    ### Weighted Average Life (WAL)

    WAL is the weighted average time in years until each dollar of principal is
    returned to the investor:

    WAL = Sum of (Principal returned in month T times T in years) divided by Total Principal

    WAL is the primary duration measure for MBS on a trading desk. It tells the investor
    how long their principal is effectively at risk. A lower WAL means principal comes
    back faster — which is good if you can reinvest at higher rates, but bad if rates
    have fallen (reinvestment risk).

    ---

    ### Effective Duration

    Unlike modified duration, effective duration accounts for the optionality embedded
    in MBS. When rates fall, homeowners refinance faster (contraction risk). When rates
    rise, they refinance slower (extension risk).

    We estimate effective duration by repricing the pool under plus and minus 100bp
    shocks, with the PSA speed adjusted by plus or minus 50 PSA per 100bps to reflect
    changing refinancing incentives:

    Effective Duration = negative of (Price at +100bps minus Price at -100bps)
    divided by (2 times Base Price times 0.01)

    This is why MBS exhibit **negative convexity** — when rates fall and the bond
    should appreciate, prepayments accelerate and cap the price gain. The investor
    effectively sold a call option to the homeowner.

    ---

    ### Key Relationships to Remember

    | If rates... | Prepayments... | WAL... | Duration... | Risk |
    |---|---|---|---|---|
    | Fall sharply | Accelerate | Shorten | Compresses | Contraction / reinvestment |
    | Rise sharply | Slow down | Extend | Lengthens | Extension / price loss |
    | Stay flat | Follow PSA ramp | As modeled | Stable | Minimal |
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Fixed Income Analytics Suite · Tool 2: Agency MBS Cash Flow & Prepayment Model</span>
    <span>Your Name · your.email@example.com</span>
</div>
""", unsafe_allow_html=True)
