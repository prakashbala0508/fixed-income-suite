"""
Page 1 — Yield Curve & Bond Risk Engine
Pulls live US Treasury yields from FRED, bootstraps a spot curve,
prices a user-defined bond portfolio, and stress-tests across 5 rate scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from scipy.interpolate import CubicSpline
from datetime import datetime

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yield Curve & Bond Risk | FI Suite",
    page_icon="📈",
    layout="wide",
)

# ── Shared institutional theme ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a1628; color: #e8edf5; }
    [data-testid="stSidebar"] {
        background-color: #0d1f3c;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c8d6e8 !important; }
    h1, h2, h3, h4 { color: #e8edf5 !important; }
    .stMetric { background: #0d1f3c; border: 1px solid #1e3a5f; 
                border-radius:6px; padding:12px; }
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
    .stAlert { background: #0d1f3c !important; border: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── FRED tickers and tenor labels ────────────────────────────────────────────
FRED_TICKERS = {
    "DGS1MO": 1/12,
    "DGS3MO": 3/12,
    "DGS6MO": 6/12,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS5":   5.0,
    "DGS10":  10.0,
    "DGS30":  30.0,
}

TENOR_LABELS = {
    1/12: "1M", 3/12: "3M", 6/12: "6M",
    1.0: "1Y", 2.0: "2Y", 5.0: "5Y",
    10.0: "10Y", 30.0: "30Y"
}

COLORS = {
    "navy":  "#0a1628",
    "blue":  "#4a9eff",
    "slate": "#a8bdd4",
    "white": "#e8edf5",
    "green": "#2ecc71",
    "red":   "#e74c3c",
    "amber": "#f39c12",
}


# ── Section 1: Data Fetching ─────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_treasury_yields() -> pd.DataFrame:
    """
    Fetch the most recent US Treasury par yields from FRED.
    Returns a DataFrame with tenor (years) and yield (%) columns.
    Caches results for 1 hour to avoid redundant API calls.
    """
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])
    records = []
    for ticker, tenor in FRED_TICKERS.items():
        series = fred.get_series(ticker)
        series = series.dropna()
        if len(series) > 0:
            latest_yield = float(series.iloc[-1])
            latest_date  = series.index[-1].strftime("%Y-%m-%d")
            records.append({
                "ticker": ticker,
                "tenor_years": tenor,
                "tenor_label": TENOR_LABELS[tenor],
                "par_yield_pct": latest_yield,
                "as_of_date": latest_date,
            })
    df = pd.DataFrame(records).sort_values("tenor_years").reset_index(drop=True)
    return df


# ── Section 2: Spot Curve Bootstrapping ─────────────────────────────────────

def bootstrap_spot_curve(par_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap a zero-coupon (spot) curve from par yields using the 
    sequential bootstrapping method.

    For tenors <= 1 year: spot rate = par yield (no coupon payments).
    For tenors > 1 year: solve for the spot rate that correctly discounts
    all cash flows (coupons + principal) to par using previously bootstrapped spots.

    Args:
        par_df: DataFrame with columns [tenor_years, par_yield_pct]

    Returns:
        DataFrame with columns [tenor_years, tenor_label, par_yield_pct, spot_rate_pct]
    """
    tenors = par_df["tenor_years"].values
    par_yields = par_df["par_yield_pct"].values / 100  # convert % to decimal

    spot_rates = np.zeros(len(tenors))

    # Build a cubic spline of spot rates as we bootstrap
    # Start with an empty dict; fill in sequentially
    bootstrapped = {}  # tenor -> spot rate (decimal)

    for i, (T, c) in enumerate(zip(tenors, par_yields)):
        if T <= 1.0:
            # Short end: discount factor is straightforward
            # spot_rate = par_yield for sub-annual / 1Y par bonds
            spot_rates[i] = c
            bootstrapped[T] = c
        else:
            # Number of annual coupon periods
            n_periods = int(round(T))
            # Assume annual coupon payments for simplicity (standard for Treasuries)
            coupon = c  # coupon rate = par yield when priced at par

            # Sum of PV of intermediate coupon payments using known spot rates
            pv_coupons = 0.0
            for k in range(1, n_periods):
                t_k = float(k)
                # Interpolate spot rate at t_k from bootstrapped points
                if len(bootstrapped) >= 2:
                    tenors_known = np.array(sorted(bootstrapped.keys()))
                    spots_known  = np.array([bootstrapped[t] for t in tenors_known])
                    cs = CubicSpline(tenors_known, spots_known, extrapolate=True)
                    r_k = float(cs(t_k))
                elif len(bootstrapped) == 1:
                    # Only one point: flat extrapolation
                    r_k = list(bootstrapped.values())[0]
                else:
                    r_k = c
                r_k = max(r_k, 0.0001)  # floor at 1bp
                pv_coupons += coupon / (1 + r_k) ** t_k

            # Solve for the terminal spot rate r_T such that:
            # pv_coupons + (1 + coupon) / (1 + r_T)^T = 1  (par)
            # => r_T = ((1 + coupon) / (1 - pv_coupons))^(1/T) - 1
            terminal_cf = 1.0 + coupon
            denom = 1.0 - pv_coupons
            if denom <= 0:
                denom = 0.0001  # numerical safety floor
            r_T = (terminal_cf / denom) ** (1.0 / T) - 1.0
            spot_rates[i] = r_T
            bootstrapped[T] = r_T

    result = par_df.copy()
    result["spot_rate_pct"] = spot_rates * 100  # back to %
    return result


def interpolate_spot_rate(spot_df: pd.DataFrame, t: float) -> float:
    """
    Interpolate a spot rate at an arbitrary tenor t (in years)
    using a cubic spline fit to the bootstrapped spot curve.

    Args:
        spot_df: DataFrame with [tenor_years, spot_rate_pct]
        t: Target tenor in years

    Returns:
        Spot rate as a decimal (not percent)
    """
    tenors = spot_df["tenor_years"].values
    spots  = spot_df["spot_rate_pct"].values / 100
    cs = CubicSpline(tenors, spots, extrapolate=True)
    return float(cs(t))



# ── Section 3: Bond Pricing & Risk Metrics ───────────────────────────────────

def price_bond(coupon_rate: float, maturity_years: float, face_value: float,
               spot_df: pd.DataFrame, shock_bps: float = 0.0,
               custom_spot_df: pd.DataFrame = None) -> float:
    """
    Price a fixed-rate bond by discounting each cash flow at the appropriate
    spot rate (with optional parallel yield curve shock in basis points).

    Cash flows: coupon payments every 6 months + face value at maturity.
    Assumes semi-annual coupon payments (US Treasury convention).

    Args:
        coupon_rate:    Annual coupon rate as a decimal (e.g., 0.045 for 4.5%)
        maturity_years: Years to maturity
        face_value:     Face/par value in dollars
        spot_df:        Bootstrapped spot curve DataFrame
        shock_bps:      Parallel shift to apply to the spot curve in basis points
        custom_spot_df: Optional pre-shocked spot curve (used for non-parallel scenarios)

    Returns:
        Dirty price of the bond in dollars
    """
    curve = custom_spot_df if custom_spot_df is not None else spot_df
    shock_dec = shock_bps / 10000.0

    # Semi-annual payment schedule
    n_periods = int(maturity_years * 2)
    semi_coupon = (coupon_rate * face_value) / 2.0

    price = 0.0
    for k in range(1, n_periods + 1):
        t = k / 2.0  # time in years
        r = interpolate_spot_rate(curve, t) + shock_dec
        r = max(r, 0.0001)  # floor at 1bp
        cf = semi_coupon
        if k == n_periods:
            cf += face_value  # principal repayment at maturity
        price += cf / (1 + r / 2.0) ** k  # semi-annual compounding

    return price


def compute_dv01(coupon_rate: float, maturity_years: float, face_value: float,
                 spot_df: pd.DataFrame) -> float:
    """
    Compute DV01 (Dollar Value of 1 basis point) as the absolute price change
    for a 1bp parallel shift in the yield curve.

    DV01 = |Price(+1bp) - Price(-1bp)| / 2

    Returns:
        DV01 in dollars (positive number)
    """
    p_up   = price_bond(coupon_rate, maturity_years, face_value, spot_df, shock_bps=+1)
    p_down = price_bond(coupon_rate, maturity_years, face_value, spot_df, shock_bps=-1)
    return abs(p_up - p_down) / 2.0


def compute_modified_duration(coupon_rate: float, maturity_years: float,
                               face_value: float, spot_df: pd.DataFrame) -> float:
    """
    Compute modified duration using the price sensitivity formula.

    Modified Duration ≈ -(1/P) * dP/dy
    Approximated numerically: -(P_up - P_down) / (2 * P_base * 0.0001)

    Returns:
        Modified duration in years
    """
    p_base = price_bond(coupon_rate, maturity_years, face_value, spot_df)
    p_up   = price_bond(coupon_rate, maturity_years, face_value, spot_df, shock_bps=+1)
    p_down = price_bond(coupon_rate, maturity_years, face_value, spot_df, shock_bps=-1)
    mod_dur = -(p_up - p_down) / (2 * p_base * 0.0001)
    return mod_dur


def compute_convexity(coupon_rate: float, maturity_years: float,
                      face_value: float, spot_df: pd.DataFrame) -> float:
    """
    Compute convexity — the second-order price sensitivity to yield changes.

    Convexity ≈ (P_up + P_down - 2*P_base) / (P_base * (0.0001)^2)
    Basis point shock of 100bps used for numerical stability.

    Returns:
        Convexity (dimensionless)
    """
    p_base = price_bond(coupon_rate, maturity_years, face_value, spot_df)
    p_up   = price_bond(coupon_rate, maturity_years, face_value, spot_df, shock_bps=+100)
    p_down = price_bond(coupon_rate, maturity_years, face_value, spot_df, shock_bps=-100)
    dy = 100 / 10000.0  # 100bps in decimal
    convexity = (p_up + p_down - 2 * p_base) / (p_base * dy ** 2)
    return convexity


def scenario_pnl(coupon_rate: float, maturity_years: float, face_value: float,
                 spot_df: pd.DataFrame, scenario: dict) -> float:
    """
    Compute the P&L (change in market value) for a bond under a given
    yield curve scenario.

    Non-parallel scenarios (steepener/flattener) apply different shocks
    to different parts of the curve.

    Args:
        scenario: dict with keys 'type' ('parallel' or 'structured') and
                  either 'shock_bps' (parallel) or 'shift_fn' (function of tenor)

    Returns:
        P&L in dollars (negative = loss)
    """
    p_base = price_bond(coupon_rate, maturity_years, face_value, spot_df)

    if scenario["type"] == "parallel":
        p_stressed = price_bond(coupon_rate, maturity_years, face_value,
                                spot_df, shock_bps=scenario["shock_bps"])
    else:
        # Build a shifted spot curve for structured scenarios
        stressed = spot_df.copy()
        stressed["spot_rate_pct"] = stressed.apply(
            lambda row: row["spot_rate_pct"] + scenario["shift_fn"](row["tenor_years"]),
            axis=1
        )
        stressed["par_yield_pct"] = par_df["par_yield_pct"] + stressed.apply(
            lambda row: scenario["shift_fn"](row["tenor_years"]), axis=1
        )
        p_stressed = price_bond(coupon_rate, maturity_years, face_value,
                                spot_df, custom_spot_df=stressed)

    return p_stressed - p_base



# ── Section 4: Stress Scenarios ─────────────────────────────────────────────

def build_scenarios() -> list:
    """
    Define the 5 standard yield curve stress scenarios used on fixed income desks.

    Parallel shifts: every point on the curve moves by the same number of bps.
    Bull steepener: short rates fall more than long rates (Fed cutting cycle).
    Bear flattener: short rates rise more than long rates (Fed hiking cycle).

    Returns:
        List of scenario dicts with name, type, and shift parameters.
    """
    return [
        {
            "name": "Parallel +100bps",
            "type": "parallel",
            "shock_bps": +100,
            "color": COLORS["red"],
        },
        {
            "name": "Parallel -100bps",
            "type": "parallel",
            "shock_bps": -100,
            "color": COLORS["green"],
        },
        {
            "name": "Parallel +200bps",
            "type": "parallel",
            "shock_bps": +200,
            "color": "#c0392b",
        },
        {
            "name": "Bull Steepener",
            "type": "structured",
            # Short end -75bps, long end -15bps — curve steepens as rates fall
            "shift_fn": lambda T: -75 + min(60, 60 * (T / 10)),
            "color": "#27ae60",
            "description": "Short rates -75bps, long rates -15bps",
        },
        {
            "name": "Bear Flattener",
            "type": "structured",
            # Short end +100bps, long end +25bps — curve flattens as rates rise
            "shift_fn": lambda T: 100 - min(75, 75 * (T / 10)),
            "color": COLORS["amber"],
            "description": "Short rates +100bps, long rates +25bps",
        },
    ]


# ── Section 5: Charting Functions ───────────────────────────────────────────

def plot_yield_curves(par_df: pd.DataFrame, spot_df: pd.DataFrame) -> go.Figure:
    """
    Plot the live par curve and bootstrapped spot curve on the same axes.
    """
    tenors_fine = np.linspace(par_df["tenor_years"].min(),
                               par_df["tenor_years"].max(), 200)
    spots_fine = []
    for t in tenors_fine:
        spots_fine.append(interpolate_spot_rate(spot_df, t) * 100)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=par_df["tenor_years"], y=par_df["par_yield_pct"],
        mode="lines+markers", name="Par Yield Curve",
        line=dict(color=COLORS["slate"], width=2, dash="dot"),
        marker=dict(size=7, color=COLORS["slate"]),
    ))

    fig.add_trace(go.Scatter(
        x=tenors_fine, y=spots_fine,
        mode="lines", name="Bootstrapped Spot Curve",
        line=dict(color=COLORS["blue"], width=2.5),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(text="US Treasury Yield Curves", font=dict(size=15,
                   color=COLORS["white"]), x=0.02),
        xaxis=dict(title="Tenor (Years)", gridcolor="#1e3a5f", zeroline=False),
        yaxis=dict(title="Yield (%)", gridcolor="#1e3a5f", zeroline=False,
                   tickformat=".2f"),
        legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=60, r=30, t=50, b=50),
        height=380,
    )
    return fig


def plot_scenario_curves(par_df: pd.DataFrame, scenarios: list) -> go.Figure:
    """
    Overlay the base curve and all stress-scenario shifted curves on one chart.
    """
    tenors_fine = np.linspace(par_df["tenor_years"].min(),
                               par_df["tenor_years"].max(), 200)
    base_yields = par_df["par_yield_pct"].values
    cs_base = CubicSpline(par_df["tenor_years"].values, base_yields, extrapolate=True)

    fig = go.Figure()

    # Base curve
    fig.add_trace(go.Scatter(
        x=tenors_fine, y=cs_base(tenors_fine),
        mode="lines", name="Base Curve",
        line=dict(color=COLORS["white"], width=2.5),
    ))

    for sc in scenarios:
        if sc["type"] == "parallel":
            shifted = cs_base(tenors_fine) + sc["shock_bps"] / 100
        else:
            shifts = np.array([sc["shift_fn"](t) / 100 for t in tenors_fine])
            shifted = cs_base(tenors_fine) + shifts

        fig.add_trace(go.Scatter(
            x=tenors_fine, y=shifted,
            mode="lines", name=sc["name"],
            line=dict(color=sc["color"], width=1.8, dash="dot"),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(text="Yield Curve — Stress Scenarios vs Base", font=dict(
                   size=15, color=COLORS["white"]), x=0.02),
        xaxis=dict(title="Tenor (Years)", gridcolor="#1e3a5f", zeroline=False),
        yaxis=dict(title="Yield (%)", gridcolor="#1e3a5f", zeroline=False,
                   tickformat=".2f"),
        legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=60, r=30, t=50, b=50),
        height=400,
    )
    return fig


def plot_pnl_waterfall(scenario_names: list, portfolio_pnls: list) -> go.Figure:
    """
    Horizontal bar chart showing portfolio P&L (in dollars) under each scenario.
    """
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in portfolio_pnls]

    fig = go.Figure(go.Bar(
        x=portfolio_pnls,
        y=scenario_names,
        orientation="h",
        marker_color=colors,
        text=[f"${v:,.0f}" for v in portfolio_pnls],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=12),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color=COLORS["white"], family="monospace"),
        title=dict(text="Portfolio P&L by Scenario (USD)", font=dict(
                   size=15, color=COLORS["white"]), x=0.02),
        xaxis=dict(title="P&L (USD)", gridcolor="#1e3a5f", zeroline=True,
                   zerolinecolor=COLORS["slate"], tickformat="$,.0f"),
        yaxis=dict(gridcolor="#1e3a5f"),
        margin=dict(l=160, r=80, t=50, b=50),
        height=320,
    )
    return fig



# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI — Page Layout
# ═══════════════════════════════════════════════════════════════════════════

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 28px 0 12px 0;">
    <div style="font-size:0.8rem; color:#4a9eff; letter-spacing:0.1em; 
                text-transform:uppercase; margin-bottom:8px;">
        Tool 1 of 3 · Fixed Income Analytics Suite
    </div>
    <h1 style="font-size:2rem; font-weight:700; color:#e8edf5; margin:0 0 8px 0;">
        Yield Curve & Bond Risk Engine
    </h1>
    <p style="color:#a8bdd4; font-size:0.95rem; margin:0;">
        Live US Treasury par yields from FRED · Bootstrapped spot curve · 
        Portfolio DV01, Duration & Convexity · 5-scenario stress test
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load FRED data ───────────────────────────────────────────────────────────
with st.spinner("Fetching live Treasury yields from FRED..."):
    try:
        par_df   = fetch_treasury_yields()
        spot_df  = bootstrap_spot_curve(par_df)
        data_ok  = True
        as_of    = par_df["as_of_date"].iloc[-1]
    except Exception as e:
        st.error(f"⚠️ FRED API error: {e}")
        st.info("Check that your FRED API key is set correctly in `.streamlit/secrets.toml`")
        data_ok = False
        st.stop()

# ── Section A: Live Yield Curve ───────────────────────────────────────────────
st.subheader("📊 Live US Treasury Yield Curve")
st.caption(f"Data as of: {as_of} · Source: FRED (Federal Reserve Bank of St. Louis)")

col_chart, col_table = st.columns([3, 2])

with col_chart:
    fig_curves = plot_yield_curves(par_df, spot_df)
    st.plotly_chart(fig_curves, use_container_width=True)
    st.markdown("""
    <div class="interp-box">
    📌 <b>Interpretation:</b> The spot (zero-coupon) curve trades above the par curve 
    when the yield curve is upward-sloping — bootstrapping removes the coupon reinvestment 
    assumption and provides the pure time-value discount rate for each maturity.
    </div>
    """, unsafe_allow_html=True)

with col_table:
    display_df = spot_df[["tenor_label", "par_yield_pct", "spot_rate_pct"]].copy()
    display_df.columns = ["Tenor", "Par Yield (%)", "Spot Rate (%)"]
    display_df["Par Yield (%)"]  = display_df["Par Yield (%)"].map("{:.3f}".format)
    display_df["Spot Rate (%)"]  = display_df["Spot Rate (%)"].map("{:.3f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=310)

st.markdown("---")

# ── Section B: Bond Portfolio Input ─────────────────────────────────────────
st.subheader("📋 Define Your Bond Portfolio")
st.caption("Enter up to 5 fixed-rate bonds. All positions are priced off the live spot curve above.")

n_bonds = st.number_input("Number of bonds in portfolio", min_value=1, max_value=5,
                            value=3, step=1)

bond_inputs = []
col_headers = st.columns([2.5, 2, 2, 2])
col_headers[0].markdown("**Bond Name**")
col_headers[1].markdown("**Annual Coupon (%)**")
col_headers[2].markdown("**Maturity (Years)**")
col_headers[3].markdown("**Face Value ($)**")

defaults = [
    ("2Y Treasury",  4.50, 2,  1_000_000),
    ("5Y Treasury",  4.25, 5,  2_000_000),
    ("10Y Treasury", 4.375, 10, 1_500_000),
    ("Corp Bond A",  5.50, 7,    500_000),
    ("Corp Bond B",  6.00, 15,   750_000),
]

for i in range(int(n_bonds)):
    d = defaults[i]
    cols = st.columns([2.5, 2, 2, 2])
    name   = cols[0].text_input("", value=d[0], key=f"name_{i}", label_visibility="collapsed")
    coupon = cols[1].number_input("", value=d[1], min_value=0.0, max_value=20.0,
                                   step=0.125, format="%.3f", key=f"cpn_{i}",
                                   label_visibility="collapsed")
    mat    = cols[2].number_input("", value=float(d[2]), min_value=0.25, max_value=30.0,
                                   step=0.5, format="%.1f", key=f"mat_{i}",
                                   label_visibility="collapsed")
    fv     = cols[3].number_input("", value=float(d[3]), min_value=1000.0,
                                   step=10000.0, format="%.0f", key=f"fv_{i}",
                                   label_visibility="collapsed")
    bond_inputs.append({
        "name": name,
        "coupon_rate": coupon / 100,
        "maturity_years": mat,
        "face_value": fv,
    })

if st.button("▶ Price Portfolio & Compute Risk Metrics", type="primary"):

    # ── Section C: Risk Summary Table ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Portfolio Risk Summary")

    risk_rows = []
    for b in bond_inputs:
        price   = price_bond(b["coupon_rate"], b["maturity_years"],
                             b["face_value"], spot_df)
        dv01    = compute_dv01(b["coupon_rate"], b["maturity_years"],
                               b["face_value"], spot_df)
        mod_dur = compute_modified_duration(b["coupon_rate"], b["maturity_years"],
                                            b["face_value"], spot_df)
        convex  = compute_convexity(b["coupon_rate"], b["maturity_years"],
                                    b["face_value"], spot_df)
        price_pct = price / b["face_value"] * 100

        risk_rows.append({
            "Bond": b["name"],
            "Coupon (%)": f"{b['coupon_rate']*100:.3f}",
            "Maturity (Yrs)": f"{b['maturity_years']:.1f}",
            "Face Value ($)": f"{b['face_value']:,.0f}",
            "Market Price ($)": f"{price:,.2f}",
            "Price (% of Par)": f"{price_pct:.3f}",
            "DV01 ($)": f"{dv01:,.2f}",
            "Mod. Duration (Yrs)": f"{mod_dur:.3f}",
            "Convexity": f"{convex:.3f}",
        })

    risk_df = pd.DataFrame(risk_rows)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # ── Portfolio totals ──────────────────────────────────────────────────────
    total_mv   = sum(float(r["Market Price ($)"].replace(",",""))   for r in risk_rows)
    total_fv   = sum(b["face_value"]                                 for b in bond_inputs)
    total_dv01 = sum(float(r["DV01 ($)"].replace(",",""))            for r in risk_rows)
    # Duration of portfolio = weighted average by market value
    port_dur   = sum(
        compute_modified_duration(b["coupon_rate"], b["maturity_years"],
                                  b["face_value"], spot_df) *
        price_bond(b["coupon_rate"], b["maturity_years"], b["face_value"], spot_df)
        for b in bond_inputs
    ) / total_mv

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Market Value", f"${total_mv:,.0f}")
    m2.metric("Total Face Value",   f"${total_fv:,.0f}")
    m3.metric("Portfolio DV01",     f"${total_dv01:,.2f}")
    m4.metric("Wtd Avg Duration",   f"{port_dur:.3f} yrs")

    st.markdown("""
    <div class="interp-box">
    📌 <b>Interpretation:</b> A portfolio DV01 of ${:,.0f} means the total position 
    gains or loses approximately that amount for every 1 basis point move in rates — 
    short duration positions benefit from rising rates while long duration positions 
    carry more rate risk.
    </div>
    """.format(total_dv01), unsafe_allow_html=True)

    # ── Section D: Stress Testing ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Yield Curve Stress Test")

    scenarios = build_scenarios()

    # Compute P&L per bond per scenario
    stress_rows = []
    portfolio_pnls = []

    for sc in scenarios:
        row = {"Scenario": sc["name"]}
        port_pnl = 0.0
        for b in bond_inputs:
            p_base = price_bond(b["coupon_rate"], b["maturity_years"],
                                b["face_value"], spot_df)
            if sc["type"] == "parallel":
                p_stressed = price_bond(b["coupon_rate"], b["maturity_years"],
                                        b["face_value"], spot_df,
                                        shock_bps=sc["shock_bps"])
            else:
                stressed_spot = spot_df.copy()
                stressed_spot["spot_rate_pct"] = stressed_spot.apply(
                    lambda r: r["spot_rate_pct"] + sc["shift_fn"](r["tenor_years"]),
                    axis=1
                )
                p_stressed = price_bond(b["coupon_rate"], b["maturity_years"],
                                        b["face_value"], spot_df,
                                        custom_spot_df=stressed_spot)
            pnl = p_stressed - p_base
            port_pnl += pnl
            row[b["name"]] = f"${pnl:+,.0f}"

        row["Portfolio P&L ($)"] = f"${port_pnl:+,.0f}"
        stress_rows.append(row)
        portfolio_pnls.append(port_pnl)

    stress_df = pd.DataFrame(stress_rows)

    # Color the P&L column
    st.dataframe(stress_df, use_container_width=True, hide_index=True)

    # Charts side by side
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_sc = plot_scenario_curves(par_df, scenarios)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("""
        <div class="interp-box">
        📌 <b>Interpretation:</b> The bear flattener compresses the 2s10s spread, 
        reflecting a Fed hiking cycle where the front end reprices faster than the 
        long end — the most common regime during aggressive tightening.
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        scenario_names = [sc["name"] for sc in scenarios]
        fig_pnl = plot_pnl_waterfall(scenario_names, portfolio_pnls)
        st.plotly_chart(fig_pnl, use_container_width=True)
        worst_sc = scenarios[portfolio_pnls.index(min(portfolio_pnls))]["name"]
        worst_pnl = min(portfolio_pnls)
        st.markdown(f"""
        <div class="interp-box">
        📌 <b>Interpretation:</b> The portfolio is most exposed to the 
        <b>{worst_sc}</b> scenario with a P&L impact of 
        <b>${worst_pnl:+,.0f}</b> — indicating the position is net long duration 
        and would benefit from a rate rally.
        </div>
        """, unsafe_allow_html=True)

# ── Methodology Expander ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Methodology — How This Tool Works"):
    st.markdown("""
    ### Yield Curve Construction

    **Par Yields** are fetched directly from FRED for 8 tenors on the US Treasury curve
    (1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y). These represent the yields on bonds that are
    currently priced at par (face value).

    **Bootstrapping the Spot Curve** converts par yields into zero-coupon (spot) rates
    using sequential stripping. The logic works as follows:

    - For tenors of 1 year or less: the spot rate equals the par yield directly,
      because there are no intermediate coupon payments to account for.
    - For tenors beyond 1 year: we solve for the terminal spot rate that satisfies
      the no-arbitrage condition — the present value of all coupons plus the final
      principal payment must equal par (100).

    In plain terms: we already know the discount rates for years 1 through N-1 from
    earlier bootstrapping steps. We use those to price all intermediate coupons, then
    solve algebraically for the one remaining unknown — the spot rate at year N.

    A cubic spline is fitted through the bootstrapped points to interpolate spot rates
    at non-benchmark tenors (for example, 3.5 years).

    ---

    ### Bond Pricing

    Each bond is priced by discounting its semi-annual cash flows at the interpolated
    spot rate for the corresponding payment date. Coupon payments occur every 6 months;
    the final payment includes both the last coupon and the full face value.

    The price is the sum of all discounted cash flows. When the coupon rate equals
    the spot rate, the bond prices at par. A higher coupon than the spot rate means
    the bond prices above par (premium); a lower coupon means below par (discount).

    ---

    ### Risk Metrics

    | Metric | What It Measures | How It Is Computed |
    |---|---|---|
    | DV01 | Dollar loss or gain per 1bp move in rates | Absolute price difference between +1bp and -1bp shocks, divided by 2 |
    | Modified Duration | % price change per 100bp move in rates | Numerical derivative of price with respect to yield, divided by price |
    | Convexity | Curvature of the price-yield relationship | Second derivative of price; positive convexity means gains exceed losses for equal rate moves |

    A portfolio DV01 is simply the sum of individual bond DV01s.
    Portfolio duration is the market-value-weighted average of individual durations.

    ---

    ### Stress Scenarios

    | Scenario | Short End Shift | Long End Shift | What It Represents |
    |---|---|---|---|
    | Parallel +100bps | +100bps | +100bps | Moderate rate hike cycle |
    | Parallel -100bps | -100bps | -100bps | Moderate rate cut cycle |
    | Parallel +200bps | +200bps | +200bps | Severe shock, ALM stress floor |
    | Bull Steepener | -75bps | -15bps | Fed cutting; long end anchored |
    | Bear Flattener | +100bps | +25bps | Fed hiking; long end lags |
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Fixed Income Analytics Suite · Tool 1: Yield Curve & Bond Risk Engine</span>
    <span>Prakash Balasubramanian · prakash.bala.work@gmail.com</span>
</div>
""", unsafe_allow_html=True)
