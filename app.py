"""
Fixed Income Analytics Suite — Landing Page
Entry point for the multi-page Streamlit application.
"""

import streamlit as st

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fixed Income Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Institutional color theme ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global background and text */
    .stApp { background-color: #0a1628; color: #e8edf5; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1f3c;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c8d6e8 !important; }

    /* Headers */
    h1, h2, h3 { color: #e8edf5 !important; }

    /* Cards */
    .tool-card {
        background: linear-gradient(135deg, #0d1f3c 0%, #122444 100%);
        border: 1px solid #1e3a5f;
        border-left: 4px solid #4a9eff;
        border-radius: 6px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }
    .tool-card h3 { color: #4a9eff !important; margin-bottom: 8px; font-size: 1.05rem; }
    .tool-card p  { color: #a8bdd4; font-size: 0.92rem; line-height: 1.6; margin: 0; }

    /* Divider */
    hr { border-color: #1e3a5f; }

    /* Footer */
    .footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #0a1628;
        border-top: 1px solid #1e3a5f;
        padding: 8px 32px;
        font-size: 0.78rem;
        color: #5a7a9a;
        display: flex;
        justify-content: space-between;
    }

    /* Metric label */
    [data-testid="stMetricLabel"] { color: #a8bdd4 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Fixed Income Suite")
    st.markdown("---")
    st.markdown("**Navigate to a tool:**")
    st.markdown("Use the pages listed above.")
    st.markdown("---")
    st.markdown("*Built for PNC CIO — ALM Investments*")

# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 40px 0 20px 0;">
    <div style="font-size:0.85rem; color:#4a9eff; letter-spacing:0.12em; 
                text-transform:uppercase; margin-bottom:10px;">
        PNC Chief Investment Office · ALM Investments Desk
    </div>
    <h1 style="font-size:2.4rem; font-weight:700; color:#e8edf5; 
               margin:0 0 12px 0; line-height:1.2;">
        Fixed Income Analytics Suite
    </h1>
    <p style="font-size:1.05rem; color:#a8bdd4; max-width:700px; line-height:1.7; margin:0;">
        A front-office quantitative toolkit for yield curve analysis, agency MBS cash flow 
        modeling, and asset-liability management. Built to institutional standards using 
        live Federal Reserve data.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Tool cards ───────────────────────────────────────────────────────────────
st.markdown("### Tools in This Suite")
st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="tool-card">
        <h3>📈 Tool 1 · Yield Curve & Bond Risk Engine</h3>
        <p>
            Bootstraps a live US Treasury spot curve from FRED par yields. 
            Prices a user-defined fixed-rate bond portfolio and computes DV01, 
            modified duration, and convexity for each position. Stress-tests 
            portfolio value across five rate scenarios including parallel shifts, 
            bull steepener, and bear flattener.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tool-card">
        <h3>🏠 Tool 2 · Agency MBS Cash Flow & Prepayment Model</h3>
        <p>
            Models monthly cash flows for a generic agency MBS pool using a 
            from-scratch PSA prepayment framework. Computes scheduled principal, 
            prepayment principal, and interest across the pool life. Outputs 
            weighted average life, yield, and effective duration across user-defined 
            PSA speed scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tool-card">
        <h3>🏦 Tool 3 · ALM NII Forecasting & Balance Sheet Stress</h3>
        <p>
            A structurally correct asset-liability model anchored to PNC's 
            public balance sheet. Computes repricing gaps, simulates net interest 
            income under four rate scenarios, and generates a one-page management 
            commentary formatted for CFO-level review.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Data sources & links ─────────────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("**Data Sources**")
    st.markdown("""
    All market data is pulled live from the 
    [FRED® API](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis).  
    Balance sheet inputs are sourced from PNC Financial Services Group public filings.
    """)

with col_b:
    st.markdown("**Source Code**")
    st.markdown("[GitHub Repository](#) *(https://github.com/prakashbala0508)*")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Fixed Income Analytics Suite · PNC CIO Internship Project</span>
    <span>Prakash Balasubramanian · prakash.bala.work@gmail.com</span>
</div>
""", unsafe_allow_html=True)
