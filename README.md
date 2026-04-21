# Fixed Income Analytics Suite
### Built for PNC Financial Services Group — Chief Investment Office, ALM Investments Desk

A professional-grade, multi-page quantitative analytics application built in Python and 
deployed via Streamlit. All market data is pulled live from the Federal Reserve Economic 
Data (FRED) API. Balance sheet data is sourced from PNC's public earnings releases.

**Live App:** [Launch Fixed Income Analytics Suite](https://your-app-url.streamlit.app)

---

## Tools in This Suite

### Tool 1 — Yield Curve & Bond Risk Engine
Pulls live US Treasury par yields from FRED (1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y) and 
bootstraps a zero-coupon spot curve. Prices a user-defined fixed-rate bond portfolio and 
computes DV01, modified duration, and convexity for each position. Stress-tests total 
portfolio value across five rate scenarios.

**Key calculations:**

Spot curve bootstrapping solves sequentially for each zero-coupon rate using the 
no-arbitrage condition that the present value of all cash flows on a par bond equals 
face value. For tenors beyond one year:

    r_n = ( (1 + coupon) / (1 - sum of discounted intermediate coupons) ) ^ (1/n) - 1

Bond price is the sum of discounted semi-annual cash flows:

    Price = sum[ (C/2) / (1 + r_t/2)^t ] + FV / (1 + r_T/2)^2T

Risk metrics are computed numerically:

    DV01           = | Price(+1bp) - Price(-1bp) | / 2
    Mod. Duration  = -( Price(+1bp) - Price(-1bp) ) / ( 2 * Price * 0.0001 )
    Convexity      = ( Price(+100bp) + Price(-100bp) - 2*Price ) / ( Price * 0.01^2 )

**Stress scenarios:** Parallel +100bps, Parallel -100bps, Parallel +200bps, 
Bull Steepener (short -75bps / long -15bps), Bear Flattener (short +100bps / long +25bps)

---

### Tool 2 — Agency MBS Cash Flow & Prepayment Model
Models monthly cash flows for a generic agency MBS pool using a from-scratch PSA 
prepayment framework — no QuantLib or black-box libraries used for core math.

**PSA prepayment model:**

The PSA benchmark ramps CPR linearly from 0% at month 1 to 6% at month 30, 
then holds flat. Any PSA speed scales this proportionally:

    CPR(t) = (PSA / 100) * min(t / 30, 1) * 6%

Monthly SMM is derived from CPR:

    SMM = 1 - (1 - CPR)^(1/12)

**Monthly cash flow waterfall:**

| Step | Component | Formula |
|------|-----------|---------|
| 1 | Interest | Balance * WAC / 12 |
| 2 | Scheduled Principal | Standard amortization payment minus interest |
| 3 | Prepayment Principal | SMM * (Balance - Scheduled Principal) |
| 4 | Total Principal | Scheduled + Prepayment |
| 5 | Ending Balance | Beginning Balance - Total Principal |

**Analytics computed:** Weighted Average Life (WAL), yield (monthly IRR * 12 via 
Newton-Raphson), effective duration (price sensitivity with PSA speed adjustment 
of +/- 50 PSA per 100bps shock to capture negative convexity).

    WAL = sum( Principal_t * t/12 ) / sum( Principal_t )

**Live data:** Current 30-year fixed mortgage rate from FRED (MORTGAGE30US) used 
to contextualize pool WAC input.

---

### Tool 3 — ALM NII Forecasting & Balance Sheet Stress
A structurally correct asset-liability model anchored to PNC's publicly reported 
balance sheet from the 1Q26 Earnings Release (March 31, 2026). Models net interest 
income across four rate scenarios and generates a repricing gap table and CFO-level 
management commentary.

**Balance sheet anchor (PNC, March 31, 2026):**

| Category | Balance |
|----------|---------|
| Commercial & Industrial Loans | $190.0B |
| Commercial Real Estate Loans | $34.8B |
| Consumer Loans | $105.0B |
| Investment Securities | $143.1B |
| Interest-Bearing Deposits | $356.9B |
| Borrowed Funds | $66.7B |
| Reported NIM | 2.95% |
| Quarterly NII | $3,961MM |

**Repricing gap:**

    Gap = Rate-Sensitive Assets - Rate-Sensitive Liabilities

A positive gap indicates asset-sensitivity — the bank benefits from rising rates 
because assets reprice faster than liabilities.

**NII stress model:**

    NII Change = (Asset repricing income) - (Liability repricing cost)
    Deposit cost change = IB Deposits * Repricing % * Rate Shift * Deposit Beta

Deposit beta of 40% is applied to interest-bearing deposits, consistent with 
PNC's observed NIM expansion during the 2022-2023 Fed tightening cycle.

**Rate scenarios:** Base Case (flat), +100bps Parallel, +200bps Parallel, 
Curve Flattener (short +150bps / long +50bps)

**Live data:** Fed Funds rate, 2Y Treasury, and 10Y Treasury from FRED used to 
set the base rate environment.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Web framework | Streamlit |
| Data source | FRED API via `fredapi` |
| Charting | Plotly |
| Numerics | NumPy, SciPy |
| Data manipulation | Pandas |

---

## Project Structure

fixed-income-suite/
├── app.py                  # Landing page and navigation
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # App theme configuration
└── pages/
├── 1_Yield_Curve.py    # Tool 1 — Yield curve and bond risk
├── 2_MBS_Model.py      # Tool 2 — MBS prepayment model
└── 3_ALM_NII.py        # Tool 3 — ALM NII forecasting


---

## Setup Instructions (Run Locally)

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/fixed-income-suite.git
cd fixed-income-suite
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your FRED API key**

Create a file at `.streamlit/secrets.toml` with the following content:
```toml
FRED_API_KEY = "your_fred_api_key_here"
```
Get a free API key at https://fred.stlouisfed.org/

**4. Run the app**
```bash
streamlit run app.py
```

---

## Data Sources

- **US Treasury yields:** FRED tickers DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30
- **30Y mortgage rate:** FRED ticker MORTGAGE30US
- **Policy and benchmark rates:** FRED tickers FEDFUNDS, DGS2, DGS10
- **PNC balance sheet:** PNC Financial Services Group 1Q26 Earnings Release, April 15, 2026

---

## Important Notes

- Repricing assumptions in the ALM tool are analyst estimates derived from PNC earnings 
  commentary and standard industry conventions — they are clearly labeled as such in the app.
- This tool is built for educational and internship demonstration purposes.
- No proprietary or non-public PNC data is used anywhere in this application.

---

*Built by [Your Name] | PNC CIO Internship | [your.email@example.com]*
