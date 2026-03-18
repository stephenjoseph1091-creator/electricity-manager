"""
Texas Electricity Plan Manager
Streamlit web app for comparing plans, tracking usage, and managing enrollments.
"""

import io
import math
from datetime import date, datetime, timedelta

import anthropic
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TX Electricity Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state initialisation helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Seed session_state keys that may be referenced before they are set."""
    defaults = {
        "usage_df": None,          # parsed CSV DataFrame
        "plans_df": None,          # fetched PowerToChoose plans
        "selected_plan": None,     # plan chosen in Compare tab
        "chat_history": [],        # Enroll tab chat messages
        "anthropic_api_key": "",   # optional key from sidebar
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "provider": "Green Mountain Energy",
    "plan_name": "Pollution Free 18",
    "zip_code": "75063",
    "contract_start": date(2024, 12, 8),
    "contract_term": 18,
    "etf": 200.0,
    "base_charge": 5.00,
    "energy_charge_cents": 8.7991,   # ¢/kWh — displayed to user
    "tdu_fixed": 4.23,
    "tdu_variable_cents": 5.2974,    # ¢/kWh — displayed to user
}

PTC_URL = "http://api.powertochoose.org/api/PowerToChoose/plans?zip_code={zip_code}"

# ---------------------------------------------------------------------------
# Utility: get the Anthropic API key (secrets first, then session state)
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Return API key from st.secrets if available, otherwise session state."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    key = st.session_state.get("anthropic_api_key", "").strip()
    return key if key else None


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_smt_csv(uploaded_file) -> pd.DataFrame | None:
    """
    Parse a Smart Meter Texas (SMT) monthly usage CSV.

    SMT files have several header/metadata rows before the actual data.
    We try skipping 0–9 rows to find a header row that contains a date
    column and a kWh/usage column.  Values may be prefixed with a leading
    apostrophe (SMT artefact) which we strip.

    Returns a DataFrame with columns:
        date  – datetime64
        kwh   – float
    or None on failure.
    """
    content = uploaded_file.read()
    uploaded_file.seek(0)  # allow re-reads if needed

    for skip in range(10):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                skiprows=skip,
                dtype=str,  # read everything as str first so we can strip apostrophes
            )
            df.columns = df.columns.str.strip()

            # Identify the date column (contains "DATE" but NOT "END DATE")
            date_col = next(
                (
                    c for c in df.columns
                    if "DATE" in c.upper() and "END" not in c.upper()
                ),
                None,
            )
            # Identify the kWh column (contains "KWH" or "USAGE")
            kwh_col = next(
                (
                    c for c in df.columns
                    if "KWH" in c.upper() or "USAGE" in c.upper()
                ),
                None,
            )

            if date_col is None or kwh_col is None:
                continue  # try next skiprows value

            # Strip leading apostrophe that SMT prepends to some values
            df[date_col] = df[date_col].str.lstrip("'").str.strip()
            df[kwh_col] = df[kwh_col].str.lstrip("'").str.strip()

            # Drop rows where either column is empty / non-parseable
            df = df[[date_col, kwh_col]].dropna()
            df = df[df[date_col] != ""]
            df = df[df[kwh_col] != ""]

            # Parse dates and kWh values
            df["date"] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
            df["kwh"] = pd.to_numeric(df[kwh_col], errors="coerce")
            df = df.dropna(subset=["date", "kwh"])

            if df.empty:
                continue

            df = df[["date", "kwh"]].sort_values("date").reset_index(drop=True)
            return df

        except Exception:
            continue  # silently try next skiprows value

    return None  # all attempts failed


# ---------------------------------------------------------------------------
# PowerToChoose API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_plans(zip_code: str) -> pd.DataFrame | None:
    """
    Fetch electricity plans from PowerToChoose API for the given ZIP code.

    Returns a cleaned DataFrame, or None if the request fails.
    """
    url = PTC_URL.format(zip_code=zip_code)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TXElectricityManager/1.0)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        st.error(f"PowerToChoose API error: {exc}")
        return None

    plans = payload.get("data", [])
    if not plans:
        st.warning("API returned no plans for that ZIP code.")
        return None

    df = pd.json_normalize(plans)
    return df


def _get_price_at(row: pd.Series, kwh: float) -> float:
    """
    Interpolate (or extrapolate) a plan's rate at a given monthly kWh usage.

    PowerToChoose provides price500, price1000, price2000 columns (in ¢/kWh).
    We do piecewise linear interpolation and linear extrapolation beyond 2000.
    Returns rate in $/kWh.
    """
    try:
        p500 = float(row.get("price_kwh500", 0) or 0)
        p1000 = float(row.get("price_kwh1000", 0) or 0)
        p2000 = float(row.get("price_kwh2000", 0) or 0)
    except (TypeError, ValueError):
        return 0.0

    # Use the midpoint price columns (cents/kWh) — convert to $/kWh at return
    if kwh <= 500:
        rate_cents = p500
    elif kwh <= 1000:
        frac = (kwh - 500) / 500
        rate_cents = p500 + frac * (p1000 - p500)
    elif kwh <= 2000:
        frac = (kwh - 1000) / 1000
        rate_cents = p1000 + frac * (p2000 - p1000)
    else:
        # Linear extrapolation beyond 2000 kWh using the 1000→2000 slope
        slope = (p2000 - p1000) / 1000
        rate_cents = p2000 + slope * (kwh - 2000)

    return max(rate_cents, 0.0) / 100.0  # convert ¢ → $


def filter_plans(df: pd.DataFrame, allow_short_term: bool, allow_bill_credit: bool) -> pd.DataFrame:
    """
    Remove plans that are prepaid, time-of-use, non-fixed-rate, or have
    missing/zero price points.  Optionally filter out short-term (<= 3 month)
    and bill-credit plans.
    """
    # Normalise column names to lowercase for safe access
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    def _flag(col: str) -> pd.Series:
        """Return boolean Series for a column, defaulting to False if absent."""
        if col in df.columns:
            return df[col].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        return pd.Series([False] * len(df), index=df.index)

    # Drop definitely incompatible plan types
    df = df[~_flag("prepaid")]
    df = df[~_flag("timeofuse")]

    # Keep only fixed-rate plans (rate_type == "Fixed" where available)
    if "rate_type" in df.columns:
        is_fixed = df["rate_type"].astype(str).str.lower().str.contains("fix", na=False)
        df = df[is_fixed]

    # Drop plans with missing or zero price points
    for col in ["price_kwh500", "price_kwh1000", "price_kwh2000"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df[col].notna() & (df[col] > 0)]

    # Optional user-controlled filters
    if not allow_short_term:
        if "term_value" in df.columns:
            df["term_value"] = pd.to_numeric(df["term_value"], errors="coerce").fillna(0)
            df = df[df["term_value"] > 3]

    if not allow_bill_credit:
        df = df[~_flag("minimum_usage")]  # minimum_usage=true signals bill-credit plans

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cost calculations
# ---------------------------------------------------------------------------

def current_plan_cost(kwh: float, base: float, energy_rate: float,
                       tdu_fixed: float, tdu_rate: float) -> float:
    """
    Monthly cost on the current plan.
    Rates must already be in $/kWh (not ¢/kWh).
    """
    return base + tdu_fixed + kwh * (energy_rate + tdu_rate)


def plan_cost_at_kwh(row: pd.Series, kwh: float) -> float:
    """
    Estimate total monthly bill for a PTC plan at a given kWh level.

    The PTC price points are all-in ¢/kWh (they include TDU pass-throughs),
    so we only need: price_per_kwh * kwh  (no separate fixed charges needed
    for comparison purposes — PTC prices are total effective rates).
    """
    return _get_price_at(row, kwh) * kwh


def score_plans(
    plans_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    base: float,
    energy_rate: float,
    tdu_fixed: float,
    tdu_rate: float,
    etf: float,
    contract_end: date,
) -> pd.DataFrame:
    """
    Add scoring columns to plans_df:

    historical_savings  – current plan total cost over history minus plan total
    post_contract_savings – 12-month projection from contract_end using seasonal avgs
    net_now             – switch-now net benefit: (remaining months savings) - ETF
    avg_rate            – average effective rate across historical months
    """
    if usage_df is None or usage_df.empty:
        return plans_df

    today = date.today()

    # ------------------------------------------------------------------
    # Seasonal (calendar-month) averages from history
    # ------------------------------------------------------------------
    usage_df = usage_df.copy()
    usage_df["month_num"] = usage_df["date"].dt.month
    seasonal = usage_df.groupby("month_num")["kwh"].mean()

    # ------------------------------------------------------------------
    # Historical cost comparison
    # ------------------------------------------------------------------
    total_current = sum(
        current_plan_cost(row["kwh"], base, energy_rate, tdu_fixed, tdu_rate)
        for _, row in usage_df.iterrows()
    )

    # ------------------------------------------------------------------
    # For each plan: compute historical total, post-contract savings, net_now
    # ------------------------------------------------------------------
    historical_savings = []
    post_contract_savings = []
    net_now_list = []
    avg_rate_list = []

    for _, plan in plans_df.iterrows():
        # --- historical total ---
        total_plan_hist = sum(
            plan_cost_at_kwh(plan, row["kwh"]) for _, row in usage_df.iterrows()
        )
        historical_savings.append(total_current - total_plan_hist)

        # --- 12-month post-contract projection ---
        proj_savings = 0.0
        proj_month = contract_end
        for i in range(12):
            m = (proj_month.month - 1 + i) % 12 + 1
            avg_kwh = seasonal.get(m, usage_df["kwh"].mean())
            curr_cost = current_plan_cost(avg_kwh, base, energy_rate, tdu_fixed, tdu_rate)
            plan_cost = plan_cost_at_kwh(plan, avg_kwh)
            proj_savings += curr_cost - plan_cost
        post_contract_savings.append(proj_savings)

        # --- switch-now net (savings over remaining months minus ETF) ---
        if today >= contract_end:
            months_rem = 0
        else:
            delta = relativedelta(contract_end, today)
            months_rem = delta.months + delta.years * 12

        net_sw = 0.0
        for i in range(months_rem):
            m = (today.month - 1 + i) % 12 + 1
            avg_kwh = seasonal.get(m, usage_df["kwh"].mean())
            curr_cost = current_plan_cost(avg_kwh, base, energy_rate, tdu_fixed, tdu_rate)
            plan_cost = plan_cost_at_kwh(plan, avg_kwh)
            net_sw += curr_cost - plan_cost
        net_now_list.append(net_sw - etf)

        # --- average effective rate across history ---
        avg_kwh_total = usage_df["kwh"].mean() if not usage_df.empty else 1000.0
        avg_rate_list.append(_get_price_at(plan, avg_kwh_total) * 100)  # back to ¢/kWh

    plans_df = plans_df.copy()
    plans_df["historical_savings"] = historical_savings
    plans_df["post_contract_savings"] = post_contract_savings
    plans_df["net_now"] = net_now_list
    plans_df["avg_rate_cents"] = avg_rate_list

    return plans_df


# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(
    plan_name: str,
    provider: str,
    usage_df: pd.DataFrame | None,
    best_plan: pd.Series | None,
    decision: str,
) -> str:
    """Build a rich system prompt with the user's full context."""
    usage_summary = "No usage data loaded."
    if usage_df is not None and not usage_df.empty:
        avg_kwh = usage_df["kwh"].mean()
        total_kwh = usage_df["kwh"].sum()
        months = len(usage_df)
        usage_summary = (
            f"{months} months of history, avg {avg_kwh:.0f} kWh/month, "
            f"total {total_kwh:.0f} kWh"
        )

    best_plan_summary = "No competitor plans available."
    if best_plan is not None:
        name = best_plan.get("plan_name", best_plan.get("company_name", "Unknown"))
        rate = best_plan.get("avg_rate_cents", 0)
        hist_sav = best_plan.get("historical_savings", 0)
        post_sav = best_plan.get("post_contract_savings", 0)
        net_now = best_plan.get("net_now", 0)
        best_plan_summary = (
            f"Best competitor: {name}, avg rate {rate:.4f} ¢/kWh, "
            f"historical savings ${hist_sav:.2f}, "
            f"post-contract 12-mo savings ${post_sav:.2f}, "
            f"switch-now net (after ETF) ${net_now:.2f}"
        )

    return (
        "You are a helpful Texas electricity plan advisor. "
        "You have access to the user's electricity account details.\n\n"
        f"Current plan: {plan_name} ({provider})\n"
        f"Usage: {usage_summary}\n"
        f"{best_plan_summary}\n"
        f"Recommendation: {decision}\n\n"
        "Answer questions concisely and accurately. "
        "When discussing rates, always clarify whether they are in ¢/kWh or $/kWh. "
        "If you are unsure about something, say so rather than guessing."
    )


def ai_explain(prompt: str, system_prompt: str, api_key: str) -> str:
    """Call Claude for a single-turn explanation."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "Rate limit reached. Please wait a moment and try again."
    except Exception as exc:
        return f"AI request failed: {exc}"


def ai_chat(messages: list[dict], system_prompt: str, api_key: str) -> str:
    """Call Claude for multi-turn chat. `messages` is a list of {role, content} dicts."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=800,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "Invalid API key. Please check your Anthropic API key in the sidebar."
    except anthropic.RateLimitError:
        return "Rate limit reached. Please wait a moment and try again."
    except Exception as exc:
        return f"AI request failed: {exc}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """
    Render the sidebar and return a dict of current plan settings.
    All rates are stored in $/kWh internally.
    """
    st.sidebar.header("⚡ Your Current Plan")

    uploaded_file = st.sidebar.file_uploader(
        "Upload SMT monthly CSV",
        type=["csv"],
        help="Download from SmartMeterTexas.com → My Account → Usage → Monthly",
    )

    if uploaded_file is not None:
        with st.spinner("Parsing CSV…"):
            df = parse_smt_csv(uploaded_file)
        if df is not None:
            st.session_state["usage_df"] = df
            st.sidebar.success(f"Loaded {len(df)} months of usage data.")
        else:
            st.sidebar.error(
                "Could not parse this CSV. Make sure it is a Smart Meter Texas "
                "monthly export and try again."
            )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Plan Details")

    provider = st.sidebar.text_input("Provider", value=DEFAULTS["provider"])
    plan_name = st.sidebar.text_input("Plan Name", value=DEFAULTS["plan_name"])
    zip_code = st.sidebar.text_input("ZIP Code", value=DEFAULTS["zip_code"])

    contract_start = st.sidebar.date_input(
        "Contract Start Date", value=DEFAULTS["contract_start"]
    )
    contract_term = st.sidebar.number_input(
        "Contract Term (months)", min_value=1, max_value=60,
        value=DEFAULTS["contract_term"], step=1,
    )
    etf = st.sidebar.number_input(
        "Early Termination Fee ($)", min_value=0.0,
        value=DEFAULTS["etf"], step=10.0, format="%.2f",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Rate Details")

    base_charge = st.sidebar.number_input(
        "Base / Customer Charge ($/mo)", min_value=0.0,
        value=DEFAULTS["base_charge"], step=0.01, format="%.2f",
    )
    energy_charge_cents = st.sidebar.number_input(
        "Energy Charge (¢/kWh)", min_value=0.0,
        value=DEFAULTS["energy_charge_cents"], step=0.0001, format="%.4f",
    )
    tdu_fixed = st.sidebar.number_input(
        "TDU Fixed Charge ($/mo)", min_value=0.0,
        value=DEFAULTS["tdu_fixed"], step=0.01, format="%.2f",
    )
    tdu_variable_cents = st.sidebar.number_input(
        "TDU Variable Charge (¢/kWh)", min_value=0.0,
        value=DEFAULTS["tdu_variable_cents"], step=0.0001, format="%.4f",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Settings (optional)")
    api_key_input = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.get("anthropic_api_key", ""),
        help="Required for AI explanations. You can also set ANTHROPIC_API_KEY in .streamlit/secrets.toml",
    )
    st.session_state["anthropic_api_key"] = api_key_input

    # Derived values
    contract_end = contract_start + relativedelta(months=int(contract_term))
    today = date.today()
    months_remaining = max(
        0,
        relativedelta(contract_end, today).months + relativedelta(contract_end, today).years * 12,
    )

    return {
        "provider": provider,
        "plan_name": plan_name,
        "zip_code": zip_code,
        "contract_start": contract_start,
        "contract_term": int(contract_term),
        "contract_end": contract_end,
        "months_remaining": months_remaining,
        "etf": etf,
        "base_charge": base_charge,
        # Store rates in $/kWh internally
        "energy_rate": energy_charge_cents / 100.0,
        "tdu_fixed": tdu_fixed,
        "tdu_rate": tdu_variable_cents / 100.0,
        # Keep ¢/kWh versions for display
        "energy_charge_cents": energy_charge_cents,
        "tdu_variable_cents": tdu_variable_cents,
    }


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(cfg: dict) -> None:
    """Render the Dashboard tab."""
    st.header("Dashboard")

    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")

    if usage_df is None:
        # Welcome / no-data state
        st.info("Upload your Smart Meter Texas CSV in the sidebar to get started.")
        st.markdown(
            """
            ### Getting started — 3 steps

            **Step 1 — Download your usage data**
            Log in at [SmartMeterTexas.com](https://www.smartmetertexas.com) →
            *My Account* → *Usage* → *Monthly* → Export CSV.

            **Step 2 — Fill in your plan details**
            Use the sidebar to enter your current provider, plan name, contract dates, and rate details.
            The form is pre-filled with common Green Mountain Energy values — adjust as needed.

            **Step 3 — Explore the tabs**
            - **Compare Plans** — fetch live plans from PowerToChoose and see how you stack up
            - **Decision** — get a STAY / SWITCH recommendation based on your actual usage
            - **Enroll** — step-by-step guidance and an AI chat assistant
            """
        )
        return

    # --- Key metrics ---
    avg_kwh = usage_df["kwh"].mean()
    avg_bill = avg_kwh * (cfg["energy_rate"] + cfg["tdu_rate"]) + cfg["base_charge"] + cfg["tdu_fixed"]
    est_annual = avg_bill * 12

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Months of Data", len(usage_df))
    col2.metric("Avg Monthly Usage", f"{avg_kwh:.0f} kWh")
    col3.metric("Avg Est. Monthly Bill", f"${avg_bill:.2f}")
    col4.metric("Est. Annual Cost", f"${est_annual:.2f}")

    st.markdown("---")

    # --- Usage bar chart ---
    st.subheader("Monthly Usage (kWh)")
    chart_df = usage_df.copy()
    chart_df["month"] = chart_df["date"].dt.strftime("%b %Y")

    fig = px.bar(
        chart_df,
        x="month",
        y="kwh",
        labels={"month": "Month", "kwh": "Usage (kWh)"},
        color="kwh",
        color_continuous_scale=["#c8f5d8", "#00A651"],
        text_auto=".0f",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45),
        margin=dict(t=30, b=60),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Monthly bill breakdown ---
    st.subheader("Estimated Monthly Bills")
    bill_df = usage_df.copy()
    bill_df["est_bill"] = bill_df["kwh"].apply(
        lambda k: current_plan_cost(k, cfg["base_charge"], cfg["energy_rate"],
                                     cfg["tdu_fixed"], cfg["tdu_rate"])
    )
    bill_df["month"] = bill_df["date"].dt.strftime("%b %Y")

    fig2 = px.line(
        bill_df,
        x="month",
        y="est_bill",
        markers=True,
        labels={"month": "Month", "est_bill": "Est. Bill ($)"},
        line_shape="spline",
    )
    fig2.update_traces(line_color="#00A651", marker_color="#00A651")
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45),
        margin=dict(t=30, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Compare Plans
# ---------------------------------------------------------------------------

def render_compare(cfg: dict) -> None:
    """Render the Compare Plans tab."""
    st.header("Compare Plans")

    col_fetch, col_filters = st.columns([2, 3])

    with col_fetch:
        if st.button("🔄 Fetch Plans from PowerToChoose", type="primary"):
            with st.spinner("Fetching plans…"):
                raw = fetch_plans(cfg["zip_code"])
            if raw is not None:
                st.session_state["plans_df"] = raw
                st.success(f"Fetched {len(raw)} plans.")
            else:
                st.error("Failed to fetch plans. Check ZIP code and network connection.")

    with col_filters:
        allow_short = st.checkbox("Include short-term plans (≤ 3 months)", value=False)
        allow_credit = st.checkbox("Include bill-credit plans", value=False)

    plans_df: pd.DataFrame | None = st.session_state.get("plans_df")
    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")

    if plans_df is None:
        st.info("Click **Fetch Plans** to load available electricity plans for ZIP " + cfg["zip_code"])
        return

    # Filter plans
    filtered = filter_plans(plans_df, allow_short, allow_credit)

    if filtered.empty:
        st.warning("No plans matched the current filters. Try relaxing the filter options.")
        return

    # Score plans if usage data is available
    if usage_df is not None:
        filtered = score_plans(
            filtered,
            usage_df,
            cfg["base_charge"],
            cfg["energy_rate"],
            cfg["tdu_fixed"],
            cfg["tdu_rate"],
            cfg["etf"],
            cfg["contract_end"],
        )
        sort_options = {
            "Post-Contract Savings (12-mo)": "post_contract_savings",
            "Historical Savings": "historical_savings",
            "Switch-Now Net Benefit": "net_now",
            "Avg Rate (¢/kWh)": "avg_rate_cents",
        }
    else:
        st.info("Upload usage data (sidebar) to see savings estimates.")
        sort_options = {"Price @ 1000 kWh (¢)": "price_kwh1000"}

    sort_label = st.selectbox("Sort by", list(sort_options.keys()))
    sort_col = sort_options[sort_label]

    if sort_col in filtered.columns:
        ascending = sort_col == "avg_rate_cents"
        filtered = filtered.sort_values(sort_col, ascending=ascending)

    # Build display DataFrame
    display_cols_map = {
        "company_name": "Provider",
        "plan_name": "Plan Name",
        "term_value": "Term (mo)",
        "price_kwh500": "¢/kWh @ 500",
        "price_kwh1000": "¢/kWh @ 1000",
        "price_kwh2000": "¢/kWh @ 2000",
    }
    if usage_df is not None:
        display_cols_map.update({
            "avg_rate_cents": "Avg Rate ¢/kWh",
            "historical_savings": "Hist. Savings $",
            "post_contract_savings": "12-mo Savings $",
            "net_now": "Switch-Now Net $",
        })

    # Only keep columns that exist
    available = {k: v for k, v in display_cols_map.items() if k in filtered.columns}
    display_df = filtered[list(available.keys())].copy()
    display_df = display_df.rename(columns=available)

    # Round numeric columns for readability
    for col in display_df.select_dtypes(include="number").columns:
        display_df[col] = display_df[col].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    st.markdown(f"*Showing {len(filtered)} plans after filtering.*")

    st.markdown("---")

    # Top plan cards with Select buttons
    st.subheader("Top Plans — Select One to Enroll")
    st.caption("Click **Select this plan** on any row to load it into the Enroll tab.")

    top_n = min(10, len(filtered))
    for i, (_, row) in enumerate(filtered.head(top_n).iterrows()):
        name = str(row.get("plan_name", "Unknown Plan"))
        provider = str(row.get("company_name", "Unknown Provider"))
        term = row.get("term_value", "?")
        r1000 = row.get("price_kwh1000", 0)
        post_sav = row.get("post_contract_savings", None)
        net_now = row.get("net_now", None)

        col_info, col_btn = st.columns([5, 1])
        with col_info:
            sav_str = f"  |  12-mo savings: **${post_sav:,.0f}**" if post_sav is not None else ""
            net_str = f"  |  Switch-now net: **${net_now:,.0f}**" if net_now is not None else ""
            st.markdown(
                f"**{i+1}. {provider} — {name}**  \n"
                f"{term} months  |  {float(r1000):.1f}¢/kWh @ 1000 kWh{sav_str}{net_str}"
            )
        with col_btn:
            if st.button("Select →", key=f"select_plan_{i}"):
                st.session_state["selected_plan"] = row.to_dict()
                st.success(f"✓ Selected **{provider} — {name}**. Switch to the **Enroll** tab.")

        st.divider()


# ---------------------------------------------------------------------------
# Tab 3: Decision
# ---------------------------------------------------------------------------

def render_decision(cfg: dict) -> None:
    """Render the Decision tab."""
    st.header("Decision")

    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")
    plans_df: pd.DataFrame | None = st.session_state.get("plans_df")

    if usage_df is None:
        st.warning("Upload usage data in the sidebar to see a personalised recommendation.")
        return

    if plans_df is None:
        st.warning("Go to **Compare Plans** and fetch plans first.")
        return

    # Score and identify best plan
    filtered = filter_plans(plans_df, allow_short_term=False, allow_bill_credit=False)

    if filtered.empty:
        st.warning("No qualifying fixed-rate plans found to compare against.")
        return

    scored = score_plans(
        filtered,
        usage_df,
        cfg["base_charge"],
        cfg["energy_rate"],
        cfg["tdu_fixed"],
        cfg["tdu_rate"],
        cfg["etf"],
        cfg["contract_end"],
    )

    # Best plan = highest post-contract savings
    scored = scored.sort_values("post_contract_savings", ascending=False)
    best = scored.iloc[0]
    second = scored.iloc[1] if len(scored) > 1 else None

    months_rem = cfg["months_remaining"]
    contract_end = cfg["contract_end"]

    # --- Status banner ---
    if months_rem == 0:
        status = "🔴 SWITCH NOW — Contract already ended"
        status_color = "#ff4b4b"
    elif best["net_now"] > 0:
        status = "🟡 SWITCH NOW — ETF is worth paying"
        status_color = "#ffa500"
    else:
        status = f"🟢 STAY — Switch on {contract_end.strftime('%B %d, %Y')}"
        status_color = "#00A651"

    st.markdown(
        f"""
        <div style="background:{status_color}22;border-left:6px solid {status_color};
        padding:16px 20px;border-radius:4px;margin-bottom:16px;">
        <h2 style="margin:0;color:{status_color};">{status}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Progress bar to contract end ---
    total_days = (cfg["contract_end"] - cfg["contract_start"]).days
    elapsed_days = (date.today() - cfg["contract_start"]).days
    progress = max(0.0, min(1.0, elapsed_days / total_days if total_days > 0 else 1.0))
    days_left = max(0, (cfg["contract_end"] - date.today()).days)

    st.markdown("**Contract Progress**")
    st.progress(progress)
    st.caption(
        f"{cfg['contract_start'].strftime('%b %d, %Y')} → "
        f"{cfg['contract_end'].strftime('%b %d, %Y')} "
        f"({'expired' if months_rem == 0 else f'{days_left} days remaining'})"
    )

    st.markdown("---")

    # --- Historical billing comparison chart ---
    st.subheader("What Would You Have Paid? — Current Plan vs Top Alternative")

    best_label = f"{best.get('company_name', '')} — {best.get('plan_name', '')}"
    current_label = f"{cfg['provider']} — {cfg['plan_name']} (current)"

    chart_rows = []
    for _, urow in usage_df.iterrows():
        kwh = urow["kwh"]
        dt = urow["date"]
        curr_cost = current_plan_cost(kwh, cfg["base_charge"], cfg["energy_rate"],
                                      cfg["tdu_fixed"], cfg["tdu_rate"])
        best_cost = plan_cost_at_kwh(best, kwh)
        chart_rows.append({
            "Period": dt.strftime("%b %Y"),
            current_label: round(curr_cost, 2),
            best_label: round(best_cost, 2),
        })

    chart_df = pd.DataFrame(chart_rows)
    chart_melted = chart_df.melt(id_vars="Period", var_name="Plan", value_name="Est. Bill ($)")

    fig_compare = px.line(
        chart_melted,
        x="Period",
        y="Est. Bill ($)",
        color="Plan",
        markers=True,
        color_discrete_map={
            current_label: "#d9534f",
            best_label: "#00A651",
        },
        line_shape="spline",
    )
    fig_compare.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=60),
    )
    fig_compare.update_traces(line_width=2.5, marker_size=7)
    st.plotly_chart(fig_compare, use_container_width=True)

    total_current = chart_df[current_label].sum()
    total_best = chart_df[best_label].sum()
    diff = total_current - total_best
    st.caption(
        f"Over your {len(usage_df)} billing periods — current plan total: **${total_current:,.2f}** "
        f"vs {best_label}: **${total_best:,.2f}** — difference: **${diff:,.2f}**"
    )

    st.markdown("---")

    # --- Top plans table ---
    st.subheader("Top Plan Recommendations")

    top_plans = scored.head(10).copy()
    table_data = []
    for i, (_, p) in enumerate(top_plans.iterrows()):
        table_data.append({
            "#": i + 1,
            "Provider": p.get("company_name", ""),
            "Plan": p.get("plan_name", ""),
            "Term": f"{int(p.get('term_value', 0))} mo",
            "¢/kWh @ 500 kWh": f"{float(p.get('price_kwh500', 0)):.1f}",
            "¢/kWh @ 1000 kWh": f"{float(p.get('price_kwh1000', 0)):.1f}",
            "¢/kWh @ 2000 kWh": f"{float(p.get('price_kwh2000', 0)):.1f}",
            "12-mo Savings ($)": f"${p.get('post_contract_savings', 0):,.0f}",
            "Switch-Now Net ($)": f"${p.get('net_now', 0):,.0f}",
        })

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "**12-mo Savings** = estimated savings vs your current plan in the 12 months after your contract ends.  "
        "**Switch-Now Net** = savings over remaining contract months minus the ETF — positive means it's worth paying the ETF today."
    )

    st.markdown("---")

    # --- Select a plan to enroll ---
    st.subheader("Select a Plan to Enroll In")
    plan_labels = [
        f"{p.get('company_name', '')} — {p.get('plan_name', '')}"
        for _, p in top_plans.iterrows()
    ]
    for i, (label, (_, p)) in enumerate(zip(plan_labels, top_plans.iterrows())):
        col_label, col_efl, col_enroll, col_select = st.columns([4, 1, 1, 1])
        col_label.markdown(
            f"**{i+1}. {label}**  \n"
            f"{int(p.get('term_value', 0))} months · "
            f"{float(p.get('price_kwh1000', 0)):.1f}¢/kWh @ 1000 · "
            f"12-mo savings: ${p.get('post_contract_savings', 0):,.0f}"
        )
        if p.get("fact_sheet"):
            col_efl.link_button("EFL", p["fact_sheet"])
        if p.get("go_to_plan"):
            col_enroll.link_button("Enroll →", p["go_to_plan"])
        if col_select.button("Select ✓", key=f"dec_select_{i}", type="primary"):
            st.session_state["selected_plan"] = p.to_dict()
            st.success(f"Selected **{label}** — go to the **Enroll** tab to proceed.")

    st.markdown("---")

    # --- AI explanation ---
    api_key = _get_api_key()

    if api_key:
        if st.button("🤖 Explain this recommendation", key="explain_btn"):
            sys_prompt = _build_system_prompt(
                cfg["plan_name"], cfg["provider"], usage_df, best, status
            )
            user_prompt = (
                f"Explain why the recommendation is '{status}' and summarise the "
                f"top plan option in plain English. Include specific dollar amounts."
            )
            with st.spinner("Asking Claude…"):
                explanation = ai_explain(user_prompt, sys_prompt, api_key)
            st.markdown("**AI Explanation**")
            st.info(explanation)
    else:
        st.caption(
            "Add an Anthropic API key in the sidebar to enable AI explanations."
        )


# ---------------------------------------------------------------------------
# Tab 4: Enroll
# ---------------------------------------------------------------------------

def render_enroll(cfg: dict) -> None:
    """Render the Enroll tab."""
    st.header("Enroll in a New Plan")

    selected: dict | None = st.session_state.get("selected_plan")

    if selected is None:
        st.info("Go to **Compare Plans** and select a plan first.")
        return

    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")

    # --- Selected plan card ---
    p_name = selected.get("plan_name", "Unknown Plan")
    p_provider = selected.get("company_name", "Unknown Provider")
    p_term = selected.get("term_value", "?")
    p1000 = selected.get("price_kwh1000", 0)
    post_sav = selected.get("post_contract_savings", "N/A")
    net_now = selected.get("net_now", "N/A")
    efl_url = selected.get("fact_sheet", selected.get("efl_url", "#"))
    enroll_url = selected.get("enroll_url", selected.get("go_to_plan", "#"))

    st.markdown(
        f"""
        <div style="background:#f0fdf4;border:2px solid #00A651;border-radius:8px;padding:20px;margin-bottom:16px;">
        <h2 style="margin:0 0 4px;">{p_provider}</h2>
        <h3 style="margin:0 0 12px;color:#444;">{p_name}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contract Term", f"{p_term} months")
    c2.metric("Rate @ 1000 kWh", f"{float(p1000):.2f} ¢/kWh")
    if isinstance(post_sav, (int, float)):
        c3.metric("Post-Contract 12-mo Savings", f"${post_sav:.2f}")
    else:
        c3.metric("Post-Contract 12-mo Savings", "—")
    if isinstance(net_now, (int, float)):
        c4.metric("Switch-Now Net (after ETF)", f"${net_now:.2f}")
    else:
        c4.metric("Switch-Now Net (after ETF)", "—")

    st.markdown("---")

    # --- EFL and enrollment links ---
    link_col1, link_col2 = st.columns(2)
    with link_col1:
        if efl_url and efl_url != "#":
            st.link_button("📄 View Electricity Facts Label (EFL)", efl_url)
        else:
            st.button("📄 EFL not available", disabled=True)
    with link_col2:
        if enroll_url and enroll_url != "#":
            st.link_button("🔗 Enroll with Provider →", enroll_url, type="primary")
        else:
            st.button("🔗 Enroll link not available", disabled=True)

    st.markdown("---")

    # --- Enrollment checklist ---
    st.subheader("📋 Enrollment Checklist")
    st.markdown(
        """
        Before you call or go online to switch, gather the following:

        - [ ] **ESIID number** — found on your current electricity bill (16-digit number)
        - [ ] **Service address** — confirm the address you want service at
        - [ ] **Requested start date** — must be a future business date; typically 3–5 days out
        - [ ] **Credit card** — for deposit if required (many plans require no deposit)
        - [ ] **Note new contract end date** — once enrolled, mark it in your calendar

        **After enrolling:**
        - [ ] Save the confirmation email
        - [ ] Note that the switch takes 1–2 billing cycles to take effect
        - [ ] Download and save your new Electricity Facts Label (EFL)
        """
    )

    st.markdown("---")

    # --- AI Chat assistant ---
    st.subheader("🤖 AI Enrollment Assistant")

    api_key = _get_api_key()

    if not api_key:
        st.warning(
            "Add an Anthropic API key in the sidebar to enable the AI chat assistant."
        )
        return

    # Build system prompt with full context
    plans_df = st.session_state.get("plans_df")
    best_plan = None
    if plans_df is not None and usage_df is not None:
        try:
            filtered = filter_plans(plans_df, False, False)
            if not filtered.empty:
                scored = score_plans(
                    filtered, usage_df,
                    cfg["base_charge"], cfg["energy_rate"],
                    cfg["tdu_fixed"], cfg["tdu_rate"],
                    cfg["etf"], cfg["contract_end"],
                )
                best_plan = scored.sort_values("post_contract_savings", ascending=False).iloc[0]
        except Exception:
            pass

    decision_str = "Recommendation not yet computed"
    months_rem = cfg["months_remaining"]
    if months_rem == 0:
        decision_str = "SWITCH NOW — Contract already ended"
    elif best_plan is not None and best_plan.get("net_now", 0) > 0:
        decision_str = "SWITCH NOW — ETF is worth paying"
    elif cfg["contract_end"]:
        decision_str = f"STAY — Switch on {cfg['contract_end'].strftime('%B %d, %Y')}"

    sys_prompt = _build_system_prompt(
        cfg["plan_name"], cfg["provider"], usage_df, best_plan, decision_str
    )
    # Add enrollment-specific context
    sys_prompt += (
        f"\n\nThe user is currently looking at enrolling in: {p_name} from {p_provider}. "
        "Help them understand the enrollment process, what to watch out for, and answer "
        "any questions about their Texas electricity plan."
    )

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about enrolling, your plan, or Texas electricity…")
    if user_input:
        # Append user message and display it
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = ai_chat(st.session_state["chat_history"], sys_prompt, api_key)
            st.markdown(reply)

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat history"):
            st.session_state["chat_history"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("⚡ Texas Electricity Plan Manager")
    st.caption("Analyse your usage, compare live plans, and make an informed switch decision.")

    # Render sidebar and collect plan configuration
    cfg = render_sidebar()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "🔍 Compare Plans", "🎯 Decision", "✅ Enroll"]
    )

    with tab1:
        render_dashboard(cfg)

    with tab2:
        render_compare(cfg)

    with tab3:
        render_decision(cfg)

    with tab4:
        render_enroll(cfg)


if __name__ == "__main__":
    main()
