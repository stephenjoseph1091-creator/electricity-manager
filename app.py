"""
Texas Electricity Plan Manager
Streamlit web app for comparing plans, tracking usage, and managing enrollments.
"""

import io
import math
import re
import uuid
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
        "chat_history": [],        # AI chat messages
        "efl_data": {},            # parsed EFL values
        "profile_save_result": None,
        "contract_start_override": None,
        "zip_code": "",            # entered by user
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PTC_URL = "http://api.powertochoose.org/api/PowerToChoose/plans?zip_code={zip_code}"

# ---------------------------------------------------------------------------
# Utility: get the Anthropic API key (secrets first, then session state)
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Return Anthropic API key from st.secrets."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def _get_supabase_client():
    """Return a Supabase client if credentials are configured, else None."""
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


def _save_profile(email: str, efl: dict, zip_code: str) -> tuple[bool, str]:
    """Upsert a user profile into Supabase. Returns (success, message)."""
    client = _get_supabase_client()
    if client is None:
        return False, "Supabase not configured — notifications unavailable."

    contract_start = efl.get("form_contract_start")
    row = {
        "email":                 email.strip().lower(),
        "zip_code":              zip_code.strip(),
        "provider":              efl.get("form_provider", ""),
        "plan_name":             efl.get("form_plan_name", ""),
        "contract_start":        contract_start.isoformat() if contract_start else None,
        "contract_term_months":  int(efl.get("form_contract_term", 12)),
        "etf_dollars":           float(efl.get("form_etf", 0)),
        "base_charge":           float(efl.get("form_base_charge", 0)),
        "energy_rate":           float(efl.get("form_energy_charge_cents", 0)) / 100.0,
        "tdu_fixed":             float(efl.get("form_tdu_fixed", 0)),
        "tdu_rate":              float(efl.get("form_tdu_variable_cents", 0)) / 100.0,
        # Reset alert flags whenever the profile is saved/updated
        "alert_60_sent":         False,
        "alert_30_sent":         False,
        "alert_14_sent":         False,
        "unsubscribe_token":     str(uuid.uuid4()),
    }
    try:
        client.table("user_profiles").upsert(row, on_conflict="email").execute()
        return True, f"Profile saved! You'll receive alerts at **{email}**."
    except Exception as exc:
        return False, f"Could not save profile: {exc}"


def _remove_profile(email: str) -> tuple[bool, str]:
    """Delete a user profile from Supabase. Returns (success, message)."""
    client = _get_supabase_client()
    if client is None:
        return False, "Supabase not configured."
    try:
        client.table("user_profiles").delete().eq("email", email.strip().lower()).execute()
        return True, f"**{email}** removed. You won't receive any more alerts."
    except Exception as exc:
        return False, f"Could not remove profile: {exc}"


APP_URL = "https://electricity-manager.streamlit.app"


def _get_unsubscribe_url(email: str) -> str:
    """Look up the unsubscribe token for an email and return the unsubscribe URL."""
    client = _get_supabase_client()
    if client is None:
        return APP_URL
    try:
        result = client.table("user_profiles").select("unsubscribe_token").eq(
            "email", email.strip().lower()
        ).execute()
        if result.data:
            token = result.data[0].get("unsubscribe_token", "")
            if token:
                return f"{APP_URL}/?unsubscribe={token}"
    except Exception:
        pass
    return APP_URL


def _send_test_email(email: str, cfg: dict) -> tuple[bool, str]:
    """Send a sample alert email so the user can preview the format."""
    try:
        import resend as _resend
        _resend.api_key = st.secrets["RESEND_API_KEY"]
        from_email = st.secrets.get("FROM_EMAIL", "onboarding@resend.dev")
    except Exception:
        return False, "Resend not configured in secrets."

    provider      = cfg.get("provider", "Your Provider")
    plan_name     = cfg.get("plan_name", "your current plan")
    end_date      = cfg.get("contract_end")
    end_str       = end_date.strftime("%B %d, %Y") if end_date else "N/A"
    unsub_url     = _get_unsubscribe_url(email)

    html = f"""
    <div style="font-family:sans-serif;max-width:640px;margin:0 auto;color:#222">
      <div style="background:#00A651;padding:20px 24px;border-radius:8px 8px 0 0">
        <h1 style="margin:0;color:#fff;font-size:22px">⚡ Texas Electricity Plan Monitor</h1>
      </div>
      <div style="background:#f9f9f9;padding:24px;border-radius:0 0 8px 8px;border:1px solid #e0e0e0">
        <p style="font-size:16px"><strong>This is a test email</strong> — your notifications are working correctly.</p>
        <p>When a real alert fires, it will look like this but with live plan data and savings estimates.</p>
        <hr style="border:none;border-top:1px solid #e0e0e0;margin:20px 0">
        <p><strong>Your current plan on file:</strong><br>
        {provider} — {plan_name}<br>
        Contract ends: {end_str}</p>
        <p><strong>You'll be alerted when:</strong></p>
        <ul>
          <li>Your contract ends in 60, 30, or 14 days</li>
          <li>Your contract has expired</li>
          <li>A plan appears that would save you $150+ over 12 months</li>
        </ul>
        <hr style="border:none;border-top:1px solid #e0e0e0;margin:20px 0">
        <p style="font-size:12px;color:#888">
          Don't want these emails? <a href="{unsub_url}" style="color:#00A651">Unsubscribe in one click</a>.
        </p>
      </div>
    </div>
    """
    try:
        _resend.Emails.send({
            "from":    from_email,
            "to":      [email.strip()],
            "subject": "⚡ Test — Texas Electricity Plan Monitor is set up",
            "html":    html,
        })
        return True, f"Test email sent to **{email}**. Check your inbox."
    except Exception as exc:
        return False, f"Could not send email: {exc}"


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
# EFL PDF parsing
# ---------------------------------------------------------------------------

def parse_efl_pdf(uploaded_file) -> dict:
    """
    Parse a Texas Electricity Facts Label PDF and extract rate structure.
    Texas EFLs are standardized regulated documents — fields are always in the same format.
    Returns a dict with session-state form keys populated, plus "_error" key on failure.
    """
    try:
        import pdfplumber
    except ImportError:
        return {"_error": "pdfplumber not installed"}

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        return {"_error": f"Could not read PDF: {e}"}

    if not text.strip():
        return {"_error": "No text found in PDF — it may be a scanned image."}

    result = {}

    # Provider name — "XYZ Company (REP Cert. No. XXXXX)"
    m = re.search(r'^(.+?)\s*\(REP Cert', text, re.MULTILINE)
    if m:
        result["form_provider"] = m.group(1).strip()

    # Plan name — line immediately after the provider line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for i, line in enumerate(lines):
        if '(REP Cert' in line and i + 1 < len(lines):
            plan_line = lines[i + 1]
            plan_line = re.sub(r'\bTM\b|™|®|\(TM\)', '', plan_line).strip()
            if len(plan_line) < 60 and 'service area' not in plan_line.lower():
                result["form_plan_name"] = plan_line
            break

    # Base charge: "Base Charge: $5.00 per billing cycle"
    m = re.search(r'Base Charge[:\s]+\$?([\d.]+)\s+per billing cycle', text, re.IGNORECASE)
    if m:
        result["form_base_charge"] = float(m.group(1))

    # Energy charge: "Energy Charge: 8.7991¢ per kWh"
    m = re.search(r'Energy Charge[:\s]+([\d.]+)\s*.{0,3}\s*per kWh', text, re.IGNORECASE)
    if m:
        result["form_energy_charge_cents"] = float(m.group(1))

    # TDU charges: "$4.23 per billing cycle and 5.2974¢ per kWh"
    m = re.search(
        r'Delivery Charges[:\s]+\$?([\d.]+)\s+per billing cycle and ([\d.]+)\s*.{0,3}\s*per kWh',
        text, re.IGNORECASE,
    )
    if m:
        result["form_tdu_fixed"] = float(m.group(1))
        result["form_tdu_variable_cents"] = float(m.group(2))

    # Contract term: "Contract Term  18 Months"
    m = re.search(r'Contract Term\s+([\d]+)\s+Month', text, re.IGNORECASE)
    if m:
        result["form_contract_term"] = int(m.group(1))

    # Contract start date: "Date:12/08/2024" in the EFL header
    m = re.search(r'\bDate[:\s]+([\d]{1,2}/[\d]{1,2}/[\d]{4})', text, re.IGNORECASE)
    if m:
        try:
            result["form_contract_start"] = datetime.strptime(m.group(1), "%m/%d/%Y").date()
        except ValueError:
            pass

    # ETF — several possible patterns
    m = re.search(r'Yes[.\s]*\$\s*([\d,]+)\s+Applies', text, re.IGNORECASE)
    if not m:
        m = re.search(r'\$([\d,]+)\s+Applies through the end of the contract', text, re.IGNORECASE)
    if not m:
        m = re.search(r'Cancellation Fee[:\s]+\$\s*([\d,]+)', text, re.IGNORECASE)
    if m:
        result["form_etf"] = float(m.group(1).replace(',', ''))

    return result


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
    """Render the sidebar. Returns cfg dict."""
    st.sidebar.header("⚡ Setup")
    st.sidebar.caption("Upload your files and click Analyze — everything else is automatic.")

    with st.sidebar.form("setup_form", clear_on_submit=False):
        uploaded_csv = st.file_uploader(
            "1 · Smart Meter Texas CSV",
            type=["csv"], key="csv_uploader",
            help="Download from SmartMeterTexas.com → My Account → Usage → Monthly → Export CSV",
        )
        uploaded_efl = st.file_uploader(
            "2 · Electricity Facts Label (PDF)",
            type=["pdf"], key="efl_uploader",
            help="The 1–2 page PDF from your provider — in your welcome email or provider website",
        )
        zip_code = st.text_input(
            "3 · Your ZIP code",
            value=st.session_state.get("zip_code", ""),
            placeholder="e.g. 75063",
        )
        notif_email = st.text_input(
            "4 · Your email for alerts",
            value=st.session_state.get("notif_email_stored", ""),
            placeholder="you@example.com",
            help="Get notified 60, 30, and 14 days before your contract ends — and when a better plan appears.",
        )
        contract_start_input = st.date_input(
            "5 · Service start date (optional)",
            value=st.session_state.get("contract_start_override", None),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
            help="If your provider's portal shows a different contract end date, enter your actual service start date here to override the estimate.",
        )
        submitted = st.form_submit_button(
            "Analyze My Plan →",
            type="primary",
            disabled=(uploaded_csv is None and uploaded_efl is None),
        )

    if submitted:
        if uploaded_csv is not None:
            with st.spinner("Reading usage data…"):
                df = parse_smt_csv(uploaded_csv)
            if df is not None:
                st.session_state["usage_df"] = df
            else:
                st.sidebar.error("Could not parse CSV — make sure it's a Smart Meter Texas monthly export.")

        if uploaded_efl is not None:
            with st.spinner("Reading EFL…"):
                efl_result = parse_efl_pdf(uploaded_efl)
            if "_error" in efl_result:
                st.sidebar.error(f"Could not read EFL: {efl_result['_error']}")
            else:
                st.session_state["efl_data"] = efl_result

        st.session_state["zip_code"]              = zip_code
        st.session_state["notif_email_stored"]    = notif_email
        st.session_state["contract_start_override"] = contract_start_input if contract_start_input else None

        if notif_email and st.session_state.get("efl_data"):
            ok, msg = _save_profile(
                notif_email,
                st.session_state["efl_data"],
                zip_code,
            )
            st.session_state["profile_save_result"] = (ok, msg)

        st.rerun()

    # ── Display parsed results (read-only) ────────────────────────────────
    efl      = st.session_state.get("efl_data", {})
    usage_df = st.session_state.get("usage_df")
    notif_email = st.session_state.get("notif_email_stored", "")

    if efl or usage_df is not None:
        st.sidebar.markdown("---")

    if usage_df is not None:
        st.sidebar.success(f"✓ Usage: {len(usage_df)} billing periods loaded")

    if efl:
        provider  = efl.get("form_provider", "")
        plan_name = efl.get("form_plan_name", "")
        term      = int(efl.get("form_contract_term", 0))
        etf_val   = float(efl.get("form_etf", 0))
        energy_c  = float(efl.get("form_energy_charge_cents", 0))
        tdu_var_c = float(efl.get("form_tdu_variable_cents", 0))
        base      = float(efl.get("form_base_charge", 0))
        tdu_fix   = float(efl.get("form_tdu_fixed", 0))
        efl_start = efl.get("form_contract_start")

        override_start  = st.session_state.get("contract_start_override")
        effective_start = override_start if override_start else efl_start
        est_end   = (effective_start + relativedelta(months=term)) if (effective_start and term) else None
        end_label = "Est. contract end" if not override_start else "Contract end"
        end_str   = est_end.strftime("%b %d, %Y") if est_end else "unknown"

        st.sidebar.success("✓ EFL parsed")
        st.sidebar.markdown(
            f"**{provider}**  \n"
            f"{plan_name}  \n"
            f"ETF: **${etf_val:.0f}**  \n"
            f"Rate: **{energy_c + tdu_var_c:.2f}¢/kWh** + **${base + tdu_fix:.2f}/mo** fixed  \n"
            f"{end_label}: **{end_str}**"
        )

    # ── Show profile save result ───────────────────────────────────────────
    save_result = st.session_state.pop("profile_save_result", None)
    if save_result:
        ok, msg = save_result
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(f"Notification setup failed: {msg}")

    # ── Notification actions ───────────────────────────────────────────────
    if efl and notif_email:
        st.sidebar.markdown("---")
        st.sidebar.caption(f"📬 Alerts → **{notif_email}**")
        col_test, col_remove = st.sidebar.columns(2)

        if col_test.button("Send test", help="Preview what an alert email looks like"):
            _cs  = st.session_state.get("contract_start_override") or efl.get("form_contract_start", date.today())
            _ct  = int(efl.get("form_contract_term", 12))
            ok, msg = _send_test_email(notif_email, {
                "provider":     efl.get("form_provider", ""),
                "plan_name":    efl.get("form_plan_name", ""),
                "contract_end": _cs + relativedelta(months=_ct),
            })
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

        if col_remove.button("Remove me", help="Stop all email alerts"):
            ok, msg = _remove_profile(notif_email)
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

    # ── Build cfg from parsed EFL ─────────────────────────────────────────
    efl             = st.session_state.get("efl_data", {})
    zip_code_stored = st.session_state.get("zip_code", "")

    efl_start        = efl.get("form_contract_start", date.today())
    override_start   = st.session_state.get("contract_start_override")
    contract_start   = override_start if override_start else efl_start
    contract_term    = int(efl.get("form_contract_term", 12))
    contract_end     = contract_start + relativedelta(months=contract_term)
    today            = date.today()
    delta            = relativedelta(contract_end, today)
    months_remaining = max(0, delta.months + delta.years * 12)

    base_charge          = float(efl.get("form_base_charge", 0.0))
    energy_charge_cents  = float(efl.get("form_energy_charge_cents", 0.0))
    tdu_fixed            = float(efl.get("form_tdu_fixed", 0.0))
    tdu_variable_cents   = float(efl.get("form_tdu_variable_cents", 0.0))

    return {
        "provider":            efl.get("form_provider", ""),
        "plan_name":           efl.get("form_plan_name", ""),
        "zip_code":            zip_code_stored,
        "contract_start":      contract_start,
        "contract_term":       contract_term,
        "contract_end":        contract_end,
        "months_remaining":    months_remaining,
        "etf":                 float(efl.get("form_etf", 0.0)),
        "base_charge":         base_charge,
        "energy_rate":         energy_charge_cents / 100.0,
        "tdu_fixed":           tdu_fixed,
        "tdu_rate":            tdu_variable_cents / 100.0,
        "energy_charge_cents": energy_charge_cents,
        "tdu_variable_cents":  tdu_variable_cents,
    }


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(cfg: dict) -> None:
    """Render the Dashboard tab."""

    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")

    if usage_df is None:
        st.markdown("## Welcome to your Texas Electricity Manager")
        st.caption("See exactly what you're paying, find better plans, and never let your contract expire unnoticed.")
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1 · Upload your usage CSV")
            st.markdown(
                "Log in at **[SmartMeterTexas.com](https://www.smartmetertexas.com)**  \n"
                "My Account → Usage → Monthly → Export CSV"
            )
        with c2:
            st.markdown("#### 2 · Upload your EFL")
            st.markdown(
                "Your **Electricity Facts Label** PDF — in your welcome email or on your provider's website."
            )
        with c3:
            st.markdown("#### 3 · Enter ZIP + email")
            st.markdown(
                "Your ZIP pulls live plan rates. Your email signs you up for automatic renewal alerts."
            )

        st.markdown("")
        st.info("📱 On mobile? Tap the **arrow in the top-left corner** to open the sidebar and get started.")
        return

    # --- Contract countdown banner ---
    contract_end  = cfg.get("contract_end")
    months_rem    = cfg.get("months_remaining", 0)
    if contract_end:
        days_rem = (contract_end - date.today()).days
        if days_rem < 0:
            st.error(f"⚠️ Your contract expired {abs(days_rem)} days ago. You're on month-to-month — switch now.")
        elif days_rem <= 30:
            st.warning(f"⏰ Your contract ends in **{days_rem} days** ({contract_end.strftime('%B %d, %Y')}). Time to pick a new plan.")
        elif days_rem <= 60:
            st.warning(f"📅 Your contract ends in **{days_rem} days** ({contract_end.strftime('%B %d, %Y')}). Start comparing plans.")
        else:
            st.success(f"✓ Contract runs until **{contract_end.strftime('%B %d, %Y')}** ({months_rem} months remaining)")

    st.markdown("")

    # --- Key metrics (2x2 for mobile friendliness) ---
    avg_kwh  = usage_df["kwh"].mean()
    avg_bill = avg_kwh * (cfg["energy_rate"] + cfg["tdu_rate"]) + cfg["base_charge"] + cfg["tdu_fixed"]

    r1, r2 = st.columns(2)
    r1.metric("Avg Monthly Usage", f"{avg_kwh:.0f} kWh")
    r2.metric("Avg Est. Monthly Bill", f"${avg_bill:.2f}")
    r3, r4 = st.columns(2)
    r3.metric("Est. Annual Cost", f"${avg_bill * 12:.0f}")
    r4.metric("Months of Data", len(usage_df))

    st.markdown("---")

    # --- Monthly usage chart (aggregated to one bar per month) ---
    st.subheader("Monthly Usage")
    chart_df = usage_df.copy()
    chart_df["period"] = pd.to_datetime(chart_df["date"]).dt.to_period("M")
    chart_df = chart_df.groupby("period", as_index=False)["kwh"].sum()
    chart_df = chart_df.sort_values("period")
    chart_df["month"] = chart_df["period"].dt.strftime("%b %Y")

    fig = px.bar(
        chart_df,
        x="month",
        y="kwh",
        labels={"month": "", "kwh": "kWh"},
        color="kwh",
        color_continuous_scale=["#c8f5d8", "#00A651"],
        text_auto=".0f",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        margin=dict(t=20, b=40, l=20, r=20),
        height=320,
    )
    fig.update_traces(textposition="outside", textfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Estimated monthly bill chart ---
    st.subheader("Estimated Monthly Bills")
    bill_df = usage_df.copy()
    bill_df["period"] = pd.to_datetime(bill_df["date"]).dt.to_period("M")
    bill_df = bill_df.groupby("period", as_index=False)["kwh"].sum()
    bill_df = bill_df.sort_values("period")
    bill_df["month"]    = bill_df["period"].dt.strftime("%b %Y")
    bill_df["est_bill"] = bill_df["kwh"].apply(
        lambda k: current_plan_cost(k, cfg["base_charge"], cfg["energy_rate"],
                                    cfg["tdu_fixed"], cfg["tdu_rate"])
    )

    fig2 = px.line(
        bill_df,
        x="month",
        y="est_bill",
        markers=True,
        labels={"month": "", "est_bill": "Est. Bill ($)"},
        line_shape="spline",
    )
    fig2.update_traces(line_color="#00A651", marker_color="#00A651", line_width=2)
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        margin=dict(t=20, b=40, l=20, r=20),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Compare Plans
# ---------------------------------------------------------------------------

def render_compare(cfg: dict) -> None:
    """Render the Compare Plans tab."""
    st.header("Compare Plans")
    st.caption("Live rates pulled directly from PowerToChoose — the official Texas plan registry.")

    col_fetch, col_filters = st.columns([2, 3])

    with col_fetch:
        if st.button("🔄 Refresh Plans from PowerToChoose", type="primary"):
            st.cache_data.clear()
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

    # Auto-fetch if zip code is set and plans not yet loaded
    if plans_df is None and cfg.get("zip_code"):
        with st.spinner("Loading plans from PowerToChoose…"):
            raw = fetch_plans(cfg["zip_code"])
        if raw is not None:
            st.session_state["plans_df"] = raw
            plans_df = raw

    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")

    if plans_df is None:
        st.info("Click **Refresh Plans** to load available electricity plans for ZIP " + cfg["zip_code"])
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

    st.caption(f"{len(filtered)} plans shown. Select one below to see enrollment details.")

    st.markdown("---")

    st.subheader("Top Plans")

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
                st.success(f"✓ Selected **{provider} — {name}**. Go to the Decision tab to enroll.")

        st.divider()

    # ── Multi-plan side-by-side comparison ───────────────────────────────
    st.markdown("---")
    st.subheader("Side-by-Side Comparison")
    st.caption("Select 2–5 plans to compare in detail.")

    all_labels = [
        f"{row.get('company_name', '')} — {row.get('plan_name', '')}"
        for _, row in filtered.iterrows()
    ]
    chosen = st.multiselect("Pick plans to compare", all_labels, max_selections=5, key="compare_multiselect")

    if len(chosen) >= 2:
        chosen_rows = [filtered.iloc[all_labels.index(c)] for c in chosen]

        # Grouped bar chart: rates at 500 / 1000 / 2000 kWh
        bar_data = []
        for row in chosen_rows:
            name = f"{row.get('company_name', '')} — {row.get('plan_name', '')}"
            for usage, col in [(500, "price_kwh500"), (1000, "price_kwh1000"), (2000, "price_kwh2000")]:
                bar_data.append({"Plan": name, "Usage Level": f"{usage} kWh", "¢/kWh": float(row.get(col, 0))})

        fig_bar = px.bar(
            pd.DataFrame(bar_data),
            x="Usage Level", y="¢/kWh", color="Plan",
            barmode="group",
            title="All-In Rate (¢/kWh) at Each Usage Level",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Comparison table
        compare_rows = []
        for row in chosen_rows:
            compare_rows.append({
                "Provider": row.get("company_name", ""),
                "Plan": row.get("plan_name", ""),
                "Term": f"{int(row.get('term_value', 0))} mo",
                "¢/kWh @ 500 kWh":  f"{float(row.get('price_kwh500', 0)):.1f}",
                "¢/kWh @ 1000 kWh": f"{float(row.get('price_kwh1000', 0)):.1f}",
                "¢/kWh @ 2000 kWh": f"{float(row.get('price_kwh2000', 0)):.1f}",
                "12-mo Savings ($)": f"${row.get('post_contract_savings', 0):,.0f}" if "post_contract_savings" in row else "—",
                "Switch-Now Net ($)": f"${row.get('net_now', 0):,.0f}" if "net_now" in row else "—",
            })
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        # EFL + enroll links
        for row in chosen_rows:
            name = f"{row.get('company_name', '')} — {row.get('plan_name', '')}"
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f"**{name}**")
            if row.get("fact_sheet"):
                c2.link_button("EFL", row["fact_sheet"])
            if row.get("go_to_plan"):
                c3.link_button("Enroll →", row["go_to_plan"])


# ---------------------------------------------------------------------------
# Tab 3: Decision
# ---------------------------------------------------------------------------

def render_decision(cfg: dict) -> None:
    """Render the Decision tab."""
    st.header("Decision")

    usage_df: pd.DataFrame | None = st.session_state.get("usage_df")
    plans_df: pd.DataFrame | None = st.session_state.get("plans_df")

    # Auto-fetch if zip code is set and plans not yet loaded
    if plans_df is None and cfg.get("zip_code"):
        with st.spinner("Loading plans from PowerToChoose…"):
            raw = fetch_plans(cfg["zip_code"])
        if raw is not None:
            st.session_state["plans_df"] = raw
            plans_df = raw

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
            st.success(f"Selected **{label}** — scroll down to see the enrollment checklist and AI assistant.")

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

    # --- Enrollment checklist and AI chat (shown when a plan is selected) ---
    selected: dict | None = st.session_state.get("selected_plan")
    if selected is not None:
        st.markdown("---")

        p_name = selected.get("plan_name", "Unknown Plan")
        p_provider = selected.get("company_name", "Unknown Provider")
        p_term = selected.get("term_value", "?")
        p1000 = selected.get("price_kwh1000", 0)
        post_sav = selected.get("post_contract_savings", "N/A")
        net_now_sel = selected.get("net_now", "N/A")
        efl_url = selected.get("fact_sheet", selected.get("efl_url", "#"))
        enroll_url = selected.get("enroll_url", selected.get("go_to_plan", "#"))

        st.markdown(
            f"""
            <div style="background:#f0fdf4;border:2px solid #00A651;border-radius:8px;padding:20px;margin-bottom:16px;">
            <h3 style="margin:0 0 4px;">Selected: {p_provider} — {p_name}</h3>
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
        if isinstance(net_now_sel, (int, float)):
            c4.metric("Switch-Now Net (after ETF)", f"${net_now_sel:.2f}")
        else:
            c4.metric("Switch-Now Net (after ETF)", "—")

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

        with st.expander("📋 Enrollment Checklist", expanded=True):
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
        st.subheader("🤖 AI Enrollment Assistant")

        api_key_chat = _get_api_key()

        if not api_key_chat:
            st.warning("Add an Anthropic API key in the sidebar to enable the AI chat assistant.")
        else:
            plans_df_chat = st.session_state.get("plans_df")
            best_plan_chat = None
            if plans_df_chat is not None and usage_df is not None:
                try:
                    filtered_chat = filter_plans(plans_df_chat, False, False)
                    if not filtered_chat.empty:
                        scored_chat = score_plans(
                            filtered_chat, usage_df,
                            cfg["base_charge"], cfg["energy_rate"],
                            cfg["tdu_fixed"], cfg["tdu_rate"],
                            cfg["etf"], cfg["contract_end"],
                        )
                        best_plan_chat = scored_chat.sort_values("post_contract_savings", ascending=False).iloc[0]
                except Exception:
                    pass

            decision_str_chat = "Recommendation not yet computed"
            months_rem_chat = cfg["months_remaining"]
            if months_rem_chat == 0:
                decision_str_chat = "SWITCH NOW — Contract already ended"
            elif best_plan_chat is not None and best_plan_chat.get("net_now", 0) > 0:
                decision_str_chat = "SWITCH NOW — ETF is worth paying"
            elif cfg["contract_end"]:
                decision_str_chat = f"STAY — Switch on {cfg['contract_end'].strftime('%B %d, %Y')}"

            sys_prompt_chat = _build_system_prompt(
                cfg["plan_name"], cfg["provider"], usage_df, best_plan_chat, decision_str_chat
            )
            sys_prompt_chat += (
                f"\n\nThe user is currently looking at enrolling in: {p_name} from {p_provider}. "
                "Help them understand the enrollment process, what to watch out for, and answer "
                "any questions about their Texas electricity plan."
            )

            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask anything about enrolling, your plan, or Texas electricity…")
            if user_input:
                st.session_state["chat_history"].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        reply = ai_chat(st.session_state["chat_history"], sys_prompt_chat, api_key_chat)
                    st.markdown(reply)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

            if st.session_state["chat_history"]:
                if st.button("🗑️ Clear chat history"):
                    st.session_state["chat_history"] = []
                    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Handle one-click unsubscribe ──────────────────────────────────────
    token = st.query_params.get("unsubscribe")
    if token:
        st.title("⚡ Texas Electricity Plan Monitor")
        client = _get_supabase_client()
        unsubscribed = False
        if client:
            try:
                result = client.table("user_profiles").select("email").eq(
                    "unsubscribe_token", token
                ).execute()
                if result.data:
                    email = result.data[0]["email"]
                    client.table("user_profiles").delete().eq("unsubscribe_token", token).execute()
                    st.success(f"✓ You've been unsubscribed. **{email}** will no longer receive alerts.")
                    unsubscribed = True
            except Exception:
                pass
        if not unsubscribed:
            st.warning("This unsubscribe link has already been used or is invalid.")
        st.markdown("You can re-enroll any time by visiting the app and entering your email.")
        st.stop()

    st.title("⚡ Texas Electricity Plan Manager")
    st.caption("Know exactly what you're paying, find a better plan, and never miss a renewal.")

    # Render sidebar and collect plan configuration
    cfg = render_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Dashboard", "🔍 Compare Plans", "🎯 Decision"]
    )

    with tab1:
        render_dashboard(cfg)

    with tab2:
        render_compare(cfg)

    with tab3:
        render_decision(cfg)


if __name__ == "__main__":
    main()
