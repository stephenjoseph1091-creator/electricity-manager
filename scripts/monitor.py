"""
monitor.py — Weekly plan monitoring and email alerts.

For each user profile stored in Supabase, this script:
  1. Calculates days until their contract ends
  2. Fetches live plans from PowerToChoose
  3. Finds the best available plan for their usage profile
  4. Sends an email alert when:
     - Contract ends in 60, 30, or 14 days (once per threshold)
     - Contract has already expired (every run until they update their profile)
     - A plan appears that would save $150+ over 12 months vs their current rate

Required environment variables (set as GitHub Secrets):
  SUPABASE_URL      — from Supabase project settings
  SUPABASE_KEY      — service_role key (not anon key) for server-side access
  RESEND_API_KEY    — from resend.com
  FROM_EMAIL        — verified sender address in Resend (e.g. alerts@yourdomain.com)
                      or use onboarding@resend.dev for testing

Usage: python3 scripts/monitor.py
"""

import json
import os
import urllib.request
from datetime import date, datetime

import resend
from dateutil.relativedelta import relativedelta
from supabase import create_client

# ── Config ────────────────────────────────────────────────────────────────────

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
RESEND_API_KEY = os.environ["RESEND_API_KEY"]
FROM_EMAIL     = os.environ.get("FROM_EMAIL", "onboarding@resend.dev")

# Alert when a competing plan saves more than this vs current plan (12 months)
SAVINGS_ALERT_THRESHOLD = 150.0

PTC_URL = "http://api.powertochoose.org/api/PowerToChoose/plans?zip_code={zip_code}"


# ── PowerToChoose helpers ─────────────────────────────────────────────────────

def fetch_plans(zip_code: str) -> list[dict]:
    url = PTC_URL.format(zip_code=zip_code)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept":     "application/json",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read())
    plans = raw.get("data", raw) if isinstance(raw, dict) else raw
    return plans if isinstance(plans, list) else []


def filter_plans(plans: list[dict]) -> list[dict]:
    kept = []
    for p in plans:
        if p.get("prepaid"):                          continue
        if p.get("timeofuse"):                        continue
        if p.get("rate_type", "").lower() != "fixed": continue
        r500  = p.get("price_kwh500")
        r1000 = p.get("price_kwh1000")
        r2000 = p.get("price_kwh2000")
        if not all(isinstance(r, (int, float)) and r > 0 for r in [r500, r1000, r2000]):
            continue
        kept.append(p)
    return kept


def interpolate_rate(kwh: float, r500: float, r1000: float, r2000: float) -> float:
    if kwh <= 500:
        return r500
    elif kwh <= 1000:
        frac = (kwh - 500) / 500
        return r500 + frac * (r1000 - r500)
    elif kwh <= 2000:
        frac = (kwh - 1000) / 1000
        return r1000 + frac * (r2000 - r1000)
    else:
        slope = (r2000 - r1000) / 1000
        return r2000 + slope * (kwh - 2000)


# Typical Texas monthly usage by calendar month (kWh) — used when no usage data
SEASONAL_DEFAULT = {
    1: 1100, 2: 1050, 3: 950, 4: 850, 5: 1100, 6: 1500,
    7: 1800, 8: 1750, 9: 1400, 10: 1000, 11: 900, 12: 1050,
}


def project_12mo_current(contract_end: date, base_charge: float,
                          energy_rate: float, tdu_fixed: float,
                          tdu_rate: float) -> float:
    """Estimate 12-month cost on current plan starting at contract_end."""
    total = 0.0
    fixed    = base_charge + tdu_fixed
    variable = energy_rate + tdu_rate
    for i in range(12):
        d   = contract_end + relativedelta(months=i)
        kwh = SEASONAL_DEFAULT.get(d.month, 1200)
        total += fixed + kwh * variable
    return total


def project_12mo_plan(contract_end: date, plan: dict) -> float:
    """Estimate 12-month cost on a competing plan starting at contract_end."""
    r500, r1000, r2000 = plan["price_kwh500"], plan["price_kwh1000"], plan["price_kwh2000"]
    total = 0.0
    for i in range(12):
        d   = contract_end + relativedelta(months=i)
        kwh = SEASONAL_DEFAULT.get(d.month, 1200)
        rate = interpolate_rate(kwh, r500, r1000, r2000)
        total += kwh * rate / 100
    return total


def best_plan_for_profile(profile: dict, plans: list[dict]) -> dict | None:
    """Return the plan with the highest 12-month post-contract savings, or None."""
    contract_start = datetime.strptime(profile["contract_start"], "%Y-%m-%d").date()
    contract_end   = contract_start + relativedelta(months=int(profile["contract_term_months"]))

    current_12mo = project_12mo_current(
        contract_end,
        float(profile["base_charge"]),
        float(profile["energy_rate"]),
        float(profile["tdu_fixed"]),
        float(profile["tdu_rate"]),
    )

    best = None
    best_savings = -999999.0
    for p in plans:
        new_12mo = project_12mo_plan(contract_end, p)
        savings  = current_12mo - new_12mo
        if savings > best_savings:
            best_savings = savings
            best = {**p, "post_contract_savings": savings, "current_12mo": current_12mo}
    return best


# ── Email builder ─────────────────────────────────────────────────────────────

def _plan_row(p: dict) -> str:
    name     = f"{p.get('company_name', '')} — {p.get('plan_name', '')}"
    savings  = p.get("post_contract_savings", 0)
    rate1000 = float(p.get("price_kwh1000", 0))
    term     = int(p.get("term_value", 0))
    enroll   = p.get("go_to_plan", "")
    efl      = p.get("fact_sheet", "")
    links    = ""
    if efl:
        links += f' &nbsp;<a href="{efl}">EFL</a>'
    if enroll:
        links += f' &nbsp;<a href="{enroll}">Enroll →</a>'
    return (
        f"<tr>"
        f"<td style='padding:6px 12px'>{name}</td>"
        f"<td style='padding:6px 12px;text-align:center'>{term}mo</td>"
        f"<td style='padding:6px 12px;text-align:center'>{rate1000:.1f}¢</td>"
        f"<td style='padding:6px 12px;text-align:center;color:#00A651;font-weight:bold'>${savings:,.0f}</td>"
        f"<td style='padding:6px 12px'>{links}</td>"
        f"</tr>"
    )


APP_URL = "https://electricity-manager.streamlit.app"


def build_email_html(profile: dict, trigger: str, days_remaining: int,
                     contract_end: date, top_plans: list[dict]) -> tuple[str, str]:
    """Return (subject, html_body) for the alert email."""
    name      = profile.get("provider", "Your Provider")
    plan      = profile.get("plan_name", "your current plan")
    end_str   = contract_end.strftime("%B %d, %Y")
    token     = profile.get("unsubscribe_token", "")
    unsub_url = f"{APP_URL}/?unsubscribe={token}" if token else APP_URL

    if trigger == "expired":
        subject = f"⚡ Action needed: Your electricity contract expired {abs(days_remaining)} days ago"
        headline = (
            f"Your contract with <strong>{name}</strong> ({plan}) expired "
            f"<strong>{abs(days_remaining)} days ago</strong>. "
            f"You're likely on a month-to-month rate — switching now could save you money immediately."
        )
    elif trigger == "14_days":
        subject = f"⚡ 14 days left — time to pick your next electricity plan"
        headline = (
            f"Your contract with <strong>{name}</strong> ({plan}) ends on "
            f"<strong>{end_str}</strong> — just 14 days away. "
            f"Switching takes about 5 minutes and the new plan starts the day your contract ends."
        )
    elif trigger == "30_days":
        subject = f"⚡ 30 days until your electricity contract ends"
        headline = (
            f"Your contract with <strong>{name}</strong> ({plan}) ends on "
            f"<strong>{end_str}</strong>. Now is the time to review your options so you're not caught off guard."
        )
    elif trigger == "60_days":
        subject = f"⚡ Heads up — your electricity contract ends in 60 days"
        headline = (
            f"Your contract with <strong>{name}</strong> ({plan}) ends on "
            f"<strong>{end_str}</strong>. Rates are good right now — it's worth a look."
        )
    else:  # better_plan
        savings = top_plans[0].get("post_contract_savings", 0) if top_plans else 0
        subject = f"⚡ A plan just appeared that could save you ${savings:,.0f} on electricity"
        headline = (
            f"A new plan just hit PowerToChoose that would save you an estimated "
            f"<strong>${savings:,.0f}</strong> over 12 months compared to staying on "
            f"<strong>{name} {plan}</strong> after your contract ends."
        )

    rows_html = "".join(_plan_row(p) for p in top_plans[:5])

    html = f"""
    <div style="font-family:sans-serif;max-width:640px;margin:0 auto;color:#222">
      <div style="background:#00A651;padding:20px 24px;border-radius:8px 8px 0 0">
        <h1 style="margin:0;color:#fff;font-size:22px">⚡ Texas Electricity Plan Monitor</h1>
      </div>
      <div style="background:#f9f9f9;padding:24px;border-radius:0 0 8px 8px;border:1px solid #e0e0e0">
        <p style="font-size:16px;line-height:1.6">{headline}</p>

        <h2 style="font-size:16px;margin:24px 0 8px">Top Plans Right Now</h2>
        <table style="width:100%;border-collapse:collapse;font-size:14px">
          <thead>
            <tr style="background:#e8f5e9">
              <th style="padding:6px 12px;text-align:left">Plan</th>
              <th style="padding:6px 12px">Term</th>
              <th style="padding:6px 12px">¢/kWh @ 1000</th>
              <th style="padding:6px 12px">Est. 12-mo savings</th>
              <th style="padding:6px 12px">Links</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>

        <p style="font-size:12px;color:#888;margin-top:24px">
          Savings are estimated vs continuing on your current plan for 12 months after contract end,
          using typical Texas seasonal usage patterns. Always read the EFL before enrolling.<br><br>
          You're receiving this because you signed up at the Texas Electricity Plan Monitor.
          Don't want these emails? <a href="{unsub_url}" style="color:#00A651">Unsubscribe in one click</a>.
        </p>
      </div>
    </div>
    """
    return subject, html


# ── Core check logic ──────────────────────────────────────────────────────────

def check_profile(profile: dict, supabase) -> None:
    email = profile["email"]
    print(f"  Checking {email}…")

    today          = date.today()
    contract_start = datetime.strptime(profile["contract_start"], "%Y-%m-%d").date()
    contract_end   = contract_start + relativedelta(months=int(profile["contract_term_months"]))
    days_remaining = (contract_end - today).days

    # Fetch and filter live plans
    try:
        raw_plans = fetch_plans(profile["zip_code"])
        plans     = filter_plans(raw_plans)
        print(f"    {len(plans)} fixed-rate plans loaded for ZIP {profile['zip_code']}")
    except Exception as exc:
        print(f"    WARNING: could not fetch plans — {exc}")
        plans = []

    # Score plans
    top_plans = []
    best      = None
    if plans:
        best = best_plan_for_profile(profile, plans)
        # Build a ranked list for the email
        scored = []
        for p in plans:
            contract_start_d = datetime.strptime(profile["contract_start"], "%Y-%m-%d").date()
            ce               = contract_start_d + relativedelta(months=int(profile["contract_term_months"]))
            current_12mo     = project_12mo_current(
                ce,
                float(profile["base_charge"]), float(profile["energy_rate"]),
                float(profile["tdu_fixed"]),   float(profile["tdu_rate"]),
            )
            savings = current_12mo - project_12mo_plan(ce, p)
            scored.append({**p, "post_contract_savings": savings})
        top_plans = sorted(scored, key=lambda x: x["post_contract_savings"], reverse=True)[:5]

    # Determine trigger
    trigger = None
    updates = {}

    if days_remaining < 0:
        trigger = "expired"          # always alert when expired; no flag to set
    elif days_remaining <= 14 and not profile.get("alert_14_sent"):
        trigger = "14_days"
        updates["alert_14_sent"] = True
    elif days_remaining <= 30 and not profile.get("alert_30_sent"):
        trigger = "30_days"
        updates["alert_30_sent"] = True
    elif days_remaining <= 60 and not profile.get("alert_60_sent"):
        trigger = "60_days"
        updates["alert_60_sent"] = True
    elif best and best.get("post_contract_savings", 0) > SAVINGS_ALERT_THRESHOLD:
        trigger = "better_plan"

    if trigger is None:
        print(f"    No alert needed (days remaining: {days_remaining})")
        return

    # Build and send email
    subject, html = build_email_html(profile, trigger, days_remaining, contract_end, top_plans)
    try:
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from":    FROM_EMAIL,
            "to":      [email],
            "subject": subject,
            "html":    html,
        })
        print(f"    ✓ Alert sent ({trigger})")
    except Exception as exc:
        print(f"    ERROR sending email: {exc}")
        return

    # Update alert flags
    if updates:
        updates["updated_at"] = datetime.utcnow().isoformat()
        supabase.table("user_profiles").update(updates).eq("email", email).execute()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Monitor run: {date.today()}")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    profiles = supabase.table("user_profiles").select("*").execute().data
    print(f"Loaded {len(profiles)} user profile(s)")

    for profile in profiles:
        try:
            check_profile(profile, supabase)
        except Exception as exc:
            print(f"  ERROR processing {profile.get('email', '?')}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
