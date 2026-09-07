"""Synthetic-but-realistic BFSI datasets for the curriculum.

Everything here is generated from a fixed seed, so the numbers in the notebooks
are reproducible. The generators deliberately bake in the problems you meet in
regulated-industry modelling work:

* class imbalance that ranges from mild (churn) to brutal (card fraud)
* missingness that is *not* missing-at-random
* a genuine leakage trap (a field only populated after the outcome is known)
* temporal drift, so a random split flatters the model and a time split does not
* correlated proxies for protected attributes, for the fairness lab

No real customer data is involved. The causal structure is invented; treat
coefficient values as pedagogy, not as domain truth.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 20261007

DATA_DIR = Path(
    os.environ.get(
        "SKMASTERY_DATA",
        Path(__file__).resolve().parents[2] / "data",
    )
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# 1. Credit risk — the workhorse dataset for most modules
# --------------------------------------------------------------------------
def make_credit_risk(n: int = 12_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Consumer loan applications with a 36-month default outcome.

    Columns
    -------
    application_id       : str, unique
    application_month    : period-like string 'YYYY-MM' spanning 3 years
    age                  : int
    employment_years     : float, missing for the self-employed (MNAR)
    annual_income        : float, right-skewed
    loan_amount          : float
    term_months          : int in {12, 24, 36, 48, 60}
    interest_rate        : float, priced off the (unobserved) latent risk
    debt_to_income       : float
    credit_score         : int, missing for thin-file applicants (MNAR)
    n_open_accounts      : int
    n_delinq_2yr         : int, zero-inflated
    home_ownership       : category
    purpose              : category
    employment_type      : category
    region               : category (correlates with the fairness proxy)
    channel              : category, mix drifts over time
    prior_customer       : bool
    collections_flag     : LEAKAGE — only ever set after default is known
    default              : target, 1 = defaulted within 36 months
    """
    rng = np.random.default_rng(seed)

    months = pd.period_range("2022-01", periods=36, freq="M")
    # Volume grows over time, so later months carry more weight.
    weights = np.linspace(0.6, 1.6, len(months))
    weights = weights / weights.sum()
    month_idx = rng.choice(len(months), size=n, p=weights)
    application_month = months[month_idx].astype(str)
    t = month_idx / (len(months) - 1)  # 0 -> 1 across the window

    age = np.clip(np.exp(rng.normal(np.log(38.0), 0.30, size=n)), 19, 85).astype(int)

    employment_type = rng.choice(
        ["salaried", "self_employed", "contract", "retired", "unemployed"],
        size=n,
        p=[0.58, 0.18, 0.13, 0.08, 0.03],
    )

    employment_years = np.clip(
        rng.gamma(shape=2.0, scale=3.0, size=n) * (age / 40.0), 0, 45
    )

    log_income = (
        10.2
        + 0.020 * (age - 40)
        - 0.00022 * (age - 40) ** 2
        + 0.030 * np.minimum(employment_years, 20)
        + rng.normal(0, 0.42, size=n)
    )
    log_income += np.select(
        [
            employment_type == "self_employed",
            employment_type == "contract",
            employment_type == "retired",
            employment_type == "unemployed",
        ],
        [0.10, -0.06, -0.30, -0.85],
        default=0.0,
    )
    annual_income = np.exp(log_income).round(-2)

    term_months = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.10, 0.22, 0.34, 0.20, 0.14])

    loan_amount = np.clip(
        annual_income * rng.beta(2.0, 6.0, size=n) * 1.9, 1_000, 120_000
    ).round(-2)

    home_ownership = rng.choice(
        ["rent", "mortgage", "own"], size=n, p=[0.42, 0.44, 0.14]
    )
    purpose = rng.choice(
        [
            "debt_consolidation",
            "home_improvement",
            "major_purchase",
            "medical",
            "small_business",
            "auto",
            "other",
        ],
        size=n,
        p=[0.38, 0.16, 0.12, 0.07, 0.06, 0.13, 0.08],
    )
    region = rng.choice(
        ["northeast", "midwest", "south", "west"], size=n, p=[0.20, 0.21, 0.37, 0.22]
    )

    # Acquisition channel mix drifts: digital grows, branch shrinks.
    p_digital = 0.30 + 0.35 * t
    channel = np.where(
        rng.random(n) < p_digital,
        "digital",
        rng.choice(["branch", "broker", "partner"], size=n, p=[0.45, 0.33, 0.22]),
    )

    prior_customer = rng.random(n) < (0.28 + 0.10 * (home_ownership == "mortgage"))

    n_open_accounts = rng.poisson(lam=np.clip(2 + age / 14, 2, 12), size=n)
    n_delinq_2yr = rng.poisson(
        lam=np.clip(0.22 + 0.6 * (employment_type == "unemployed"), 0.05, 3.0), size=n
    ) * (rng.random(n) < 0.35)

    monthly_payment = loan_amount / term_months * 1.09
    other_debt = annual_income / 12 * rng.beta(2.0, 9.0, size=n)
    debt_to_income = np.clip(
        (monthly_payment + other_debt) / (annual_income / 12), 0.01, 1.4
    )

    # Latent creditworthiness -> observed credit score (noisy proxy).
    latent = (
        0.9 * np.log1p(annual_income / 40_000)
        - 1.9 * debt_to_income
        - 0.42 * n_delinq_2yr
        + 0.028 * np.minimum(employment_years, 15)
        + 0.35 * prior_customer
        + rng.normal(0, 0.55, size=n)
    )
    credit_score = np.clip(640 + 62 * latent + rng.normal(0, 26, size=n), 300, 850).astype(int)

    # Risk-based pricing: the lender already knows a lot. This makes
    # interest_rate a strong, partly-circular feature -- a great discussion.
    interest_rate = np.clip(
        18.6
        - 0.0155 * (credit_score - 300)
        + 3.1 * debt_to_income
        + 0.020 * term_months
        + rng.normal(0, 0.85, size=n),
        3.0,
        30.0,
    ).round(2)

    # Macro shock in the final third of the window -> temporal drift in the target.
    macro = 0.55 * np.clip(t - 0.62, 0, None) / 0.38

    logit = (
        -4.62
        - 1.05 * latent
        + 2.35 * debt_to_income
        + 0.075 * interest_rate
        + 0.30 * n_delinq_2yr
        + 0.34 * (employment_type == "unemployed")
        + 0.16 * (purpose == "small_business")
        - 0.20 * prior_customer
        + 0.011 * (term_months - 36)
        + 1.35 * macro
        + rng.normal(0, 0.30, size=n)
    )
    default = (rng.random(n) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame(
        {
            "application_id": [f"APP{i:06d}" for i in range(n)],
            "application_month": application_month,
            "age": age,
            "employment_years": employment_years.round(1),
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "term_months": term_months,
            "interest_rate": interest_rate,
            "debt_to_income": debt_to_income.round(4),
            "credit_score": credit_score,
            "n_open_accounts": n_open_accounts,
            "n_delinq_2yr": n_delinq_2yr,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "employment_type": employment_type,
            "region": region,
            "channel": channel,
            "prior_customer": prior_customer,
            "default": default,
        }
    )

    # --- Missingness, deliberately NOT at random -------------------------
    # Self-employed applicants rarely have verifiable tenure.
    mask_emp = (df["employment_type"] == "self_employed") & (rng.random(n) < 0.62)
    mask_emp |= rng.random(n) < 0.03
    df.loc[mask_emp, "employment_years"] = np.nan

    # Thin-file applicants (young, few accounts) have no bureau score.
    thin = _sigmoid(-0.40 - 0.06 * (df["age"] - 28) - 0.28 * df["n_open_accounts"])
    df.loc[rng.random(n) < thin, "credit_score"] = np.nan

    # A few income values were never verified.
    df.loc[rng.random(n) < 0.018, "annual_income"] = np.nan

    # --- The leakage trap ------------------------------------------------
    # `collections_flag` is written by the servicing system only once an
    # account has already gone bad. It is in the warehouse table, it is
    # tempting, and it will give you AUC ~0.97 and a worthless model.
    df["collections_flag"] = np.where(
        df["default"] == 1, (rng.random(n) < 0.88).astype(int), (rng.random(n) < 0.02).astype(int)
    )

    cols = [c for c in df.columns if c != "default"] + ["default"]
    return df[cols]


# --------------------------------------------------------------------------
# 2. Card fraud — extreme imbalance, time-ordered
# --------------------------------------------------------------------------
def make_card_fraud(n: int = 60_000, seed: int = RANDOM_STATE + 1) -> pd.DataFrame:
    """Card-present / card-not-present transactions, ~0.5% fraud.

    Ordered by `timestamp`. Fraud arrives in bursts (compromised BINs), which
    is exactly why an i.i.d. shuffle split lies to you here.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2025-01-01")
    offsets = np.sort(rng.uniform(0, 180 * 24 * 3600, size=n))
    ts = start + pd.to_timedelta(offsets, unit="s")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()

    amount = np.round(np.exp(rng.normal(3.05, 1.15, size=n)), 2)
    merchant_category = rng.choice(
        ["grocery", "fuel", "restaurant", "travel", "electronics", "gaming", "pharmacy", "other"],
        size=n,
        p=[0.24, 0.14, 0.18, 0.07, 0.08, 0.06, 0.09, 0.14],
    )
    entry_mode = rng.choice(
        ["chip", "contactless", "ecommerce", "magstripe", "manual"],
        size=n,
        p=[0.34, 0.30, 0.28, 0.05, 0.03],
    )
    is_cnp = np.isin(entry_mode, ["ecommerce", "manual"]).astype(int)

    card_id = rng.integers(0, 9_000, size=n)
    device_age_days = np.where(
        rng.random(n) < 0.12, rng.integers(0, 3, size=n), rng.integers(3, 900, size=n)
    )
    country_mismatch = (rng.random(n) < 0.04).astype(int)
    n_txn_1h = rng.poisson(0.6, size=n)
    # Each card has its own spend level; the ratio is what a real feature store
    # would compute, so it carries signal rather than being pure noise.
    card_median = np.exp(rng.normal(3.05, 0.60, size=9_000))[card_id]
    amount_vs_card_median = np.round(amount / card_median, 3)

    # Per-card risk. Most cards carry a mild idiosyncratic effect; ~2% are
    # compromised and see repeated fraud. This is what makes `card_id` a
    # *group*: transactions on the same card are not independent, so splitting
    # them across train and test lets a model memorise the card. Module 04
    # measures exactly that.
    card_effect = rng.normal(0, 0.70, size=9_000)
    card_effect[rng.random(9_000) < 0.016] += 9.80
    card_risk = card_effect[card_id]

    # Fraud bursts: a handful of windows where compromised cards are hit hard.
    burst_centres = rng.uniform(0.05, 0.95, size=9)
    tnorm = offsets / offsets.max()
    burst = np.zeros(n)
    for c in burst_centres:
        burst += np.exp(-0.5 * ((tnorm - c) / 0.012) ** 2)
    burst = np.clip(burst, 0, 1.6)

    logit = (
        -14.95
        + 2.90 * is_cnp
        + 2.30 * (entry_mode == "magstripe")
        + 3.70 * country_mismatch
        + 1.25 * np.log1p(amount / 100)
        + 1.70 * np.log(np.clip(amount_vs_card_median, 1e-3, None))
        + 0.80 * n_txn_1h
        + 2.50 * (device_age_days < 3)
        + 1.15 * np.isin(merchant_category, ["electronics", "gaming", "travel"])
        + 0.75 * ((hour >= 1) & (hour <= 5))
        + 1.10 * burst
        + card_risk
    )
    is_fraud = (rng.random(n) < _sigmoid(logit)).astype(int)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "card_id": card_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "entry_mode": entry_mode,
            "is_cnp": is_cnp,
            "hour": hour,
            "day_of_week": dow,
            "device_age_days": device_age_days,
            "country_mismatch": country_mismatch,
            "n_txn_1h": n_txn_1h,
            "amount_vs_card_median": amount_vs_card_median,
            "is_fraud": is_fraud,
        }
    )


# --------------------------------------------------------------------------
# 3. Insurance claims — skewed, zero-inflated regression target
# --------------------------------------------------------------------------
def make_insurance_claims(n: int = 15_000, seed: int = RANDOM_STATE + 2) -> pd.DataFrame:
    """Motor policies with exposure, claim count and claim cost.

    The cost target is zero-inflated and heavy-tailed, which is where squared
    error quietly stops being the right loss and Poisson / Tweedie start.
    """
    rng = np.random.default_rng(seed)

    exposure = np.clip(rng.beta(5, 2, size=n), 0.08, 1.0).round(3)
    driver_age = np.clip(rng.gamma(8, 5, size=n) + 18, 18, 92).astype(int)
    vehicle_age = np.clip(rng.gamma(2.2, 3.0, size=n), 0, 30).round(1)
    vehicle_power = np.clip(rng.normal(115, 32, size=n), 45, 320).round(0)
    bonus_malus = np.clip(rng.gamma(3, 18, size=n) + 50, 50, 230).astype(int)
    annual_mileage = np.clip(rng.normal(12_500, 4_800, size=n), 1_000, 45_000).round(-2)
    area = rng.choice(["urban", "suburban", "rural"], size=n, p=[0.42, 0.38, 0.20])
    fuel = rng.choice(["petrol", "diesel", "hybrid", "electric"], size=n, p=[0.48, 0.31, 0.14, 0.07])
    coverage = rng.choice(["third_party", "comprehensive"], size=n, p=[0.38, 0.62])

    log_lambda = (
        -2.35
        + np.log(exposure)
        + 0.0135 * (bonus_malus - 100)
        + 0.30 * (area == "urban")
        - 0.16 * (area == "rural")
        + 0.000021 * (annual_mileage - 12_500)
        + 0.0045 * (vehicle_power - 115)
        + 0.85 * np.exp(-((driver_age - 22) ** 2) / 90)
        + 0.35 * np.exp(-((driver_age - 82) ** 2) / 120)
    )
    claim_count = rng.poisson(np.exp(log_lambda))

    severity_scale = np.exp(
        7.10
        + 0.0032 * (vehicle_power - 115)
        - 0.021 * vehicle_age
        + 0.22 * (coverage == "comprehensive")
        + 0.12 * (area == "urban")
    )
    claim_cost = np.array(
        [
            rng.gamma(shape=1.55, scale=s / 1.55, size=k).sum() if k else 0.0
            for k, s in zip(claim_count, severity_scale)
        ]
    ).round(2)

    return pd.DataFrame(
        {
            "policy_id": [f"POL{i:06d}" for i in range(n)],
            "exposure": exposure,
            "driver_age": driver_age,
            "vehicle_age": vehicle_age,
            "vehicle_power": vehicle_power,
            "bonus_malus": bonus_malus,
            "annual_mileage": annual_mileage,
            "area": area,
            "fuel": fuel,
            "coverage": coverage,
            "claim_count": claim_count,
            "claim_cost": claim_cost,
        }
    )


# --------------------------------------------------------------------------
# 4. Telco churn — categorical-heavy, mild imbalance
# --------------------------------------------------------------------------
def make_telco_churn(n: int = 8_000, seed: int = RANDOM_STATE + 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=n)
    contract = rng.choice(
        ["month_to_month", "one_year", "two_year"], size=n, p=[0.55, 0.24, 0.21]
    )
    internet = rng.choice(["dsl", "fiber", "none"], size=n, p=[0.34, 0.47, 0.19])
    payment = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        size=n,
        p=[0.34, 0.19, 0.24, 0.23],
    )
    paperless = rng.random(n) < 0.59
    n_services = rng.integers(0, 7, size=n)
    monthly_charges = np.round(
        20
        + 26 * (internet == "fiber")
        + 12 * (internet == "dsl")
        + 5.2 * n_services
        + rng.normal(0, 4.5, size=n),
        2,
    )
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.93, 1.05, n), 2)
    support_tickets_90d = rng.poisson(0.7, size=n)
    autopay = np.isin(payment, ["bank_transfer", "credit_card"])

    logit = (
        -0.95
        - 0.055 * tenure_months
        + 1.55 * (contract == "month_to_month")
        - 1.05 * (contract == "two_year")
        + 0.85 * (internet == "fiber")
        + 0.70 * (payment == "electronic_check")
        - 0.45 * autopay
        + 0.020 * (monthly_charges - 65)
        + 0.40 * support_tickets_90d
        - 0.16 * n_services
        + rng.normal(0, 0.20, size=n)
    )
    churn = (rng.random(n) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUS{i:06d}" for i in range(n)],
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "n_services": n_services,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets_90d": support_tickets_90d,
            "churn": churn,
        }
    )
    # Classic real-world quirk: TotalCharges is blank for brand-new accounts.
    df.loc[df["tenure_months"] <= 1, "total_charges"] = np.nan
    return df


# --------------------------------------------------------------------------
# 5. Support tickets — short text for the NLP module
# --------------------------------------------------------------------------
def make_support_tickets(n: int = 6_000, seed: int = RANDOM_STATE + 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    templates = {
        "fraud_dispute": [
            "I did not authorise the {amount} charge from {merchant} on my card",
            "there are {k} transactions I do not recognise please block the card",
            "someone used my debit card at {merchant}, I want a chargeback",
            "unauthorised payment of {amount} showing on statement, card was in my wallet",
            "my account was compromised after a phishing text, {amount} is missing",
        ],
        "billing": [
            "why was I charged {amount} this month when my plan is fixed",
            "the annual fee of {amount} was applied twice to my statement",
            "please explain the interest charge on my last statement",
            "I was billed {amount} after I already cancelled the service",
            "duplicate direct debit taken, need a refund of {amount}",
        ],
        "account_access": [
            "cannot log in to the app, it says my credentials are invalid",
            "the one time passcode never arrives on my phone",
            "locked out after three attempts, need my password reset",
            "biometric login stopped working after the latest update",
            "my username is not recognised even though the account is active",
        ],
        "loan_servicing": [
            "I want to make an overpayment of {amount} on my loan",
            "can I change my monthly instalment date to the {k}th",
            "requesting a payment holiday for {k} months on my loan",
            "what is the early settlement figure for my loan account",
            "my direct debit for the loan failed, how do I catch up",
        ],
        "complaint": [
            "I have been waiting {k} weeks for a response and this is unacceptable",
            "the branch staff were rude and refused to escalate my case",
            "third time raising this issue, nobody has called me back",
            "I want to make a formal complaint about the handling of my case",
            "this is the worst service I have received, I am moving my accounts",
        ],
        "general_enquiry": [
            "what are your branch opening hours on a saturday",
            "how do I order a replacement card for a lost one",
            "do you offer joint accounts for students",
            "what documents do I need to update my address",
            "is there a fee for withdrawing cash abroad",
        ],
    }
    merchants = ["AMZN MKTPLACE", "STEAM GAMES", "UBER TRIP", "TESCO EXPRESS", "APPLE.COM/BILL", "RYANAIR"]
    labels = list(templates)
    probs = np.array([0.14, 0.22, 0.19, 0.15, 0.11, 0.19])

    # Vague messages that a human agent could route to almost anything.
    # Without these the task is trivially separable and every model scores 1.0,
    # which teaches you nothing about model selection.
    ambiguous = [
        "there is a problem with my account please call me",
        "payment issue on my account, need this sorted today",
        "my card is not working, what do I do",
        "something is wrong with my statement",
        "need help with a charge on my account",
        "please review my account, the numbers look wrong",
        "issue with my payment, second time asking",
        "money missing from my account",
        "can someone look at my account urgently",
        "the amount is not what I expected",
    ]
    openers = ["", "hi ", "hello ", "good morning ", "dear team ", "hi team "]
    closers = ["", ".", "!", "!!", " thanks", " please help", " regards", " many thanks"]

    rows = []
    for i in range(n):
        label = rng.choice(labels, p=probs)
        if rng.random() < 0.16:
            text = ambiguous[rng.integers(len(ambiguous))]
        else:
            tmpl = templates[label][rng.integers(len(templates[label]))]
            text = tmpl.format(
                amount=f"£{rng.integers(5, 900)}.{rng.integers(0, 99):02d}",
                merchant=merchants[rng.integers(len(merchants))],
                k=int(rng.integers(2, 9)),
            )
        text = openers[rng.integers(len(openers))] + text + closers[rng.integers(len(closers))]
        if rng.random() < 0.25:
            text = text.upper()
        if rng.random() < 0.18:
            text = text.replace("the ", "teh ", 1)
        if rng.random() < 0.12:
            text = text.replace("account", "acount", 1)

        # ~5% of tickets are mislabelled by the agent who tagged them, which is
        # what an irreducible-error ceiling looks like in practice.
        recorded = label if rng.random() > 0.05 else rng.choice(labels)
        rows.append(
            {
                "ticket_id": f"TCK{i:06d}",
                "text": text,
                "channel": rng.choice(["app", "email", "phone_transcript", "web"], p=[0.4, 0.3, 0.12, 0.18]),
                "category": recorded,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Loaders (build-and-cache to CSV/Parquet under data/)
# --------------------------------------------------------------------------
_BUILDERS = {
    "credit_risk": make_credit_risk,
    "card_fraud": make_card_fraud,
    "insurance_claims": make_insurance_claims,
    "telco_churn": make_telco_churn,
    "support_tickets": make_support_tickets,
}


def _load(name: str, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    if refresh or not path.exists():
        _BUILDERS[name]().to_csv(path, index=False)
    parse_dates = ["timestamp"] if name == "card_fraud" else None
    return pd.read_csv(path, parse_dates=parse_dates)


def load_credit_risk(refresh: bool = False) -> pd.DataFrame:
    return _load("credit_risk", refresh)


def load_card_fraud(refresh: bool = False) -> pd.DataFrame:
    return _load("card_fraud", refresh)


def load_insurance_claims(refresh: bool = False) -> pd.DataFrame:
    return _load("insurance_claims", refresh)


def load_telco_churn(refresh: bool = False) -> pd.DataFrame:
    return _load("telco_churn", refresh)


def load_support_tickets(refresh: bool = False) -> pd.DataFrame:
    return _load("support_tickets", refresh)


def build_all(refresh: bool = True) -> dict[str, tuple[int, int]]:
    out = {}
    for name in _BUILDERS:
        df = _load(name, refresh=refresh)
        out[name] = df.shape
    return out


if __name__ == "__main__":
    for name, shape in build_all().items():
        print(f"{name:20s} {shape}")
