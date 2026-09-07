# %% [markdown]
# # Solutions — Module 01: Data Representation

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skmastery import load_card_fraud, load_credit_risk, load_insurance_claims, load_support_tickets, load_telco_churn, set_plot_style

set_plot_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## 1.1 — Recover a planted signal

# %%
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=2000, n_features=40, n_informative=6, n_redundant=6,
    class_sep=0.8, flip_y=0.03, shuffle=False, random_state=0,
)
INFORMATIVE, REDUNDANT = set(range(6)), set(range(6, 12))

et = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(X, y)
et_rank = np.argsort(et.feature_importances_)[::-1][:6]

l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=4000).fit(StandardScaler().fit_transform(X), y)
l1_rank = np.argsort(np.abs(l1.coef_.ravel()))[::-1][:6]

summary = pd.DataFrame([
    {"method": "ExtraTrees importance", "top6": list(et_rank),
     "n_informative": len(set(et_rank) & INFORMATIVE), "n_redundant": len(set(et_rank) & REDUNDANT),
     "n_noise": len(set(et_rank) - INFORMATIVE - REDUNDANT)},
    {"method": "L1 logistic |coef|", "top6": list(l1_rank),
     "n_informative": len(set(l1_rank) & INFORMATIVE), "n_redundant": len(set(l1_rank) & REDUNDANT),
     "n_noise": len(set(l1_rank) - INFORMATIVE - REDUNDANT)},
]).set_index("method")
summary

# %% [markdown]
# **Why they differ.** The redundant features are exact linear combinations of
# the informative ones, so they carry the same information.
#
# - The **tree ensemble** picks whichever of the equivalent columns happens to
#   give a marginally better split at each node, so importance is spread across
#   the informative/redundant group roughly at random. It is not "fooled" so much
#   as *indifferent* — which is the same problem when you read the ranking as a
#   causal statement.
# - **L1** actively selects: once it has one member of a collinear group, adding
#   another costs penalty and buys nothing, so it zeroes the rest. Which member
#   survives is arbitrary (Module 05's collinearity result), but the *count* is
#   right.
#
# Neither can distinguish informative from redundant, because **nothing in the
# data can** — they are informationally identical. That is the real lesson.

# %% [markdown]
# ## 1.2 — Diagnose five broken targets

# %%
from sklearn.utils.multiclass import type_of_target

cases = {
    "y1 DataFrame": pd.Series([0, 1, 1, 0]).to_frame(),
    "y2 numeric strings": np.array(["1", "0", "1", "0"]),
    "y3 float class ids": np.arange(10) / 1.0,
    "y4 2-D binary": np.array([[1, 0], [0, 1], [1, 1], [0, 0]]),
    "y5 NaN in target": pd.Series([0, 1, 1, np.nan]),
}

rows = []
for name, y_ in cases.items():
    try:
        t = type_of_target(y_)
    except Exception as e:
        t = f"raises {type(e).__name__}"
    rows.append({"case": name, "type_of_target": t})
pd.DataFrame(rows).set_index("case")

# %%
diagnosis = {
    "y1 DataFrame": ("Inferred as multilabel-indicator or triggers DataConversionWarning. "
                     "FIX: y.squeeze() or y.iloc[:, 0] — y must be 1-D."),
    "y2 numeric strings": ("Correctly binary; sklearn handles string labels and stores them in "
                           "classes_. FIX: none needed, but note predict() returns STRINGS, so "
                           "downstream arithmetic will fail."),
    "y3 float class ids": ("Ten distinct floats with no repeats reads as continuous -> sklearn "
                           "thinks regression. FIX: y.astype(int) if these are class ids."),
    "y4 2-D binary": ("multilabel-indicator: each row can have several labels. If you meant "
                      "two separate binary problems, that is multi-output and needs "
                      "MultiOutputClassifier. FIX: state which you meant."),
    "y5 NaN in target": ("NaN in y raises at fit time. FIX: drop those rows — imputing a target "
                         "invents the thing you are trying to predict."),
}
for k, v in diagnosis.items():
    print(f"{k}\n  {v}\n")

# %%
# Demonstrating y3, the subtle one.
from sklearn.model_selection import cross_val_score

X_small = np.random.default_rng(0).normal(size=(200, 4))
y_float = np.repeat(np.arange(4) / 1.0, 50)
print("as floats:", type_of_target(y_float))
print("as ints  :", type_of_target(y_float.astype(int)))

# %% [markdown]
# `[0.0, 1.0, 2.0, 3.0]` with repeats is inferred as `multiclass`; ten distinct
# floats with no repeats is `continuous`. The inference depends on the *number of
# distinct values*, which makes it fragile exactly when your classes are numerous.

# %% [markdown]
# ## 1.3 — Sparse budget

# %%
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

fraud = load_card_fraud()
n_cards = fraud["card_id"].nunique()
print(f"{len(fraud):,} rows x {n_cards:,} distinct cards")
print(f"dense one-hot would be : {len(fraud) * n_cards * 8 / 1e9:.2f} GB")

card = fraud[["card_id"]].astype(str)
sparse = OneHotEncoder(handle_unknown="ignore").fit_transform(card)
sparse_mb = (sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes) / 1e6
print(f"sparse one-hot         : {sparse_mb:.2f} MB   ({len(fraud) * n_cards * 8 / 1e9 * 1000 / sparse_mb:,.0f}x smaller)")

# %%
numeric = fraud[["amount", "hour", "day_of_week", "device_age_days", "country_mismatch",
                 "n_txn_1h", "amount_vs_card_median", "is_cnp"]].to_numpy(dtype=float)
y = fraud["is_fraud"].to_numpy()

rows = []
for label, enc in [("card_id one-hot (all)", OneHotEncoder(handle_unknown="ignore")),
                   ("card_id one-hot (min_frequency=50)", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                   ("no card_id", None)]:
    if enc is None:
        M = sp.csr_matrix(numeric)
        n_cols = numeric.shape[1]
    else:
        C = enc.fit_transform(card)
        M = sp.hstack([sp.csr_matrix(numeric), C]).tocsr()
        n_cols = M.shape[1]
    Xtr, Xte, ytr, yte = train_test_split(M, y, test_size=0.3, stratify=y, random_state=0)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=60, random_state=0).fit(Xtr, ytr)
    rows.append({"features": label, "n_columns": n_cols,
                 "roc_auc": round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 4)})
pd.DataFrame(rows).set_index("features")

# %% [markdown]
# **Should `card_id` be a feature at all?** No — not as an identity.
#
# Two reasons. First, a card seen once in training gets one weight fitted from a
# handful of transactions, which is noise; a card *never* seen in training gets
# an all-zero column and no information at all, which is the majority case at
# inference time. Second and more seriously, it invites the group leakage Module
# 04 measures: a model that memorises which cards were compromised scores well in
# a random split and fails on new cards.
#
# What you actually want is **card-level aggregate features computed from past
# transactions only** — spend velocity, deviation from the card's own median,
# time since last transaction. Those generalise to unseen cards; the identity
# does not.

# %% [markdown]
# ## 1.4 — Build a data-quality report

# %%
from scipy.stats import skew


def data_quality_report(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    n = len(df)
    sentinels = (-1, -999, -9999, 999, 9999, 99999)
    rows = []
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        top_share = s.value_counts(normalize=True, dropna=False).iloc[0] if n else np.nan
        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "cardinality_ratio": round(nunique / n, 4),
            "near_constant": bool(top_share > 0.95),
            "is_identifier": bool(nunique == n and n > 20),
            "skew": np.nan,
            "sentinel_suspect": "",
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = s.dropna()
            rec["skew"] = round(float(skew(vals)), 2) if len(vals) > 2 else np.nan
            hits = [v for v in sentinels if (s == v).mean() > 0.005]
            rec["sentinel_suspect"] = ", ".join(map(str, hits))
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("column")
    out["severity"] = (
        (out["missing_pct"] > 20).astype(int) * 3
        + out["near_constant"].astype(int) * 3
        + out["is_identifier"].astype(int) * 2
        + (out["sentinel_suspect"] != "").astype(int) * 3
        + (out["skew"].abs() > 3).fillna(False).astype(int)
        + (out["cardinality_ratio"].between(0.2, 0.99)).astype(int)
    )
    return out.sort_values("severity", ascending=False)


for name, loader in [("credit_risk", load_credit_risk), ("card_fraud", load_card_fraud),
                     ("insurance_claims", load_insurance_claims), ("telco_churn", load_telco_churn),
                     ("support_tickets", load_support_tickets)]:
    rep = data_quality_report(loader())
    worst = rep.index[0]
    print(f"{name:18s} worst column: {worst:20s} severity {rep.iloc[0]['severity']}  "
          f"missing {rep.iloc[0]['missing_pct']}%  id={rep.iloc[0]['is_identifier']}  "
          f"skew={rep.iloc[0]['skew']}")

# %%
data_quality_report(load_credit_risk(), target="default").head(8)

# %% [markdown]
# The identifier columns dominate the ranking, which is right — a column with
# one distinct value per row is either a key (drop it) or a leak (investigate).
# `claim_cost` and `annual_income` surface on skew, which is the flag that says
# "consider a log transform or a distribution-matched loss".

# %% [markdown]
# ## 1.5 — MNAR in practice

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

credit = load_credit_risk()
num = credit.select_dtypes(include=[np.number]).drop(columns=["default"])
y_c = credit["default"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# A: drop rows with a missing credit_score
keep = credit["credit_score"].notna()
a = cross_val_score(HistGradientBoostingClassifier(random_state=0, max_iter=200),
                    num[keep], y_c[keep], cv=cv, scoring="roc_auc", n_jobs=-1)

b = cross_val_score(Pipeline([("i", SimpleImputer(strategy="median")),
                              ("m", HistGradientBoostingClassifier(random_state=0, max_iter=200))]),
                    num, y_c, cv=cv, scoring="roc_auc", n_jobs=-1)

c = cross_val_score(Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                              ("m", HistGradientBoostingClassifier(random_state=0, max_iter=200))]),
                    num, y_c, cv=cv, scoring="roc_auc", n_jobs=-1)

pd.DataFrame([
    {"treatment": "A: drop missing rows", "n_rows": int(keep.sum()), "roc_auc": round(a.mean(), 4), "std": round(a.std(), 4)},
    {"treatment": "B: median impute", "n_rows": len(num), "roc_auc": round(b.mean(), 4), "std": round(b.std(), 4)},
    {"treatment": "C: median + indicator", "n_rows": len(num), "roc_auc": round(c.mean(), 4), "std": round(c.std(), 4)},
]).set_index("treatment")

# %%
missing = credit["credit_score"].isna()
print(f"default rate where credit_score is MISSING: {y_c[missing].mean():.4f}  (n={missing.sum():,})")
print(f"default rate where credit_score is PRESENT: {y_c[~missing].mean():.4f}  (n={(~missing).sum():,})")
print(f"\nmissingness by age band:")
print(credit.assign(band=pd.cut(credit.age, [18, 28, 38, 50, 90]))
      .groupby("band", observed=True)["credit_score"].apply(lambda s: round(s.isna().mean(), 3)).to_string())

# %% [markdown]
# **Why A is a trap.** It does not merely "lose some data" — it **removes a
# subpopulation**, and a non-random one: thin-file applicants are younger and
# have fewer open accounts. The cross-validated score for A is computed on a
# different, easier population than the one the model will meet in production, so
# it is not comparable to B or C at all. Worse, at scoring time those applicants
# still arrive, and the model has never seen anyone like them.
#
# Deleting rows on a condition that correlates with the target is a *sampling*
# decision disguised as a *cleaning* decision. If a segment cannot be scored,
# that is a business rule to state explicitly, not a preprocessing step.
