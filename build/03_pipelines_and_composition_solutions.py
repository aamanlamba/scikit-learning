# %% [markdown]
# # Solutions — Module 03: Pipelines and Composition

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from skmastery import load_card_fraud, load_credit_risk, set_plot_style

set_plot_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)
rng = np.random.default_rng(0)

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

credit = load_credit_risk()
y = credit["default"]
X = credit.drop(columns=["default", "application_id", "collections_flag", "application_month"])
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]),
         make_column_selector(dtype_include=[np.number, "bool"])),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="__missing__")),
                          ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False))]),
         make_column_selector(dtype_include=["object", "string"])),
    ])


# %% [markdown]
# ## 3.1 — Trace the calls

# %%
CALLS = []


class Loud(TransformerMixin, BaseEstimator):
    def __init__(self, tag="step"):
        self.tag = tag

    def fit(self, X, y=None):
        CALLS.append({"step": self.tag, "method": "fit", "n_rows": np.shape(X)[0]})
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X):
        CALLS.append({"step": self.tag, "method": "transform", "n_rows": np.shape(X)[0]})
        return X


Xd = rng.normal(size=(300, 4))
yd = (Xd[:, 0] > 0).astype(int)

CALLS.clear()
pipe = Pipeline([("a", Loud("A")), ("b", Loud("B")), ("c", Loud("C")), ("clf", LogisticRegression())])
cross_val_score(pipe, Xd, yd, cv=3)

tally = pd.DataFrame(CALLS).groupby(["step", "method"]).agg(times=("n_rows", "size"), rows_each=("n_rows", "unique"))
tally

# %% [markdown]
# **The arithmetic.** With `cv=3` on 300 rows, each fold trains on 200 and tests
# on 100. Per step, per fold:
#
# - `fit` on 200 (once — `fit_transform` calls `fit` then `transform`)
# - `transform` on 200 (completing `fit_transform` during training)
# - `transform` on 100 (at scoring time)
#
# So **3 fits and 6 transforms per step** across 3 folds. Note that
# `TransformerMixin.fit_transform` calls `fit(X, y).transform(X)`, which is why
# each fit is followed by a transform of the same size.

# %%
print(f"expected: 3 fits/step, 6 transforms/step")
print(f"observed: {tally.xs('fit', level='method')['times'].to_dict()} fits, "
      f"{tally.xs('transform', level='method')['times'].to_dict()} transforms")

# %% [markdown]
# ## 3.2 — The `remainder` trap

# %%
X_with_id = credit.drop(columns=["default", "collections_flag", "application_month"])   # keeps application_id

leaky_prep = ColumnTransformer(
    [("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include=[np.number, "bool"]))],
    remainder="passthrough",
)
try:
    Pipeline([("prep", leaky_prep), ("clf", HistGradientBoostingClassifier(random_state=0))]).fit(X_with_id, y)
except Exception as e:
    print(f"passthrough with a string ID -> {type(e).__name__}: {str(e).splitlines()[0][:110]}")

# %%
# Now the version that does NOT raise, and is far worse for it.
X_numeric_id = X_with_id.copy()
X_numeric_id["application_id"] = X_numeric_id["application_id"].str[3:].astype(int)

num_only = X_numeric_id.select_dtypes(include=[np.number, "bool"])
with_id = cross_val_score(
    Pipeline([("i", SimpleImputer(strategy="median")), ("m", HistGradientBoostingClassifier(random_state=0, max_iter=250))]),
    num_only, y, cv=cv, scoring="roc_auc", n_jobs=-1)
without_id = cross_val_score(
    Pipeline([("i", SimpleImputer(strategy="median")), ("m", HistGradientBoostingClassifier(random_state=0, max_iter=250))]),
    num_only.drop(columns=["application_id"]), y, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"with numeric application_id    : {with_id.mean():.4f}")
print(f"without                        : {without_id.mean():.4f}")
print(f"difference                     : {with_id.mean() - without_id.mean():+.4f}")

# %% [markdown]
# **Nothing happened. Check why before concluding anything.**

# %%
corr = np.corrcoef(X_numeric_id["application_id"],
                   credit["application_month"].str.replace("-", "").astype(int))[0, 1]
print(f"corr(application_id, YYYYMM) = {corr:.4f}")

# %% [markdown]
# The correlation is ~0: in this generated dataset the ids were assigned before
# the months were drawn, so the id genuinely carries no information and the model
# correctly ignores it. **The absence of an effect here is not evidence that
# passthrough is safe** — it is evidence that this particular id is random.
#
# Real warehouse ids are almost never random. Build the realistic case: sort by
# date, then assign ids in order, exactly as an auto-increment primary key does.

# %%
ordered = credit.drop(columns=["collections_flag"]).sort_values("application_month").reset_index(drop=True)
ordered["seq_id"] = np.arange(len(ordered))            # an auto-increment key
num_ord = ordered.select_dtypes(include=[np.number, "bool"]).drop(columns=["default"])

print("default rate by seq_id decile — the id is a clock:")
print(ordered.assign(d=pd.qcut(ordered["seq_id"], 10, labels=False))
      .groupby("d")["default"].mean().round(4).to_string())

# %%
# Under RANDOM cross-validation the gain is small...
with_seq = cross_val_score(
    Pipeline([("i", SimpleImputer(strategy="median")), ("m", HistGradientBoostingClassifier(random_state=0, max_iter=250))]),
    num_ord, ordered["default"], cv=cv, scoring="roc_auc", n_jobs=-1)
without_seq = cross_val_score(
    Pipeline([("i", SimpleImputer(strategy="median")), ("m", HistGradientBoostingClassifier(random_state=0, max_iter=250))]),
    num_ord.drop(columns=["seq_id"]), ordered["default"], cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\nrandom 5-fold, with seq_id : {with_seq.mean():.4f}")
print(f"random 5-fold, without     : {without_seq.mean():.4f}")

# %%
# ...but the harm is out of time, which is where the model will actually live.
from sklearn.metrics import roc_auc_score

split = int(len(ordered) * 0.75)
rows = []
for cols, label in [(list(num_ord.columns), "with seq_id"), (list(num_ord.columns.drop("seq_id")), "without seq_id")]:
    m = Pipeline([("i", SimpleImputer(strategy="median")),
                  ("m", HistGradientBoostingClassifier(random_state=0, max_iter=250))]
                 ).fit(num_ord.iloc[:split][cols], ordered["default"][:split])
    p = m.predict_proba(num_ord.iloc[split:][cols])[:, 1]
    rows.append({"features": label,
                 "out_of_time_auc": round(roc_auc_score(ordered["default"][split:], p), 4),
                 "mean_predicted": round(p.mean(), 4),
                 "observed": round(ordered["default"][split:].mean(), 4)})
pd.DataFrame(rows).set_index("features")

# %% [markdown]
# **There it is.** Under random cross-validation the sequential id looks
# harmless — the folds are interleaved, so every test row has an id inside the
# training range. Trained on the first 75% and scored on the last 25%, it costs
# **0.024 ROC AUC**: every future id is beyond anything the tree fitted, so all of
# them fall down the single highest-risk branch and the model loses its ability
# to rank within them.
#
# Note the subtlety in `mean_predicted`: the id-using model is *better
# calibrated* out of time (0.157 vs 0.096 against an observed 0.171), because
# that extrapolated high-risk branch accidentally compensates for the base-rate
# shift. It got the level closer by ranking worse. That is exactly the kind of
# result that gets a bad feature kept, if you look at only one metric.
#
# `remainder="drop"` discards the column silently and safely.
# `remainder="passthrough"` feeds it in — and, on the version of the data you are
# most likely to have in production, only a temporal evaluation shows the cost.

# %% [markdown]
# ## 3.3 — Quantify five leaks

# %%
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import TargetEncoder

num_cols = make_column_selector(dtype_include=[np.number, "bool"])(X)
cat_cols = make_column_selector(dtype_include=["object", "string"])(X)
base_clf = LogisticRegression(max_iter=3000, C=0.3)


def cvs(est, X_, y_):
    return cross_val_score(est, X_, y_, cv=cv, scoring="roc_auc", n_jobs=-1).mean()


results = []

# 1. scaling before the split
X_num = X[num_cols].fillna(X[num_cols].median())
results.append({"leak": "1. scale before split",
                "leaky": cvs(base_clf, pd.DataFrame(StandardScaler().fit_transform(X_num), columns=num_cols), y),
                "clean": cvs(Pipeline([("s", StandardScaler()), ("c", base_clf)]), X_num, y)})

# 2. feature selection before the split
sel = SelectKBest(f_classif, k=6).fit(X_num, y)
results.append({"leak": "2. SelectKBest before split",
                "leaky": cvs(Pipeline([("s", StandardScaler()), ("c", base_clf)]), X_num.loc[:, sel.get_support()], y),
                "clean": cvs(Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif, k=6)), ("c", base_clf)]), X_num, y)})

# 3. target encoding on the full data
te = TargetEncoder(random_state=0)
X_te_leak = X.copy()
X_te_leak[cat_cols] = te.fit(X[cat_cols], y).transform(X[cat_cols])       # fit-then-transform: leaks
results.append({"leak": "3. TargetEncoder on full data",
                "leaky": cvs(Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", base_clf)]), X_te_leak, y),
                "clean": cvs(Pipeline([("p", ColumnTransformer([
                    ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_cols),
                    ("c", TargetEncoder(random_state=0), cat_cols)])), ("c", base_clf)]), X, y)})

# 4. outlier removal before the split
z = np.abs((X_num - X_num.mean()) / X_num.std())
keep = (z < 3).all(axis=1)
results.append({"leak": "4. 3-sigma trim before split",
                "leaky": cvs(Pipeline([("s", StandardScaler()), ("c", base_clf)]), X_num[keep], y[keep]),
                "clean": cvs(Pipeline([("s", StandardScaler()), ("c", base_clf)]), X_num, y)})

# 5. SMOTE before the split
X_res, y_res = SMOTE(random_state=0).fit_resample(X_num, y)
results.append({"leak": "5. SMOTE before split",
                "leaky": cvs(Pipeline([("s", StandardScaler()), ("c", base_clf)]), X_res, y_res),
                "clean": cvs(ImbPipeline([("s", StandardScaler()), ("sm", SMOTE(random_state=0)), ("c", base_clf)]), X_num, y)})

leaks = pd.DataFrame(results)
leaks["inflation"] = (leaks["leaky"] - leaks["clean"]).round(4)
leaks.round(4).sort_values("inflation", ascending=False).set_index("leak")

# %% [markdown]
# **Two of these show essentially zero inflation, and that is a finding about
# the dataset rather than about the leak.** With 11 genuinely informative
# numeric features and 12,000 rows, `SelectKBest(k=6)` picks the same six columns
# on every fold, so pre-selecting them changes nothing. The leak has no room to
# operate.
#
# Selection leakage needs **many candidates relative to rows**. Sweep both
# dimensions and watch it appear.

# %%
# Two conditions have to hold for selection leakage to bite:
#   (a) many candidates, so some correlate with y by chance, and
#   (b) few rows, so chance correlations are large.
# Sweep both.
small_cv = StratifiedKFold(4, shuffle=True, random_state=0)
wide_rows = []
for n_rows in [400, 12_000]:
    for n_noise in [0, 400, 3_000]:
        idx = rng.choice(len(X_num), n_rows, replace=False)
        Xi = X_num.iloc[idx].reset_index(drop=True)
        yi = y.iloc[idx].reset_index(drop=True)
        if n_noise:
            Xi = pd.concat([Xi, pd.DataFrame(rng.normal(size=(n_rows, n_noise)),
                                             columns=[f"noise_{i}" for i in range(n_noise)])], axis=1)
        k = min(6, Xi.shape[1])
        sel_i = SelectKBest(f_classif, k=k).fit(Xi, yi)
        leaky = cross_val_score(Pipeline([("s", StandardScaler()), ("c", base_clf)]),
                                Xi.loc[:, sel_i.get_support()], yi, cv=small_cv, scoring="roc_auc", n_jobs=-1).mean()
        clean = cross_val_score(Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif, k=k)), ("c", base_clf)]),
                                Xi, yi, cv=small_cv, scoring="roc_auc", n_jobs=-1).mean()
        wide_rows.append({"n_rows": n_rows, "n_noise_features": n_noise,
                          "leaky": round(leaky, 4), "clean": round(clean, 4),
                          "inflation": round(leaky - clean, 4)})
pd.DataFrame(wide_rows).set_index(["n_rows", "n_noise_features"])

# %% [markdown]
# **Is "largest gap" the same as "most dangerous"? No.**
#
# The 3-sigma trim (4) shows a *negative* number: removing extreme rows made the
# score worse, because the extremes were informative (very high DTI genuinely
# does predict default). It is still a leak — the population changed — but the
# lesson is that a leak does not have to *inflate* to invalidate a comparison.
# The trimmed number is measured on a different, easier population and is simply
# not comparable to the others.
#
# Rank them by *how much the gap can grow* and *how likely you are to miss it*:
#
# 1. **Feature selection (2)** is the most dangerous. Its inflation scales with
#    the number of candidates — Module 03 showed 0.5 → 0.82 AUC on **pure noise**
#    with 3,000 features and 300 rows. On a wide dataset it is catastrophic, and the workflow
#    ("screen first, then model the shortlist") is extremely common.
# 2. **Target encoding (3)** is next: inflation scales with cardinality, and the
#    difference between the safe and unsafe call is `fit_transform` versus
#    `fit(...).transform(...)`, which reads as a style choice.
# 3. **SMOTE (5)** inflates because synthetic points interpolate between rows
#    that then land in different folds — the test fold contains partial copies of
#    training rows.
# 4. **Outlier trimming (4)** is subtle: it does not just inflate the score, it
#    changes the *population*. The number is not comparable to the others at all.
# 5. **Scaling (1)** is the smallest, and the one everyone talks about.

# %% [markdown]
# ## 3.4 — Feature-name forensics

# %%
model = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=3000))]).fit(X, y)


def explain_feature(pipeline, name: str) -> dict:
    ct = pipeline.named_steps["prep"]
    out = {"output_name": name}

    branch_name = name.split("__")[0]
    out["branch"] = branch_name
    sub = dict(ct.named_transformers_).get(branch_name)
    if sub is None:
        return out | {"error": "no such branch"}

    cols = dict((n, c) for n, _, c in ct.transformers_).get(branch_name)
    cols = cols(pipeline.feature_names_in_) if callable(cols) else list(cols)
    out["branch_input_columns"] = len(cols)
    out["transformers_applied"] = [type(s).__name__ for _, s in sub.steps] if hasattr(sub, "steps") else [type(sub).__name__]

    rest = name.split("__", 1)[1]
    if "missingindicator_" in rest:
        src = rest.replace("missingindicator_", "")
        out |= {"source_column": src, "kind": "missingness indicator",
                "meaning": f"1 where {src} was NaN in the raw data"}
    elif branch_name == "cat":
        # OneHotEncoder names are "<column>_<category>"; the column is the longest
        # prefix that is an actual input column.
        src = max((c for c in cols if rest.startswith(c + "_")), key=len, default=None)
        cat = rest[len(src) + 1:] if src else None
        out |= {"source_column": src, "kind": "one-hot", "category": cat,
                "meaning": f"1 where {src} == {cat!r}" if src else "unresolved"}
        if cat == "infrequent_sklearn":
            enc = sub.named_steps["o"]
            idx = cols.index(src)
            rare = list(enc.infrequent_categories_[idx]) if enc.infrequent_categories_[idx] is not None else []
            out["meaning"] = f"1 where {src} is one of {len(rare)} rare levels: {rare[:5]}"
    else:
        out |= {"source_column": rest, "kind": "numeric",
                "meaning": f"{rest}, median-imputed then standardised"}
    return out


names = model[:-1].get_feature_names_out()
picks = [
    names[0],
    next(n for n in names if "missingindicator" in n),
    next(n for n in names if n.startswith("cat__purpose")),
    next((n for n in names if "infrequent" in n), next(n for n in names if n.startswith("cat__channel"))),
    next(n for n in names if n.startswith("cat__home_ownership")),
]
pd.DataFrame([explain_feature(model, n) for n in picks]).set_index("output_name")

# %% [markdown]
# ## 3.5 — Design a pipeline for the fraud data

# %%
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

fraud = load_card_fraud().sort_values("timestamp").reset_index(drop=True)

print("""
DESIGN DECISIONS

timestamp  -> DECOMPOSE, do not use raw.
   The raw timestamp is monotone and will be out of range for every future
   transaction. Extract hour-of-day and day-of-week (already present as
   columns), and encode hour CYCLICALLY -- 23:00 and 01:00 are adjacent, and a
   tree would need many splits to learn that a linear encoding hides.
   Keep the timestamp itself ONLY as the ordering key for the splitter.

card_id    -> DROP as an identity; use it as the GROUP key instead.
   ~9,000 cards over 60,000 transactions. As a feature it invites memorisation
   (Module 04 measures +34% AP inflation from letting cards span folds). As a
   group key it makes the evaluation honest. What we actually want are
   card-level aggregates computed from PAST transactions only -- velocity,
   deviation from the card's own median -- which the dataset already
   approximates with `amount_vs_card_median` and `n_txn_1h`.

scaling    -> NOT for the tree model; YES if you also fit a linear baseline.
   The primary model is HistGradientBoosting, which is threshold-based. But
   `amount` spans four orders of magnitude, so a log1p transform helps a linear
   challenger and costs the tree nothing.

resampling -> NO.
   0.7% positives is a threshold problem, not a data problem (Module 08). SMOTE
   would destroy calibration and here does not improve ranking. If training time
   became a constraint, under-sample the negatives for SPEED and then
   recalibrate -- a computational decision, not a statistical one.

metric     -> average precision, plus precision at a fixed daily alert volume.
""")

# %%
from sklearn.preprocessing import FunctionTransformer

fX = fraud.drop(columns=["is_fraud", "timestamp", "card_id"])
fy = fraud["is_fraud"]
groups = fraud["card_id"]


def cyclical_hour(df):
    h = df["hour"].to_numpy()
    return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]


fraud_prep = ColumnTransformer([
    ("hour_cyc", FunctionTransformer(cyclical_hour, feature_names_out=lambda s, i: ["hour_sin", "hour_cos"]), ["hour"]),
    ("log_amount", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["amount"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     make_column_selector(dtype_include=["object", "string"])),
    ("num", "passthrough", ["device_age_days", "country_mismatch", "n_txn_1h",
                            "amount_vs_card_median", "is_cnp", "day_of_week"]),
], remainder="drop")

fraud_model = Pipeline([("prep", fraud_prep),
                        ("clf", HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.06))])

grouped = cross_val_score(fraud_model, fX, fy, groups=groups,
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                          scoring="average_precision", n_jobs=-1)
print(f"grouped average precision : {grouped.mean():.4f} ± {grouped.std():.4f}")
print(f"base rate (no-skill AP)   : {fy.mean():.4f}")
print(f"lift over base rate       : {grouped.mean() / fy.mean():.1f}x")

# %% [markdown]
# The AP is low in absolute terms and that is the honest answer, not a failure
# of the pipeline: in this dataset most of the fraud sits on a minority of
# compromised cards whose risk is not observable from a single transaction. The
# way to move this number is **entity-level history features**, not a better
# classifier — and saying so is the useful output of the exercise.
