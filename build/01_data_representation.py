# %% [markdown]
# # Module 01 — Data Representation
#
# Most scikit-learn errors are not modelling errors. They are **shape, dtype and
# type-of-target errors**, and they occur before any learning happens. This
# module makes those failure modes boring by making the representation explicit.
#
# ### Learning objectives
#
# 1. Use all three dataset families (`load_*`, `fetch_*`, `make_*`) and know when
#    each is the right choice.
# 2. Read a `Bunch` and know when to bypass it with `return_X_y` / `as_frame`.
# 3. Reason about dtypes: why `object` columns break estimators, when to use
#    pandas `category`, and what `float32` buys you.
# 4. Use `type_of_target` to diagnose "why does sklearn think this is a
#    regression problem".
# 5. Work with sparse matrices without accidentally densifying 40 GB.
# 6. Profile a new table quickly enough that you actually do it.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from skmastery import describe_frame, load_credit_risk, set_plot_style

set_plot_style()
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## 1. The three dataset families
#
# | Prefix | Where the data lives | Use it for |
# |---|---|---|
# | `load_*` | shipped inside the wheel | instant, offline, tiny — smoke tests and teaching |
# | `fetch_*` | downloaded and cached to `~/scikit_learn_data` | realistic size, needs network once |
# | `make_*` | generated on the fly from a known process | **you control the ground truth** |
#
# The third family is the one experienced practitioners use most and beginners
# use least. When you generate the data you know the true number of informative
# features, the true noise level, the true cluster count — so you can check
# whether a method *recovers* what you put in. That is how you build intuition
# about when a method works, which is not something a real dataset can tell you.

# %%
from sklearn.datasets import load_diabetes, make_classification, make_regression

# load_* : a Bunch, which is a dict that also allows attribute access.
bunch = load_diabetes()
print(type(bunch).__name__, "keys:", list(bunch.keys()))
print("data", bunch.data.shape, "| target", bunch.target.shape)
print("feature_names:", bunch.feature_names)
print("\nFirst 400 chars of DESCR:\n", bunch.DESCR[:400].strip())

# %%
# Three ways to unpack the same loader. Pick deliberately.
X_arr, y_arr = load_diabetes(return_X_y=True)                   # numpy, fastest
X_df, y_ser = load_diabetes(return_X_y=True, as_frame=True)     # pandas, named
frame = load_diabetes(as_frame=True).frame                      # one frame, X+y

print(type(X_arr).__name__, type(X_df).__name__, type(frame).__name__)
X_df.head(3)

# %% [markdown]
# ### `make_*`: ground truth you can check against
#
# `make_classification` builds a problem out of `n_informative` genuinely useful
# features, `n_redundant` linear combinations of those, and
# `n_repeated` + noise columns. Because you specified the split, you can ask
# whether your feature-selection method finds it.

# %%
X, y, true_coef = make_regression(
    n_samples=400,
    n_features=20,
    n_informative=5,     # only 5 features actually matter
    noise=12.0,
    coef=True,           # return the ground-truth coefficients
    random_state=0,
)

truth = pd.Series(true_coef).round(1)
print("Truly informative features:", truth[truth != 0].to_dict())

from sklearn.linear_model import LassoCV

recovered = pd.Series(LassoCV(cv=5, random_state=0).fit(X, y).coef_).round(1)
comparison = pd.DataFrame({"true": truth, "lasso": recovered})
comparison[(comparison.true != 0) | (comparison.lasso.abs() > 0.5)]

# %% [markdown]
# That is a genuinely useful habit: **before trusting a method on real data,
# check it recovers a signal you planted.** If Lasso cannot find five features
# out of twenty when you know they are there, it will not find them when you
# don't.

# %%
# Other generators worth knowing, and what each is for.
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    make_s_curve,
)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
for ax, (name, (Xg, yg)) in zip(
    axes,
    {
        "blobs\n(k-means works)": make_blobs(n_samples=300, centers=3, random_state=0),
        "moons\n(k-means fails)": make_moons(n_samples=300, noise=0.06, random_state=0),
        "circles\n(needs a kernel)": make_circles(n_samples=300, noise=0.04, factor=0.5, random_state=0),
        "anisotropic\n(scaling matters)": (
            make_blobs(n_samples=300, centers=3, random_state=0)[0] @ [[0.6, -0.7], [-0.4, 0.9]],
            make_blobs(n_samples=300, centers=3, random_state=0)[1],
        ),
    }.items(),
):
    ax.scatter(Xg[:, 0], Xg[:, 1], c=yg, s=8, cmap="viridis")
    ax.set_title(name, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Generated datasets exist to falsify your intuitions", y=1.05, fontsize=11)
plt.tight_layout()

# %% [markdown]
# ## 2. dtypes: where things quietly go wrong
#
# scikit-learn's numeric estimators ultimately want a float array. The path from
# your DataFrame to that array is where the errors live.

# %%
credit = load_credit_risk()
describe_frame(credit)

# %% [markdown]
# Four dtype situations, and what each means for you:
#
# | What you have | What sklearn does | What to do |
# |---|---|---|
# | `int64` / `float64` | uses it directly | nothing |
# | `bool` | casts to 0/1 | nothing |
# | `object` / `str` | **raises** on numeric estimators | encode it (Module 02) |
# | `category` | raises *unless* the estimator declares support | encode, or use a native-categorical estimator |
#
# That last row is the interesting one. `HistGradientBoostingClassifier`,
# LightGBM and CatBoost can consume categoricals natively — no one-hot, no
# ordinal codes, no cardinality explosion. That is often a better answer than
# encoding, and it is invisible if you only ever reach for `OneHotEncoder`.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

X = credit.drop(columns=["default", "application_id", "collections_flag"])
y = credit["default"]

try:
    LogisticRegression().fit(X, y)
except ValueError as e:
    print("LogisticRegression on raw mixed types ->", str(e).split("\n")[0][:110])

# HistGradientBoosting can be told to treat object/category columns natively.
X_cat = X.copy()
for col in X_cat.select_dtypes(include=["object", "string"]).columns:
    X_cat[col] = X_cat[col].astype("category")

hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)
hgb.fit(X_cat, y)
print("\nHistGradientBoosting with native categoricals -> fitted, no encoding at all")
print("categorical columns detected:", int(np.sum(hgb.is_categorical_)))

# %% [markdown]
# ### Memory: `float64` is the default and is usually twice what you need

# %%
big = np.random.default_rng(0).normal(size=(200_000, 60))
print(f"float64 : {big.nbytes / 1e6:6.1f} MB")
print(f"float32 : {big.astype(np.float32).nbytes / 1e6:6.1f} MB")

# Most sklearn estimators accept float32 and keep it (check `dtype` in the
# docstring's `check_array` call). Tree ensembles internally use float32 anyway.
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=5, random_state=0).fit(big[:2000].astype(np.float32), big[:2000, 0])
print("\nfitted on float32 without complaint:", rf.n_features_in_, "features")

# %% [markdown]
# ### pandas `category` for cardinality, not just tidiness

# %%
mem_object = credit["purpose"].memory_usage(deep=True)
mem_category = credit["purpose"].astype("category").memory_usage(deep=True)
print(f"purpose as string   : {mem_object / 1e3:7.1f} KB")
print(f"purpose as category : {mem_category / 1e3:7.1f} KB  ({mem_object / mem_category:.1f}x smaller)")

# The other benefit: the category list is *part of the dtype*, so an unseen
# level at scoring time is detectable rather than silently absent.
cat = pd.Categorical(credit["purpose"])
print("\nknown levels:", list(cat.categories))

# %% [markdown]
# ## 3. `type_of_target`: the function that ends the confusion
#
# "Why does `cross_val_score` want a regression metric?" is nearly always
# because sklearn inferred a different target type from the one you meant.
# `type_of_target` tells you what it inferred.

# %%
from sklearn.utils.multiclass import type_of_target

cases = {
    "binary ints": np.array([0, 1, 1, 0]),
    "binary floats": np.array([0.0, 1.0, 1.0, 0.0]),
    "binary strings": np.array(["good", "bad", "bad", "good"]),
    "multiclass": np.array([0, 1, 2, 1]),
    "multiclass-strings": np.array(["a", "b", "c", "b"]),
    "continuous": np.array([0.1, 0.7, 2.3, 1.1]),
    "continuous but few values": np.array([1.5, 2.5, 1.5, 2.5]),
    "multilabel": np.array([[1, 0, 1], [0, 1, 0]]),
    "multiclass-multioutput": np.array([[1, 2], [0, 3]]),
    "column vector (a common bug)": np.array([[0], [1], [1], [0]]),
}
pd.Series({k: type_of_target(v) for k, v in cases.items()}, name="type_of_target").to_frame()

# %% [markdown]
# Two rows deserve attention.
#
# - **`continuous but few values` → `multiclass`.** A target of `[1.5, 2.5,
#   1.5, 2.5]` is inferred as classification because it has few distinct
#   values and no fractional pattern sklearn recognises as continuous. If your
#   regression target is coarsely rounded, say so explicitly rather than letting
#   inference decide.
# - **`column vector` → `multilabel-indicator`** on some shapes and triggers a
#   `DataConversionWarning` on others. `y` should be 1-D. `y.ravel()` or
#   `y.squeeze()` before you fit.

# %%
from sklearn.linear_model import LinearRegression
import warnings

y_2d = np.array([[0], [1], [1], [0]])
X_small = np.arange(8).reshape(4, 2).astype(float)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    LinearRegression().fit(X_small, y_2d)
    print("fit with 2-D y:", [w.category.__name__ for w in caught] or "no warning (silently multi-output)")
print("shape of coef_ :", LinearRegression().fit(X_small, y_2d).coef_.shape, "<- 2-D, i.e. multi-output")
print("shape with 1-D :", LinearRegression().fit(X_small, y_2d.ravel()).coef_.shape)

# %% [markdown]
# ## 4. Sparse matrices
#
# Text features, one-hot encodings of high-cardinality columns, and interaction
# expansions all produce matrices that are mostly zeros. Storing them densely is
# the single most common way to run out of memory in a modelling script.

# %%
from sklearn.preprocessing import OneHotEncoder

high_card = pd.DataFrame({"merchant": [f"M{i % 5000:05d}" for i in range(120_000)]})

dense = OneHotEncoder(sparse_output=False).fit_transform(high_card[:5_000])
sparse = OneHotEncoder(sparse_output=True).fit_transform(high_card)

print(f"dense  (5k rows × 5k cols) : {dense.nbytes / 1e6:8.1f} MB")
print(f"sparse (120k rows × 5k cols): {(sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes) / 1e6:8.1f} MB")
print(f"\nfull dense equivalent would be : {120_000 * 5_000 * 8 / 1e9:.1f} GB")
print("density:", f"{sparse.nnz / (sparse.shape[0] * sparse.shape[1]):.2e}")

# %% [markdown]
# ### Which estimators accept sparse input
#
# There is no universal rule, so check rather than assume. The docstring of
# `fit` states `X : {array-like, sparse matrix}` when sparse is supported.
# As a working heuristic:
#
# - **Sparse-friendly:** linear models with `saga`/`liblinear`, `SGDClassifier`,
#   `MultinomialNB`, `TruncatedSVD`, `LinearSVC`, most of `feature_selection`.
# - **Dense-only:** `HistGradientBoosting*`, `PCA` (use `TruncatedSVD`),
#   `GaussianNB`, `QuadraticDiscriminantAnalysis`, most of `sklearn.manifold`.
#
# The dense-only list is why `sparse_output=False` on your `OneHotEncoder` is
# sometimes required — and why you then need `min_frequency`/`max_categories` to
# stop the column count exploding.

# %%
from sklearn.decomposition import PCA, TruncatedSVD

try:
    PCA(n_components=2).fit(sparse)
except TypeError as e:
    print("PCA on sparse ->", str(e).split("\n")[0][:100])

svd = TruncatedSVD(n_components=2, random_state=0).fit(sparse)
print("TruncatedSVD on sparse -> OK, explained variance:", svd.explained_variance_ratio_.round(4))

# %% [markdown]
# ## 5. Missing values: representation, not strategy
#
# Strategy is Module 02. Representation is here, because getting it wrong
# produces silent nonsense rather than an error.
#
# - `np.nan` is the canonical missing marker. It is a **float**, so a column
#   with any missing value cannot be `int64` (unless you use pandas' nullable
#   `Int64`).
# - `None` in an `object` column is *not* `np.nan` and comparisons behave
#   differently.
# - Sentinel values (`-999`, `-1`, `9999`, empty string, `"NA"`, `"unknown"`)
#   are the real-world hazard: they are numerically valid, so nothing raises,
#   and your model happily learns that income of −999 predicts default.

# %%
dirty = pd.DataFrame(
    {
        "income": [45_000, -999, 82_000, 9999999, np.nan],
        "score": [710, 640, -1, 800, 690],
        "status": ["active", "", "active", "unknown", None],
    }
)

print("pandas thinks this is missing:")
print(dirty.isna().sum().to_string())

print("\nBut look at the distributions:")
print(dirty[["income", "score"]].describe().loc[["min", "max"]].to_string())

# %%
# A cheap, effective sentinel sniffer to run on every new table.
def sniff_sentinels(df: pd.DataFrame, suspects=(-1, -999, -9999, 0, 999, 9999, 99999)) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes("number").columns:
        s = df[col]
        for v in suspects:
            n = int((s == v).sum())
            if n and n / len(s) > 0.005:
                rows.append({"column": col, "value": v, "count": n, "share": round(n / len(s), 4)})
    # also flag values far outside the bulk of the distribution
    for col in df.select_dtypes("number").columns:
        s = df[col].dropna()
        if len(s) > 10:
            q1, q99 = s.quantile([0.01, 0.99])
            extreme = int(((s < q1 - 10 * (q99 - q1)) | (s > q99 + 10 * (q99 - q1))).sum())
            if extreme:
                rows.append({"column": col, "value": "far-outlier", "count": extreme, "share": round(extreme / len(s), 4)})
    return pd.DataFrame(rows)


sniff_sentinels(dirty)

# %%
# On the (clean-ish) credit data it should find little -- which is the point:
# the function is worth running precisely because it usually returns nothing.
sniff_sentinels(credit)

# %% [markdown]
# > 💼 **Consulting lens.** Sentinel encoding is a *data contract* failure, not a
# > modelling failure, and the fix belongs upstream. When you find one, the
# > useful output of the engagement is not the imputer you wrote — it is the
# > note to the source-system owner. Models built on undocumented sentinels are
# > the most common cause of "it worked in the POC and not in production".

# %% [markdown]
# ## 6. Which estimators tolerate NaN natively
#
# Since 1.1 a growing set handles missing values without an imputer, by learning
# a default direction at each split. Knowing the list saves you from imputing
# where imputation destroys information.

# %%
from sklearn.utils import all_estimators
from sklearn.utils.validation import check_array

def handles_nan(cls) -> bool:
    try:
        est = cls()
        return bool(est.__sklearn_tags__().input_tags.allow_nan)
    except Exception:
        return False


nan_ok = sorted(
    name
    for name, cls in all_estimators()
    if name.startswith(("Hist", "Random", "Extra", "Simple", "Iterative", "KNN", "Decision", "Bagging", "MissingInd", "Quantile"))
    and handles_nan(cls)
)
print("Estimators that accept NaN directly (subset):")
for n in nan_ok:
    print("  ", n)

# %%
# Demonstrated: HistGradientBoosting on the raw credit data, missing values and all.
from sklearn.model_selection import cross_val_score

X_nan = X_cat  # still has NaN in employment_years / annual_income / credit_score
print("NaN present:", int(X_nan.isna().sum().sum()))
scores = cross_val_score(
    HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0),
    X_nan, y, cv=3, scoring="roc_auc",
)
print(f"ROC AUC without any imputation: {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# ## 7. A profiling routine worth keeping
#
# The reason people skip data profiling is that it takes twenty cells. Here it
# takes one. Steal this.

# %%
def profile(df: pd.DataFrame, target: str | None = None) -> None:
    n, p = df.shape
    print(f"shape: {n:,} rows × {p} columns    memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    dupes = int(df.duplicated().sum())
    print(f"duplicate rows: {dupes:,}")
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const:
        print(f"constant columns: {const}")
    ids = [c for c in df.columns if df[c].nunique() == n and n > 20]
    if ids:
        print(f"candidate identifiers (drop before modelling): {ids}")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        print("\nmissingness:")
        print((miss * 100).round(2).to_string())
    if target and target in df:
        t = df[target]
        print(f"\ntarget '{target}': type_of_target={type_of_target(t)}")
        if type_of_target(t) in {"binary", "multiclass"}:
            vc = t.value_counts(normalize=True).round(4)
            print(vc.to_string())
            print(f"imbalance ratio: {vc.max() / vc.min():.1f} : 1")
        else:
            print(t.describe().round(2).to_string())


profile(credit, target="default")

# %%
# Missingness co-occurrence: is it random, or structured?
miss_mat = credit.isna()
cols = miss_mat.columns[miss_mat.any()]
print("Missingness by employment_type (is it MAR or MNAR?):")
print(
    credit.groupby("employment_type")[list(cols)]
    .apply(lambda g: g.isna().mean())
    .round(3)
    .to_string()
)

# %% [markdown]
# `employment_years` is missing for ~60% of the self-employed and ~3% of
# everyone else. That is **missing not at random**: the missingness is itself
# informative. Imputing the median and moving on throws away a signal. The right
# response is usually to impute *and* add a `MissingIndicator` — which is
# exactly what `SimpleImputer(add_indicator=True)` does, and the subject of
# Module 02.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1.1 — Recover a planted signal
#
# Generate a classification problem with `make_classification`:
# 2,000 samples, 40 features, 6 informative, 6 redundant, `class_sep=0.8`,
# `flip_y=0.03`, `random_state=0`. Then:
#
# 1. Fit an `ExtraTreesClassifier` and rank features by importance.
# 2. Fit a `LogisticRegression(penalty="l1", solver="saga")` and rank by
#    `|coef|`.
# 3. `make_classification` places the informative features first, so the truth
#    is "indices 0–5 informative, 6–11 redundant". How many of the top 6 from
#    each method are genuinely informative? Which method is fooled by the
#    redundant features, and why?

# %%
# Your code here.


# %% [markdown]
# ### Exercise 1.2 — Diagnose five broken targets
#
# For each of the following, use `type_of_target` and a one-line fit attempt to
# say what sklearn will do and how to fix it:
#
# ```python
# y1 = pd.Series([0, 1, 1, 0]).to_frame()          # DataFrame, not Series
# y2 = np.array(["1", "0", "1", "0"])              # numeric-looking strings
# y3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) / 1.0   # float class ids
# y4 = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])  # ambiguous
# y5 = pd.Series([0, 1, 1, np.nan])                # NaN in the target
# ```

# %%
# Your code here.


# %% [markdown]
# ### Exercise 1.3 — Sparse budget
#
# The `card_fraud` dataset has a `card_id` column with ~9,000 distinct values.
#
# 1. Compute the dense memory footprint of one-hot encoding it for all 60,000
#    rows, in GB. Then compute the sparse footprint.
# 2. One-hot encode it sparsely, hstack it with the numeric columns using
#    `scipy.sparse.hstack`, and fit an `SGDClassifier(loss="log_loss")`.
# 3. Now do the same with `min_frequency=50` on the encoder. How many columns
#    survive, and what happened to the AUC? Write two sentences on whether
#    `card_id` should be a feature at all.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 1.4 — Build a data-quality report
#
# Extend the `profile` function above into `data_quality_report(df, target)`
# that additionally reports, per column: cardinality ratio (`nunique / n`),
# whether the column is near-constant (top value > 95%), skewness for numerics,
# and a flag for suspected sentinels. Return a DataFrame sorted so the most
# problematic columns are at the top. Run it on all five bundled datasets and
# note the worst issue in each.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 1.5 — MNAR in practice
#
# Using the credit data, compare three treatments of `credit_score`:
#
# - **A:** drop rows where it is missing
# - **B:** `SimpleImputer(strategy="median")`
# - **C:** `SimpleImputer(strategy="median", add_indicator=True)`
#
# Score each with 5-fold `roc_auc` using a `HistGradientBoostingClassifier`.
# Then compute the default rate among rows where `credit_score` is missing
# versus present, and explain the ranking you observed. (Treatment A is a trap —
# say precisely why.)

# %%
# Your code here.


# %% [markdown]
# ---
# ## Takeaways
#
# - `make_*` generators are for **falsifying your own intuitions**; use them
#   before trusting a method on real data.
# - The dtype of a column determines which estimators can see it. Native
#   categorical support (`HistGradientBoosting`, LightGBM, CatBoost) is often
#   better than encoding, and is routinely overlooked.
# - `type_of_target` resolves nearly every "sklearn misunderstood my labels"
#   problem in one line.
# - Sparse is not an optimisation, it is a requirement above a few thousand
#   one-hot columns — and it constrains which estimators you can use downstream.
# - Sentinel-coded missing values are silent and dangerous. Sniff for them.
# - Missingness that is structured (MNAR) is a feature, not a nuisance.
#
# **Next:** Module 02 — transformers: scaling, encoding and imputation, and how
# each one changes what your model can learn.
