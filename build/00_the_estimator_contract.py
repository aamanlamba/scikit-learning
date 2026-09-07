# %% [markdown]
# # Module 00 — The Estimator Contract
#
# **The single most useful thing to know about scikit-learn is that it is not
# a collection of algorithms. It is one interface, implemented ~200 times.**
#
# Almost everyone who "knows sklearn from notebooks" knows a handful of
# algorithms and stitches them together by copying cells. Everyone who is
# genuinely fluent knows the *contract* — and can therefore use an estimator
# they have never seen before, correctly, on the first try, from the docstring
# alone.
#
# This module is about the contract.
#
# ---
#
# ### Learning objectives
#
# By the end you will be able to:
#
# 1. State the four verbs (`fit`, `transform`, `predict`, `score`) and the rules
#    about which objects have which.
# 2. Explain the trailing-underscore convention and why it is load-bearing, not
#    cosmetic.
# 3. Distinguish a *hyperparameter* (set in `__init__`, visible to
#    `get_params`) from a *learned parameter* (set in `fit`, trailing
#    underscore) — and explain why that distinction is what makes
#    `GridSearchCV` possible at all.
# 4. Read any scikit-learn docstring and predict the object's behaviour.
# 5. Navigate the library's namespace by *task* rather than by memory.
#
# ### How to work through this
#
# Run every cell. Then, before you read the output of a cell, guess it. The
# guessing is the learning; the running only confirms it.

# %%
# --- Standard preamble used by every notebook in this curriculum -----------
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sklearn

from skmastery import set_plot_style

set_plot_style()
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 40)

print(f"scikit-learn {sklearn.__version__}")
print(f"numpy        {np.__version__}")
print(f"pandas       {pd.__version__}")

# %% [markdown]
# ## 1. The four verbs
#
# Every object in scikit-learn is one of four kinds, defined entirely by which
# methods it has:
#
# | Kind | Has | Example | What it is for |
# |---|---|---|---|
# | **Estimator** | `fit(X, y=None)` | everything | learns something from data |
# | **Transformer** | `+ transform(X)` | `StandardScaler` | changes the feature matrix |
# | **Predictor** | `+ predict(X)` | `LogisticRegression` | produces an output per row |
# | **Model** | `+ score(X, y)` | most predictors | reports a default quality number |
#
# These compose. `PCA` is an estimator *and* a transformer.
# `LogisticRegression` is an estimator, a predictor and a model — but not a
# transformer. `KMeans` is unusual: it is all four (it can transform `X` into
# distances-to-centroids).
#
# That is the whole taxonomy. There is no fifth thing.

# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VERBS = ["fit", "transform", "fit_transform", "predict", "predict_proba", "score", "inverse_transform"]

survey = pd.DataFrame(
    {
        obj.__class__.__name__: [hasattr(obj, v) for v in VERBS]
        for obj in [StandardScaler(), PCA(), LogisticRegression(), KMeans(n_init="auto")]
    },
    index=VERBS,
)
survey

# %% [markdown]
# **Read that table carefully.** It answers questions people usually resolve by
# trial and error:
#
# - Why can't you put `LogisticRegression` in the middle of a `Pipeline`?
#   Because it has no `transform`. A pipeline is *n* transformers followed by at
#   most one predictor.
# - Why does `StandardScaler` have `inverse_transform` but `PCA`'s is lossy?
#   Because scaling is a bijection and projection is not.
# - Why does `KMeans` have both `predict` and `transform`? Because "which
#   cluster" and "how far from each centroid" are both useful outputs.

# %% [markdown]
# ## 2. The trailing underscore is a promise
#
# scikit-learn splits an object's state into two disjoint sets:
#
# - **Hyperparameters** — passed to `__init__`, stored *unmodified* on `self`
#   under the same name, no underscore. `alpha`, `n_estimators`, `C`.
# - **Learned parameters** — computed in `fit`, stored with a **trailing
#   underscore**. `coef_`, `mean_`, `classes_`, `n_features_in_`.
#
# The rule is enforced by the library's own test suite and it is not decoration.
# Three things depend on it:
#
# 1. `get_params()` / `set_params()` can enumerate everything tunable, which is
#    what lets `GridSearchCV` exist without knowing anything about the estimator.
# 2. `clone(est)` can produce an unfitted twin by copying only the
#    hyperparameters — which is what lets cross-validation avoid leaking a fit
#    from one fold into the next.
# 3. `check_is_fitted(est)` can tell fitted from unfitted by looking for *any*
#    trailing-underscore attribute.
#
# The corollary, which trips people up: **`__init__` must not validate or
# transform its arguments.** `StandardScaler(with_mean="yes")` will not complain
# until you call `fit`. That feels wrong until you realise `clone` depends on
# `init` being a pure assignment.

# %%
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

scaler = StandardScaler()

print("Hyperparameters (before fit):", scaler.get_params())
print("Trailing-underscore attrs   :", [a for a in vars(scaler) if a.endswith("_")])

try:
    check_is_fitted(scaler)
except NotFittedError as e:
    print("\ncheck_is_fitted ->", type(e).__name__)

X_demo = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 45.0]])
scaler.fit(X_demo)

print("\nAfter fit:", {a: np.round(v, 3) for a, v in vars(scaler).items() if a.endswith("_")})
check_is_fitted(scaler)
print("check_is_fitted -> passes")

# %%
# clone() copies hyperparameters ONLY. This is the mechanism that makes
# cross-validation honest: each fold gets a virgin estimator.
twin = clone(scaler)
print("twin params identical:", twin.get_params() == scaler.get_params())
print("twin is fitted       :", any(a.endswith("_") for a in vars(twin)))
print("twin is same object  :", twin is scaler)

# %% [markdown]
# ### Why this matters more than it looks
#
# A very common bug in hand-rolled ML code:
#
# ```python
# model = LogisticRegression()
# for train_idx, test_idx in folds:
#     model.fit(X[train_idx], y[train_idx])   # same object, refitted
#     ...
# ```
#
# With sklearn estimators this happens to be fine, because `fit` resets state.
# With a stateful custom estimator, or with `warm_start=True`, it silently is
# not. `cross_val_score` calls `clone` for exactly this reason — it never
# trusts `fit` to reset.

# %% [markdown]
# ## 3. `fit` / `transform` / `predict`: who is allowed to see what
#
# This is the rule that prevents most data leakage:
#
# > **`fit` may look at the training data. `transform` and `predict` may only
# > look at the row in front of them plus the learned parameters.**
#
# `StandardScaler.fit` computes `mean_` and `scale_` from the training set.
# `transform` then applies those *same numbers* to any data, including the test
# set. If you instead called `fit_transform` on the test set, the test set's own
# mean would leak into its representation and your estimate of generalisation
# would be optimistic.
#
# Below, the same idea shown numerically.

# %%
rng = np.random.default_rng(0)
X_train = rng.normal(loc=50, scale=10, size=(200, 1))
X_test = rng.normal(loc=58, scale=14, size=(50, 1))  # test set has drifted

sc = StandardScaler().fit(X_train)

correct = sc.transform(X_test)          # train statistics applied to test
wrong = StandardScaler().fit_transform(X_test)  # test statistics -- leakage

print(f"train mean/scale learned : {sc.mean_[0]:.2f} / {sc.scale_[0]:.2f}")
print(f"correct  -> test mean {correct.mean():+.3f}  std {correct.std():.3f}   (drift is visible)")
print(f"wrong    -> test mean {wrong.mean():+.3f}  std {wrong.std():.3f}   (drift erased)")

# %% [markdown]
# The "wrong" version produces a beautifully standardised test set — mean 0,
# std 1 — and in doing so **destroys the very evidence** that the population has
# shifted. The model looks fine in your notebook and degrades in production.
#
# > 💼 **Consulting lens.** When you review someone's modelling work, "did any
# > `fit` ever touch data that stands in for the future?" catches more real
# > defects than any question about model choice. Ask it first.

# %% [markdown]
# ## 4. Hyperparameters are introspectable — that is the whole trick
#
# `get_params(deep=True)` walks nested estimators and returns a flat dict keyed
# with `__`. That flat namespace is what you type into a grid search. Learn to
# *derive* those key names rather than guess them.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

demo_pipe = Pipeline(
    [
        (
            "prep",
            ColumnTransformer(
                [
                    ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), ["a", "b"]),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), ["c"]),
                ]
            ),
        ),
        ("clf", LogisticRegression()),
    ]
)

keys = sorted(demo_pipe.get_params(deep=True))
print(f"{len(keys)} tunable keys. The ones you would actually search:\n")
for k in keys:
    if k.endswith(("__C", "__strategy", "__with_mean", "__min_frequency", "__penalty")):
        print(" ", k)

# %% [markdown]
# Reading `prep__num__imp__strategy` right to left: the `strategy` of the step
# named `imp`, inside the pipeline registered as `num`, inside the step named
# `prep`. Every `__` is one level down. There is no magic and nothing to
# memorise — you can always recover the name with `get_params`.

# %%
# Setting them works the same way, and returns self so it chains.
demo_pipe.set_params(clf__C=0.05, prep__num__imp__strategy="median")
print(demo_pipe.get_params()["clf__C"], demo_pipe.get_params()["prep__num__imp__strategy"])

# %% [markdown]
# ## 5. The data contract: `X` and `y`
#
# - `X` is 2-D: `(n_samples, n_features)`. Always. A single feature is
#   `(n, 1)`, not `(n,)`. A single sample is `(1, n)`.
# - `y` is 1-D `(n_samples,)` for single-output problems, 2-D for multi-output.
# - Rows are samples, columns are features. Never the transpose. (This is the
#   opposite of the convention in a lot of statistics and econometrics
#   literature, which is why people coming from R trip on it.)
#
# scikit-learn will accept numpy arrays, pandas DataFrames, scipy sparse
# matrices, and — increasingly — anything implementing the Array API
# (PyTorch tensors, CuPy arrays) for a growing subset of estimators.

# %%
one_feature = np.array([1.0, 2.0, 3.0, 4.0])
print("shape as given:", one_feature.shape)

try:
    StandardScaler().fit(one_feature)
except ValueError as e:
    print("\nsklearn says:\n ", str(e).split("\n")[0])

print("\nfixed:", StandardScaler().fit(one_feature.reshape(-1, 1)).mean_)

# %% [markdown]
# ### Feature names survive if you feed it a DataFrame
#
# Since 1.0, estimators record `feature_names_in_` when fitted on a DataFrame,
# and transformers implement `get_feature_names_out()`. This is the difference
# between a model artefact you can audit and a wall of anonymous column indices.
# **Always fit on a DataFrame when you have one.**

# %%
df_demo = pd.DataFrame({"income": [40_000, 55_000, 90_000], "age": [28, 41, 37]})
sc2 = StandardScaler().fit(df_demo)

print("feature_names_in_ :", sc2.feature_names_in_)
print("get_feature_names_out():", sc2.get_feature_names_out())
print("n_features_in_    :", sc2.n_features_in_)

# scikit-learn *enforces* those names at transform time. Renaming a column is
# a hard error -- it means the caller is passing something other than what the
# estimator was fitted on.
try:
    sc2.transform(df_demo.rename(columns={"income": "salary"}))
except ValueError as e:
    print("\nRenamed column ->", str(e).split("\n")[0])

# Dropping the names entirely is only a warning: sklearn cannot tell whether
# you meant the same columns in the same order, so it lets it through noisily.
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sc2.transform(df_demo.to_numpy())
    print("Bare numpy    ->", caught[0].category.__name__ if caught else "no warning")

# %% [markdown]
# ### `set_output` — keep DataFrames all the way through
#
# By default transformers return numpy arrays, dropping column names. Since 1.2
# you can ask for pandas (or, since 1.4, polars) output. For exploratory and
# audit work this is worth switching on globally.

# %%
sc3 = StandardScaler().set_output(transform="pandas")
out = sc3.fit_transform(df_demo)
print(type(out).__name__)
out

# %%
# Globally, for a whole session:
from sklearn import set_config

set_config(transform_output="pandas")
print(type(StandardScaler().fit_transform(df_demo)).__name__)

set_config(transform_output="default")  # revert, so later modules show both

# %% [markdown]
# ## 6. Everything at once: a first honest model
#
# Here is the complete shape of a scikit-learn workflow. Nine lines. Read it as
# the template that the next fourteen modules unpack.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
model.fit(X_tr, y_tr)

print(f"train accuracy {model.score(X_tr, y_tr):.3f}")
print(f"test  accuracy {model.score(X_te, y_te):.3f}")

# %% [markdown]
# Four things in that cell are worth naming explicitly, because they recur:
#
# 1. **`stratify=y`** — keeps the class balance identical in both splits.
#    Omit it on an imbalanced problem and your test set's base rate wanders.
# 2. **`random_state=0`** — reproducibility. Any sklearn object that makes a
#    random choice takes this. Set it, always, and note it in your write-up.
# 3. **The `Pipeline`** — the scaler is fitted on `X_tr` only. Had you scaled
#    before splitting, you would have leaked.
# 4. **`.score()`** — every model has a default metric (accuracy for
#    classifiers, R² for regressors). It is a convenience, not a decision tool.
#    Module 08 is about why.

# %% [markdown]
# ## 7. Reading the library: navigate by task, not by memory
#
# scikit-learn's namespace is organised by *what you are doing*, and knowing the
# map beats knowing the names.
#
# | Module | What lives there |
# |---|---|
# | `sklearn.datasets` | toy + fetchable datasets, `make_*` generators |
# | `sklearn.preprocessing` | scalers, encoders, discretisers, transforms of X |
# | `sklearn.impute` | missing-value strategies |
# | `sklearn.feature_extraction` | raw text/dict → features |
# | `sklearn.feature_selection` | reducing the columns you keep |
# | `sklearn.decomposition` | PCA, NMF, dictionary learning |
# | `sklearn.manifold` | t-SNE, Isomap, MDS — visualisation, not features |
# | `sklearn.compose` | `ColumnTransformer`, `TransformedTargetRegressor` |
# | `sklearn.pipeline` | `Pipeline`, `FeatureUnion`, `make_pipeline` |
# | `sklearn.model_selection` | splitters, CV, search, curves |
# | `sklearn.metrics` | scores, curves, displays, pairwise distances |
# | `sklearn.linear_model` | ~40 linear estimators |
# | `sklearn.ensemble` | forests, boosting, stacking, voting |
# | `sklearn.calibration` | probability calibration |
# | `sklearn.inspection` | permutation importance, partial dependence |
# | `sklearn.base` | the mixins you inherit from to extend the library |
# | `sklearn.utils` | validation helpers, `Bunch`, estimator checks |
#
# Two habits worth forming now:

# %%
# (a) `all_estimators` enumerates the library by type. Useful for discovery.
from sklearn.utils import all_estimators

counts = pd.Series(
    {kind: len(all_estimators(type_filter=kind)) for kind in ["classifier", "regressor", "cluster", "transformer"]},
    name="n_estimators",
)
print(counts.to_string())

print("\nRegressors that are robust to outliers, by name:")
print([n for n, _ in all_estimators("regressor") if any(k in n for k in ("Huber", "RANSAC", "TheilSen", "Quantile"))])

# %%
# (b) The HTML repr. In a notebook this renders as an interactive diagram of a
# pipeline, with each estimator's hyperparameters and links to its docs.
# It is the fastest way to check that a composed object is wired the way you
# think it is.
demo_pipe

# %% [markdown]
# ## 8. What scikit-learn is *not* for
#
# Knowing the boundary is part of knowing the tool, and it is the part that
# matters most when you are advising rather than building.
#
# **Use scikit-learn for:** tabular data that fits in memory, classical
# supervised and unsupervised learning, feature engineering pipelines,
# rigorous evaluation and model selection, and as the surrounding scaffolding
# even when the model itself comes from elsewhere.
#
# **Reach past it for:**
#
# | Need | Where to go | Why sklearn stops |
# |---|---|---|
# | Deep learning, GPUs, backprop | PyTorch, JAX, Keras | no autodiff, no GPU training |
# | Best-in-class gradient boosting | XGBoost, LightGBM, CatBoost | `HistGradientBoosting` is close but has fewer knobs |
# | Data larger than RAM | Dask-ML, Spark MLlib, Polars + partial_fit | mostly in-memory |
# | Sequence / time-series models | statsmodels, sktime, Prophet | no ARIMA, no state space |
# | Probabilistic / Bayesian inference | PyMC, Stan, NumPyro | point estimates only |
# | Causal inference | DoWhy, EconML, CausalML | predictive, not causal |
# | Serving at low latency | ONNX Runtime, Triton, BentoML | Python-process inference |
#
# The important nuance: **these are not replacements, they are cores you wrap in
# scikit-learn's scaffolding.** XGBoost ships an sklearn-compatible API
# precisely so it can sit inside your `Pipeline` and be tuned by your
# `GridSearchCV`. The interface you are learning in this module outlives the
# algorithms you plug into it.
#
# > 💼 **Consulting lens.** "We use scikit-learn" and "we use XGBoost" are not
# > competing architectural statements — the first describes the harness, the
# > second the estimator. When a team presents them as a choice, that is usually
# > a sign the harness is missing.

# %% [markdown]
# ## 9. Reproducibility: `random_state` in one page
#
# Three behaviours, and the difference matters:
#
# - `random_state=None` (default) — draws from the global numpy RNG. Different
#   every run. Never use in anything you will report.
# - `random_state=42` (an int) — the estimator seeds a *fresh* RNG from that int
#   on **every call to fit**. Two fits give identical results. This is what you
#   want almost always.
# - `random_state=rng` (a `Generator`/`RandomState` instance) — the estimator
#   *consumes* from the shared stream, so consecutive fits differ, but the whole
#   script is reproducible end to end. Useful when you deliberately want
#   variability across an experiment while keeping the experiment repeatable.

# %%
from sklearn.ensemble import RandomForestClassifier

int_seeded = [RandomForestClassifier(n_estimators=5, random_state=7).fit(X_tr, y_tr).score(X_te, y_te) for _ in range(3)]

shared = np.random.RandomState(7)
stream_seeded = [RandomForestClassifier(n_estimators=5, random_state=shared).fit(X_tr, y_tr).score(X_te, y_te) for _ in range(3)]

print("int seed    :", np.round(int_seeded, 4), "-> identical")
print("shared RNG  :", np.round(stream_seeded, 4), "-> varies, but the script as a whole replays")

# %% [markdown]
# ---
# ## Exercises
#
# Work these in the empty cells below. Worked answers are in
# `solutions/00_the_estimator_contract_solutions.ipynb` — resist until you have
# a version that runs.

# %% [markdown]
# ### Exercise 0.1 — Classify the library
#
# Write a function `describe_object(est)` that returns a dict with keys
# `is_transformer`, `is_predictor`, `is_classifier`, `is_regressor`,
# `is_fitted`. Use `sklearn.base.is_classifier` / `is_regressor` and
# `check_is_fitted` rather than string matching on the class name.
#
# Run it over `StandardScaler()`, `PCA()`, `KMeans(n_init="auto")`,
# `LogisticRegression()`, `Ridge()` and a fitted `StandardScaler`, and present
# the result as a DataFrame.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 0.2 — Derive the parameter path
#
# Build this object:
#
# ```python
# pipe = Pipeline([
#     ("prep", ColumnTransformer([
#         ("num", Pipeline([("imp", SimpleImputer()), ("poly", PolynomialFeatures())]), ["a"]),
#         ("cat", OneHotEncoder(), ["b"]),
#     ])),
#     ("sel", SelectKBest()),
#     ("clf", RandomForestClassifier()),
# ])
# ```
#
# **Without running `get_params`**, write down the key you would use to tune:
# the polynomial degree; the number of features `SelectKBest` keeps; the forest's
# `max_depth`; the one-hot encoder's `min_frequency`. *Then* verify all four
# against `pipe.get_params(deep=True)`.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 0.3 — Prove that `clone` protects you
#
# Write a deliberately broken estimator: a class with `fit` that *accumulates*
# (e.g. appends each fit's row count to a list attribute) instead of resetting.
# Show that a naive fit-in-a-loop corrupts it across folds, and that routing the
# same loop through `clone` does not. Three or four lines of output is enough —
# the point is to see the mechanism, not to build anything reusable.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 0.4 — Quantify the leak
#
# Using `load_breast_cancer`, compare two workflows:
#
# - **A (leaky):** scale the *whole* `X` with `fit_transform`, then split, then
#   fit a `LogisticRegression`.
# - **B (clean):** split, then fit a `Pipeline` of scaler + logistic regression.
#
# Report test accuracy for both across 30 random splits and summarise the
# difference. Then explain, in two sentences, why the gap here is *small* — and
# what property of `StandardScaler` and of this dataset makes it so. (The point
# of the exercise is that "the numbers barely moved" is not evidence the
# practice is safe; think about which transformers would make it move a lot.)

# %%
# Your code here.


# %% [markdown]
# ### Exercise 0.5 — Navigate without Google
#
# Using only `all_estimators`, `dir()` and docstrings, find:
#
# 1. Every classifier that exposes `partial_fit` (i.e. supports out-of-core
#    learning).
# 2. Every transformer whose name contains `Encoder`, and one sentence on when
#    each is appropriate.
# 3. The estimator you would reach for to detect outliers in an unlabelled
#    tabular dataset — and the two others you rejected, with a reason.

# %%
# Your code here.


# %% [markdown]
# ---
# ## Takeaways
#
# - scikit-learn is **one interface implemented many times**. Learn the
#   interface and every estimator, including ones added after you stop reading,
#   is already familiar.
# - Trailing underscore = learned from data. No underscore = you chose it.
#   `clone` and `GridSearchCV` are built on that separation.
# - `fit` sees the training set; `transform`/`predict` see only their input plus
#   learned state. Every leakage bug is a violation of that sentence.
# - `get_params(deep=True)` is how you discover tuning keys — never guess them.
# - Fit on DataFrames so that names survive into the artefact, and use
#   `set_output(transform="pandas")` when you want to inspect intermediates.
#
# **Next:** Module 01 — how data actually gets into that `X`, and the dtype
# decisions that quietly determine which estimators you can use.
