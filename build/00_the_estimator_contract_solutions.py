# %% [markdown]
# # Solutions — Module 00: The Estimator Contract

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from skmastery import set_plot_style

set_plot_style()
pd.set_option("display.width", 120)

# %% [markdown]
# ## 0.1 — Classify the library

# %%
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def describe_object(est) -> dict:
    try:
        check_is_fitted(est)
        fitted = True
    except NotFittedError:
        fitted = False
    return {
        "is_transformer": hasattr(est, "transform"),
        "is_predictor": hasattr(est, "predict"),
        "is_classifier": is_classifier(est),
        "is_regressor": is_regressor(est),
        "is_fitted": fitted,
    }


objects = {
    "StandardScaler": StandardScaler(),
    "PCA": PCA(),
    "KMeans": KMeans(n_init="auto"),
    "LogisticRegression": LogisticRegression(),
    "Ridge": Ridge(),
    "StandardScaler (fitted)": StandardScaler().fit(np.arange(20).reshape(-1, 2).astype(float)),
}
pd.DataFrame({k: describe_object(v) for k, v in objects.items()}).T

# %% [markdown]
# `is_classifier` / `is_regressor` read the estimator's tags, so they work for
# custom estimators too — unlike matching on the class name, which breaks the
# moment someone writes `MyGradientThing`.

# %% [markdown]
# ## 0.2 — Derive the parameter path

# %%
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer()), ("poly", PolynomialFeatures())]), ["a"]),
        ("cat", OneHotEncoder(), ["b"]),
    ])),
    ("sel", SelectKBest()),
    ("clf", RandomForestClassifier()),
])

# Predicted by reading the structure outside-in:
predicted = {
    "polynomial degree": "prep__num__poly__degree",
    "SelectKBest k": "sel__k",
    "forest max_depth": "clf__max_depth",
    "OHE min_frequency": "prep__cat__min_frequency",
}
available = set(pipe.get_params(deep=True))
pd.DataFrame([{"what": k, "path": v, "exists": v in available} for k, v in predicted.items()])

# %% [markdown]
# Every `__` is one level down, read left to right from the outermost step. If
# you ever doubt it, `sorted(pipe.get_params(deep=True))` is the ground truth.

# %% [markdown]
# ## 0.3 — Prove that `clone` protects you

# %%
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import KFold


class AccumulatingTransformer(TransformerMixin, BaseEstimator):
    """Deliberately broken: `fit` appends instead of resetting."""

    def __init__(self, tag="acc"):
        self.tag = tag

    def fit(self, X, y=None):
        if not hasattr(self, "seen_"):
            self.seen_ = []
        self.seen_.append(len(X))       # BUG: never reset
        return self

    def transform(self, X):
        return X


X = np.random.default_rng(0).normal(size=(90, 3))
cv = KFold(3)

naive = AccumulatingTransformer()
for tr, _ in cv.split(X):
    naive.fit(X[tr])                     # same object, refitted
print("fit-in-a-loop      ->", naive.seen_, "  (state accumulated across folds)")

template = AccumulatingTransformer()
per_fold = [clone(template).fit(X[tr]).seen_ for tr, _ in cv.split(X)]
print("clone-per-fold     ->", per_fold, "  (each fold independent)")
print("template untouched ->", hasattr(template, "seen_"))

# %% [markdown]
# This is why `cross_val_score` calls `clone` rather than trusting `fit` to
# reset. sklearn's own estimators do reset, but a custom one — or one with
# `warm_start=True` — may not.

# %% [markdown]
# ## 0.4 — Quantify the leak

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)

leaky, clean = [], []
for seed in range(30):
    # A: scale everything, then split
    X_all = StandardScaler().fit_transform(X_bc)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_all, y_bc, test_size=0.3, random_state=seed, stratify=y_bc)
    leaky.append(LogisticRegression(max_iter=5000).fit(Xa_tr, ya_tr).score(Xa_te, ya_te))

    # B: split, then a pipeline
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_bc, y_bc, test_size=0.3, random_state=seed, stratify=y_bc)
    pipe_b = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
    clean.append(pipe_b.fit(Xb_tr, yb_tr).score(Xb_te, yb_te))

leaky, clean = np.array(leaky), np.array(clean)
print(f"A (leaky) : {leaky.mean():.4f} ± {leaky.std():.4f}")
print(f"B (clean) : {clean.mean():.4f} ± {clean.std():.4f}")
print(f"mean difference: {leaky.mean() - clean.mean():+.5f}   (paired std {np.std(leaky - clean):.5f})")

# %% [markdown]
# **Why the gap is small here.** `StandardScaler` learns only two numbers per
# feature, and with 569 rows the test set's contribution to a mean and a standard
# deviation is a small perturbation of the training-set values. The leak is real
# but its magnitude is bounded by how much the statistic moves.
#
# **Where it would move a lot:** any transformer that learns something
# high-dimensional or target-aware — `SelectKBest`, `TargetEncoder`,
# `QuantileTransformer` with few samples per quantile, PCA on wide data, or
# anything fitted on 3,000 features and 300 rows (Module 03 measures that one:
# 0.5 → 0.75 AUC on pure noise).
#
# The lesson is not "scaling before splitting is fine". It is that **the size of
# a leak depends on how much the transformer can learn**, so a small measured gap
# on one transformer says nothing about the next one.

# %% [markdown]
# ## 0.5 — Navigate without Google

# %%
from sklearn.utils import all_estimators

# (1) out-of-core capable classifiers
partial_fit_clfs = sorted(n for n, c in all_estimators("classifier") if hasattr(c, "partial_fit"))
print("classifiers with partial_fit:")
for n in partial_fit_clfs:
    print("  ", n)

# %%
# (2) encoders
encoders = sorted(n for n, _ in all_estimators("transformer") if "Encoder" in n)
guidance = {
    "OneHotEncoder": "default for nominal categories; handle_unknown='ignore', min_frequency for cardinality",
    "OrdinalEncoder": "genuinely ordered levels, or any categorical feeding a tree model",
    "TargetEncoder": "high cardinality; cross-fitted internally so it is safe inside a pipeline",
    "LabelEncoder": "TARGETS ONLY — it takes 1-D input and does not belong in a feature pipeline",
    "LabelBinarizer": "targets; one-vs-rest label matrices",
    "MultiLabelBinarizer": "targets; a list of labels per sample",
}
pd.Series({e: guidance.get(e, "—") for e in encoders}, name="when to use").to_frame()

# %%
# (3) unlabelled outlier detection
outlier = sorted(n for n, c in all_estimators() if hasattr(c, "fit_predict") and "Outlier" in n or n in
                 {"IsolationForest", "OneClassSVM", "EllipticEnvelope", "SGDOneClassSVM"})
print("candidates:", outlier)
print("""
Choose  : IsolationForest
          Scales to large n, makes no distributional assumption, handles mixed
          feature scales, one interpretable parameter (contamination).

Rejected: EllipticEnvelope — assumes the data is a single Gaussian. On a mixed
          tabular table with skewed and categorical-derived features that is
          false, and it will flag whole legitimate subpopulations.

Rejected: OneClassSVM — O(n^2) to O(n^3) fit time, and needs gamma/nu tuned
          without labels to tune against. SGDOneClassSVM scales but inherits the
          tuning problem.
""")

# %% [markdown]
# `LabelEncoder` appearing in that list is worth noting: it is a *target*
# transformer, takes 1-D input, and is one of the most commonly misused objects
# in the library. Use `OrdinalEncoder` for features.
