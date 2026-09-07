# Cheat sheet 06 — Recipes

Copy-paste starting points. Every one assumes `X` is a DataFrame.

## The default tabular pipeline

```python
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_prep():
    """A FRESH instance each call — Pipeline does not clone its steps."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), selector(dtype_include=[np.number, "bool"])),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                                  sparse_output=False)),
        ]), selector(dtype_include=["object", "string", "category"])),
    ], remainder="drop")

model = Pipeline([("prep", make_prep()),
                  ("clf", LogisticRegression(max_iter=4000, C=0.3))])
```

`bool` goes in the numeric branch: it is already 0/1, and mixing bool with
strings makes `SimpleImputer` refuse.

## Gradient boosting with no preprocessing at all

```python
from sklearn.ensemble import HistGradientBoostingClassifier

X_cat = X.copy()
for c in X_cat.select_dtypes(include=["object", "string"]).columns:
    X_cat[c] = X_cat[c].astype("category")

HistGradientBoostingClassifier(
    categorical_features="from_dtype",   # no encoding
    random_state=0,                      # NaN handled natively
    learning_rate=0.06, max_iter=400, early_stopping=True,
).fit(X_cat, y)
```

## Monotonic constraints (BFSI)

```python
MONOTONE = {"credit_score": -1, "debt_to_income": +1, "n_delinq_2yr": +1}

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
], verbose_feature_names_out=False).fit(X)

cst = [MONOTONE.get(n, 0) for n in prep.get_feature_names_out()]
HistGradientBoostingClassifier(monotonic_cst=cst, random_state=0)
```

A **logistic regression is monotone by construction** — verify with coefficient
signs instead:

```python
coef = pd.Series(model[-1].coef_.ravel(), index=model[:-1].get_feature_names_out())
{f: bool(np.sign(coef[n]) == d)
 for f, d in MONOTONE.items()
 for n in coef.index if n.endswith(f) and "missingindicator" not in n}
```

## Honest evaluation

```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

res = cross_validate(
    model, X, y,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0),
    scoring=["roc_auc", "average_precision", "neg_brier_score"],
    return_train_score=True, n_jobs=-1,
)
ap = res["test_average_precision"]
print(f"AP {ap.mean():.4f} ± {ap.std():.4f}  (train {res['train_average_precision'].mean():.4f})")
```

## Walk-forward out-of-fold predictions

`cross_val_predict` refuses `TimeSeriesSplit` (not a partition):

```python
def walk_forward_oof(est, X, y, cv):
    oof = np.full(len(X), np.nan)
    for tr, te in cv.split(X):
        oof[te] = clone(est).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return oof

oof = walk_forward_oof(model, X, y, TimeSeriesSplit(5))
mask = ~np.isnan(oof)
```

## Cost-optimal threshold

```python
from sklearn.metrics import confusion_matrix

def expected_cost(y_true, score, t, c_fn=6000, c_fp=1800):
    tn, fp, fn, tp = confusion_matrix(y_true, (score >= t).astype(int), labels=[0, 1]).ravel()
    return (fn * c_fn + fp * c_fp) / len(y_true)

ts = np.linspace(0.01, 0.6, 300)
costs = [expected_cost(y_val, p_val, t) for t in ts]
best_t = ts[int(np.argmin(costs))]
```

Do it properly with `TunedThresholdClassifierCV` so the threshold is
cross-validated rather than chosen on your test set.

## Randomised search that is worth running

```python
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

dist = {
    "clf__learning_rate": loguniform(0.01, 0.3),      # log scale for anything
    "clf__l2_regularization": loguniform(1e-4, 10),   # spanning magnitudes
    "clf__max_leaf_nodes": randint(4, 64),
    "clf__min_samples_leaf": randint(10, 300),
}
search = RandomizedSearchCV(pipe, dist, n_iter=40, cv=cv, scoring="average_precision",
                            random_state=0, n_jobs=-1, return_train_score=True).fit(X, y)

# Then: are the best values at the EDGE of your ranges?
for p in dist:
    v = pd.DataFrame(search.cv_results_)[f"param_{p}"].astype(float)
    b = float(search.best_params_[p])
    print(f"{p:30s} [{v.min():.4g}, {v.max():.4g}] best {b:.4g}"
          f"{'  <-- widen' if b <= v.min()*1.05 or b >= v.max()*0.95 else ''}")
```

## Searching over model families

```python
grid = [
    {"clf": [LogisticRegression(max_iter=4000)], "clf__C": [0.01, 0.1, 1],
     "select": ["passthrough", SelectKBest(f_classif, k=20)]},
    {"clf": [HistGradientBoostingClassifier(random_state=0)],
     "clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [8, 31]},
]
```

Compare families on the **median** score in each block, not the max.

## Resampling, correctly

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ImbPipeline([("prep", make_prep()), ("smote", SMOTE(random_state=0)),
             ("clf", LogisticRegression(max_iter=2000))])
```

Only `imblearn`'s pipeline resamples during `fit` and not during `predict`.
Recalibrate afterwards if you need probabilities.

## Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator      # 1.6+; before that, cv="prefit"

CalibratedClassifierCV(FrozenEstimator(fitted_model), method="sigmoid").fit(X_cal, y_cal)
CalibratedClassifierCV(base_estimator, method="isotonic", cv=5).fit(X, y)  # refits
```

## Grouped permutation importance

Single-feature importance is meaningless when features are correlated:

```python
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

corr = spearmanr(X[num_cols].fillna(X[num_cols].median())).correlation
corr = (corr + corr.T) / 2; np.fill_diagonal(corr, 1.0)
link = hierarchy.ward(squareform(np.clip(1 - np.abs(corr), 0, None)))
clusters = hierarchy.fcluster(link, 0.6, criterion="distance")
groups = {f"g{k}": list(v) for k, v in pd.Series(num_cols).groupby(clusters).apply(list).items()}
# then shuffle each group with the SAME permutation
```

## Population Stability Index

```python
def psi(expected, actual, n_bins=10, eps=1e-6):
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > n_bins:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True).sort_index()
        a = pd.cut(actual, edges).value_counts(normalize=True).sort_index()
    else:
        cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
        a = actual.astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
    e, a = e.to_numpy() + eps, a.to_numpy() + eps
    return float(((a - e) * np.log(a / e)).sum())
```

Bands: < 0.1 fine, 0.1–0.25 monitor, > 0.25 investigate.

## Adversarial validation (one-number drift test)

```python
combined = pd.concat([X_train.assign(__new=0), X_recent.assign(__new=1)], ignore_index=True)
target = combined.pop("__new")
auc = cross_val_score(Pipeline([("prep", make_prep()),
                                ("clf", HistGradientBoostingClassifier(random_state=0))]),
                      combined, target, cv=3, scoring="roc_auc").mean()
# 0.5 = indistinguishable;  > 0.7 = materially different populations
```

Permutation importance on the discriminator tells you *which* features moved.

## Persistence bundle

```python
import hashlib, json, joblib, platform, sklearn
from datetime import datetime, timezone

joblib.dump(model, out / "model.joblib", compress=3)
manifest = {
    "version": "1.0.0",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model_sha256": hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest(),
    "environment": {"python": platform.python_version(),
                    "scikit_learn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__},
    "feature_order": list(X_train.columns),
    "decision": {"threshold": best_t, "basis": "C_FP/(C_FP+C_FN)"},
    "schema": {...},
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))

# The reference batch — turns "did the deployment work?" into an assertion
ref = X_val.head(200)
ref.to_json(out / "reference_inputs.json", orient="table")   # round-trips dtypes
pd.DataFrame({"expected": model.predict_proba(ref)[:, 1]}).to_csv(
    out / "reference_predictions.csv", index=False)
```

Safer persistence across a trust boundary:

```python
import skops.io as sio
sio.dump(model, "model.skops")
untrusted = sio.get_untrusted_types(file="model.skops")   # inspect BEFORE loading
model = sio.load("model.skops", trusted=untrusted)
```

## A model card, minimally

Intended use · data (rows, features, base rate, period) · performance (ranking,
calibration, at the deployed threshold) · architecture · known limitations ·
fairness table · monitoring thresholds · retraining trigger.
