# Cheat sheet 03 — Validation and leakage

## Choosing a splitter — first "yes" wins

```
Ordered in time, and you will predict forward?
    └─ yes → TimeSeriesSplit  (or a manual expanding/rolling window)

Multiple rows per entity (customer, card, patient, device)?
    └─ yes → GroupKFold / StratifiedGroupKFold / GroupShuffleSplit

Imbalanced or multiclass target?
    └─ yes → StratifiedKFold

Otherwise
    └─ KFold(shuffle=True, random_state=…)
```

The question that decides it: **at inference time, will you have seen this
entity before? Will you be predicting forward in time?**

Both time *and* groups apply? sklearn has no splitter for that. Write one —
`split()` and `get_n_splits()` is the whole protocol.

> ⚠️ **`KFold` does not shuffle by default.** Exported tables are very often
> sorted by date, segment or target.

## The three CV functions

| Function | Returns | Use for |
|---|---|---|
| `cross_val_score` | one array | a quick number |
| `cross_validate` | multiple metrics, timings, train scores, fitted estimators | **default — costs nothing extra** |
| `cross_val_predict` | one out-of-fold prediction per row | diagnostics, calibration, stacking |

```python
cross_validate(est, X, y, cv=cv,
               scoring=["roc_auc", "average_precision", "neg_brier_score"],
               return_train_score=True, return_estimator=True, n_jobs=-1)
```

`cross_val_predict` is **not** a way to compute a score — the predictions come
from *k* different models on different scales. It also refuses non-partitioning
splitters like `TimeSeriesSplit`; write a walk-forward loop instead.

## The noise floor

```python
from sklearn.model_selection import RepeatedStratifiedKFold
scores = cross_val_score(est, X, y,
                         cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0))
print(f"{scores.mean():.4f} ± {scores.std():.4f}")
```

**Any difference smaller than ~2× that standard deviation is not a result.**
Report `mean ± std` and the number of splits. When two models are inside the
interval, choose on simplicity, latency or explainability — not on the mean.

## `best_score_` is biased upward

It is the maximum of many noisy estimates. On **pure noise** it rises steadily
with the size of the search. Defences, cheapest first:

1. A never-touched holdout, scored once at the end.
2. Nested CV — unbiased, `n_outer ×` the cost.
3. A smaller search.

## The leakage catalogue

Every one of these is "a decision made by looking at data that stands in for
the future".

| Leak | Symptom | Fix |
|---|---|---|
| Scaling/imputing before the split | small inflation; hides drift | put it in the `Pipeline` |
| Feature selection before CV | **huge** inflation, scales with candidate count | `SelectKBest` inside the pipeline |
| Target encoding on full data | inflation proportional to cardinality | sklearn's `TargetEncoder` cross-fits |
| Outlier removal before the split | optimistic | inside the pipeline, or before any split at all and documented |
| Resampling (SMOTE) before the split | large inflation | `imblearn.pipeline.Pipeline` |
| Threshold chosen on the test set | optimistic decisions | `TunedThresholdClassifierCV` |
| A post-outcome column | AUC 0.95+, single feature nearly as good as the model | data lineage review |
| Entity spans train and test | model memorises the entity | `GroupKFold` |
| Future rows in the training fold | quietly optimistic | `TimeSeriesSplit` |
| A sequential ID as a feature | encodes time | drop identifiers |
| Choosing the model family after seeing test scores | unquantifiable | holdout discipline |

## The leakage smell test

```python
from sklearn.metrics import roc_auc_score
for col in X.select_dtypes("number"):
    a = roc_auc_score(y, X[col].fillna(X[col].median()))
    print(f"{col:24s} {max(a, 1 - a):.3f}")
```

Any single feature approaching your whole model's score deserves a conversation
with whoever owns the source table **before** you ship anything.

## Learning and validation curves

| Curve | Question | Reading |
|---|---|---|
| learning (score vs train size) | *would more data help?* | converged and low = bias; wide gap still rising = data-limited; wide gap flat = regularise |
| validation (score vs one parameter) | *over- or under-fitting?* | peak location tells you which way to move |

A flat learning curve at 100% of your data is the honest answer to "should we
buy more data?"

## Discipline

- Decide the validation scheme **before** modelling and write it down.
- Split the holdout off first. Score it **once**, at the end.
- If you change the model after looking at the holdout, the holdout is spent.
- Fairness and calibration checks happen **before** the holdout is opened, since
  failing either means changing the model.
