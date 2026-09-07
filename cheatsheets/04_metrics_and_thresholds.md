# Cheat sheet 04 — Metrics, calibration and thresholds

## Choose the metric from the decision

1. Need a **probability** (pricing, provisioning, expected loss, a cost-derived
   threshold, feeding another model)? → log loss / Brier, and check calibration.
2. Need a **ranking** (a fixed-capacity review queue)? → average precision if
   imbalanced, ROC AUC if not.
3. Need a **decision** with known costs? → build the cost matrix and optimise
   expected cost directly.

## Classification metrics

| Metric | Answers | Use when | Blind to |
|---|---|---|---|
| `accuracy` | fraction correct | balanced, equal costs | imbalance — completely |
| `balanced_accuracy` | mean per-class recall | imbalanced, equal class importance | prevalence |
| `precision` | of those flagged, how many were right | FPs expensive | what you missed |
| `recall` | of the real positives, how many caught | FNs expensive | alert volume |
| `f1` | harmonic mean | one number, roughly equal costs | true negatives, asymmetric costs |
| `roc_auc` | P(random positive outranks random negative) | ranking, model comparison | prevalence, calibration |
| `average_precision` | area under PR | **imbalanced ranking** | calibration |
| `neg_log_loss` | probability quality (proper score) | you need probabilities | — |
| `neg_brier_score` | mean squared probability error | calibration + discrimination | — |
| `matthews_corrcoef` | correlation of prediction and truth | one balanced number | — |

**Accuracy on imbalanced data is meaningless.** "Predict no, forever" beats most
real models.

**ROC flatters imbalance; PR does not.** ROC's x-axis divides by the enormous
negative count, so hundreds of false alarms barely move it. Precision divides by
the alerts you actually raised — which is what the operations team works.

**Average precision is prevalence-dependent.** Do not compare AP across periods
with different base rates; use lift-over-base-rate or ROC AUC.

## Multiclass averaging

| `average=` | Behaviour | Use |
|---|---|---|
| `"micro"` | pools all decisions; = accuracy for single-label | overall throughput |
| `"macro"` | classes weighted equally | **when rare classes matter** |
| `"weighted"` | by support | most flattering, least informative |

Always show `classification_report` alongside.

## Regression metrics

| Metric | Penalises | Use when |
|---|---|---|
| `MAE` | linearly | outliers should not dominate; you want the conditional median |
| `RMSE` | quadratically | large errors disproportionately bad; conditional mean |
| `MAPE` | relative | scale varies; **breaks near zero** |
| `R²` | — | cross-dataset comparison; misleading on grouped data |
| pinball | asymmetrically | you are predicting a quantile |
| Poisson/Gamma/Tweedie deviance | by the assumed distribution | counts, costs, skewed non-negative targets |

**Train on the loss you will be judged by.** MAE and RMSE can rank two models
differently and neither is "right" — they are estimating different things.

## Thresholds

The default 0.5 assumes a false positive and a false negative cost the same.
For a **calibrated** model the cost-minimising threshold is:

```
t* = C_FP / (C_FP + C_FN)
```

```python
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer, confusion_matrix

def gain(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return -(fn * COST_FN + fp * COST_FP) / len(y_true)

tuned = TunedThresholdClassifierCV(estimator=pipe, scoring=make_scorer(gain),
                                   cv=5, refit=True).fit(X_train, y_train)
tuned.best_threshold_
```

Moving the threshold is routinely the highest-return hour in an engagement, and
it is routinely skipped. Choosing it on the test set is the same leak as
anything else chosen on the test set.

**When the base rate drifts, a quantile threshold ("refer the riskiest 12%") is
more robust than an absolute probability threshold** — ranking survives drift
that calibration does not.

## Calibration

**Ranking** = does it order correctly (AUC). **Calibration** = when it says
0.20, do 20% of them default. Independent properties: squaring all probabilities
leaves AUC untouched and destroys calibration.

Cheapest check there is:

```python
print(proba.mean(), y.mean())    # must be close, or nothing derived from the
                                 # probabilities is trustworthy
```

| Method | Fits | Needs | Note |
|---|---|---|---|
| `sigmoid` (Platt) | 2-parameter logistic | hundreds of rows | can't fix non-sigmoidal distortion |
| `isotonic` | any monotone step function | 1,000+ rows | overfits on small data |
| `temperature` (1.8+) | one scaling parameter | very little | best for multiclass |

```python
from sklearn.frozen import FrozenEstimator     # 1.6+; was cv="prefit"
CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_cal, y_cal)
```

Typical miscalibration: **random forests** compress towards the centre;
**SVMs and boosted models** are over-confident; **logistic regression is
calibrated by construction** on its training distribution.

Calibration is a monotone transform → **AUC is unchanged**. It cannot make a
model rank better.

## Imbalance: three levers, in the order that works

1. **Move the threshold.** Free, instant, leaves probabilities intact.
2. **`class_weight="balanced"`.** Use when the minority is small in *absolute*
   terms (a few hundred positives). Recalibrate afterwards — it distorts
   probabilities badly.
3. **Resampling (SMOTE, under-sampling).** Last. Rarely improves ranking, can
   halve it (under-sampling throws away the boundary), and always destroys
   calibration. Legitimate mainly for computational reasons.

**Imbalance is a decision-threshold problem, not a data problem.**

## Custom scorers

```python
from sklearn.metrics import make_scorer
scorer = make_scorer(my_fn, response_method="predict_proba", greater_is_better=True)

# or a raw callable, when you need the estimator itself:
def latency_aware(estimator, X, y): ...
```

If ranking by AUC and ranking by profit disagree, optimising AUC was selecting
the wrong model — and the AUC table looked fine.

## The one-standard-error rule

```python
def one_se(cv_results):
    s, sd = cv_results["mean_test_score"], cv_results["std_test_score"]
    best = s.argmax()
    eligible = np.flatnonzero(s >= s[best] - sd[best])
    complexity = np.array([p["clf__max_leaf_nodes"] for p in cv_results["params"]])
    return int(eligible[np.argmin(complexity[eligible])])

GridSearchCV(pipe, grid, refit=one_se, ...)
```

Among models that are statistically indistinguishable, ship the simplest.
