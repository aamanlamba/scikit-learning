# Cheat sheet 01 — The scikit-learn API

## The four verbs

| Kind | Has | Example |
|---|---|---|
| Estimator | `fit(X, y=None)` | everything |
| Transformer | `+ transform(X)`, `fit_transform` | `StandardScaler` |
| Predictor | `+ predict(X)` | `LogisticRegression` |
| Model | `+ score(X, y)` | most predictors |

`KMeans` is all four. `LogisticRegression` has no `transform`, which is why it
can only be the last step of a pipeline.

## The underscore rule

```python
est.alpha          # hyperparameter — YOU set it, in __init__, unmodified
est.coef_          # learned parameter — fit() computed it
```

Everything depends on this split:

- `get_params()` / `set_params()` enumerate the no-underscore ones → grid search
- `clone(est)` copies only those → honest cross-validation
- `check_is_fitted(est)` looks for any trailing underscore

## Data shapes

| Object | Shape | Notes |
|---|---|---|
| `X` | `(n_samples, n_features)` | always 2-D; `x.reshape(-1, 1)` for one feature |
| `y` | `(n_samples,)` | 1-D for single output; `.ravel()` a column vector |
| single sample | `(1, n_features)` | `X.iloc[[0]]`, not `X.iloc[0]` |

Fit on a **DataFrame** to get `feature_names_in_` and
`get_feature_names_out()` in the artefact.

## Introspection

```python
est.get_params(deep=True)          # every tunable key, including nested
est.set_params(clf__C=0.1)         # set them; returns self
pipe.named_steps["clf"]            # by name
pipe[-1]                           # by index
pipe[:-1].transform(X)             # the intermediate matrix — debug with this
pipe[:-1].get_feature_names_out()  # what the final estimator actually sees
est                                # the HTML repr, in a notebook
```

**Parameter paths** read right to left, one `__` per level down:

```
prep__num__imp__strategy
 │     │    │    └── the `strategy` parameter
 │     │    └─────── of the step named "imp"
 │     └──────────── inside the branch named "num"
 └────────────────── inside the step named "prep"
```

Never guess a path. `sorted(pipe.get_params(deep=True))` tells you.

## Output types

```python
est.set_output(transform="pandas")           # per estimator
sklearn.set_config(transform_output="pandas") # globally
```

Requires `get_feature_names_out` on custom transformers. Free from
`TransformerMixin` otherwise.

## `random_state`

| Value | Behaviour | Use |
|---|---|---|
| `None` | different every run | never, in reported work |
| `42` (int) | fresh RNG on **every** `fit` — identical results | almost always |
| a `Generator` | consumes the shared stream — varies per fit, script replays | deliberate variation |

## Namespace map

| Module | Contains |
|---|---|
| `datasets` | `load_*` (bundled), `fetch_*` (downloaded), `make_*` (generated) |
| `preprocessing` | scalers, encoders, discretisers, splines, `FunctionTransformer` |
| `impute` | `SimpleImputer`, `KNNImputer`, `IterativeImputer`, `MissingIndicator` |
| `feature_extraction` | text and dict → features |
| `feature_selection` | `SelectKBest`, `RFE`, `SelectFromModel` |
| `decomposition` | `PCA`, `TruncatedSVD`, `NMF`, `KernelPCA` |
| `manifold` | `TSNE`, `Isomap`, `MDS`, `ClassicalMDS` — visualisation only |
| `compose` | `ColumnTransformer`, `TransformedTargetRegressor` |
| `pipeline` | `Pipeline`, `FeatureUnion`, `make_pipeline` |
| `model_selection` | splitters, CV, search, curves |
| `metrics` | scores, curves, `Display` classes, `pairwise` |
| `linear_model` | ~40 linear estimators |
| `ensemble` | forests, boosting, stacking, voting |
| `calibration` | `CalibratedClassifierCV` |
| `inspection` | `permutation_importance`, `PartialDependenceDisplay` |
| `base` | `BaseEstimator`, mixins, `clone` |
| `frozen` | `FrozenEstimator` (1.6+) |
| `utils` | `all_estimators`, `Bunch`, `estimator_checks`, `validate_data` |

## Discovery

```python
from sklearn.utils import all_estimators
all_estimators(type_filter="classifier")     # every classifier in the library
[n for n, _ in all_estimators("regressor") if "Quantile" in n]
```

## The `alpha` / `C` inversion

- **Regressors** take `alpha`: bigger = **more** regularisation.
- **Classifiers** take `C`: bigger = **less** regularisation. `C ≈ 1/alpha`.

## Writing your own estimator

```python
class MyThing(TransformerMixin, BaseEstimator):   # mixins FIRST, BaseEstimator LAST
    def __init__(self, alpha=1.0):
        self.alpha = alpha                # store unmodified; validate NOTHING

    def fit(self, X, y=None):
        if self.alpha < 0:                # validation belongs here
            raise ValueError(...)
        X = validate_data(self, X)        # sets n_features_in_, feature_names_in_
        self.something_ = ...             # trailing underscore
        return self                       # always

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return ...

    def get_feature_names_out(self, input_features=None): ...

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
```

Then: `check_estimator(MyThing())`.
