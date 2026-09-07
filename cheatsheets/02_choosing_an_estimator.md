# Cheat sheet 02 — Choosing an estimator

## The honest default for tabular data

1. **`DummyClassifier` / `DummyRegressor`** — establish the floor.
2. **Regularised linear model** — `LogisticRegression` / `Ridge`. Fast,
   calibrated, monotone by construction, one coefficient per feature.
3. **`HistGradientBoosting*`** — the strong challenger. Native categoricals and
   NaN, monotonic constraints, sensible defaults.
4. Anything else only if 2 and 3 both fail to satisfy a stated requirement.

The linear model wins more often than people expect, especially when the
underlying process is roughly additive. Always run it, always report it.

## Does it need scaling?

Ask: **does the algorithm compute a distance, an inner product, or a penalty
over coefficients?**

| Needs scaling | Does not |
|---|---|
| k-NN, k-means, DBSCAN, SVM | decision trees |
| Ridge / Lasso / penalised logistic | random forest, extra trees |
| PCA, LDA, factor analysis | all gradient boosting |
| neural networks, anything SGD | Gaussian naive Bayes |
| `Normalizer` (row-wise; rarely what you want) | — |

## Scalers

| Scaler | Robust to outliers | Keeps sparsity | Output |
|---|---|---|---|
| `StandardScaler` | ✗ | ✗ | mean 0, unbounded |
| `MinMaxScaler` | ✗✗ worst | ✗ | [0, 1] |
| `RobustScaler` | ✓ | ✗ | unbounded |
| `MaxAbsScaler` | ✗ | **✓** | [−1, 1] |
| `QuantileTransformer` | **✓✓** | ✗ | uniform or normal |
| `PowerTransformer` | ✓ | ✗ | ≈ normal |
| `Normalizer` | n/a — **per row** | ✓ | unit-norm rows |

Heavy-tailed financial data → `RobustScaler` or `QuantileTransformer`, never
`MinMaxScaler`.

## Encoders

| Encoder | When | Watch out for |
|---|---|---|
| `OneHotEncoder(handle_unknown="ignore", min_frequency=…)` | the default | column explosion; use `min_frequency` |
| `OrdinalEncoder` | genuinely ordered levels, **or any categorical + trees** | nonsense under linear models |
| `TargetEncoder` | high cardinality | **must be cross-fitted** — sklearn's is |
| native categorical (`HistGB`, LightGBM, CatBoost) | often the best answer | check the estimator supports it |

`drop="first"` only with **unpenalised** models. With regularisation it changes
the penalty depending on which level you dropped.

## Linear models: the loss × penalty grid

| | none | L2 | L1 | L1+L2 |
|---|---|---|---|---|
| squared error | `LinearRegression` | `Ridge` | `Lasso` | `ElasticNet` |
| log loss | `LogisticRegression(penalty=None)` | `LogisticRegression` | `penalty="l1"` | `penalty="elasticnet"` |
| hinge | — | `LinearSVC` | — | — |
| Huber | — | `HuberRegressor` | — | — |
| Poisson/Gamma/Tweedie | — | `PoissonRegressor` etc. | — | — |
| pinball | — | — | `QuantileRegressor` | — |

`SGDClassifier` / `SGDRegressor` let you pick loss and penalty independently.

**L1's constraint region has corners on the axes → exact zeros → feature
selection. L2's circle does not.**

## Solvers for `LogisticRegression`

| Solver | Penalties | Sparse X | Best for |
|---|---|---|---|
| `lbfgs` (default) | L2, none | ✓ | general |
| `liblinear` | L1, L2 | ✓ | small data, binary, L1 |
| `newton-cholesky` | L2, none | ✗ | **n ≫ p** — very fast |
| `saga` | all | ✓ | large n, elastic net, sparse text |

Same answer, up to an order of magnitude difference in runtime.

## Match the loss to the target

| Target | Estimator |
|---|---|
| counts, ≥ 0 | `PoissonRegressor` |
| positive, right-skewed | `GammaRegressor` |
| ≥ 0 with a mass at zero (pure premium, claim cost) | `TweedieRegressor(power=1.5)` |
| outlier-contaminated | `HuberRegressor`, `RANSACRegressor` |
| a quantile / prediction interval | `QuantileRegressor`, `HistGradientBoostingRegressor(loss="quantile")` |
| skewed but you want the mean | `TransformedTargetRegressor(func=np.log1p, ...)` |

Squared error on a non-negative zero-inflated target **predicts negative
values**. That is a model-class failure, not a tuning problem.

## Trees and ensembles

**Bagging** (random forest): low-bias, high-variance learners, averaged →
variance falls. Hard to overfit, parallel, `n_estimators` is not a tuning
parameter — more is never worse.

**Boosting**: high-bias, low-variance learners, added sequentially → bias falls.
Overfits readily, needs a learning rate and early stopping, inherently
sequential. `n_estimators` **is** a tuning parameter.

Complexity knobs, best first: `min_samples_leaf`, `max_leaf_nodes`,
`ccp_alpha`, `max_depth`, `min_samples_split`.

## Gradient boosting libraries

| | sklearn `HistGB` | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| accuracy | within noise of the others on typical tabular data | | | often best on categorical-heavy |
| speed | fast | fast | **fastest** on wide data | slower |
| categoricals | native | native | native | **best** |
| GPU | ✗ | ✓ | ✓ | ✓ |
| monotonic constraints | ✓ | ✓ | ✓ | ✓ |
| extra dependency | **none** | ✓ | ✓ | ✓ |
| good defaults | **✓** | needs tuning | needs tuning | ✓ |

Start with sklearn's. Move for a stated reason: training time (LightGBM),
high-cardinality categoricals (CatBoost), existing ecosystem (XGBoost).

## Clustering: each algorithm is an assumption

| Algorithm | Assumes clusters are | Needs *k* | Noise label |
|---|---|---|---|
| `KMeans` | spherical, similar size and density | ✓ | ✗ |
| `GaussianMixture` | elliptical Gaussians | ✓ | soft |
| `DBSCAN` | dense regions, uniform density | ✗ | ✓ |
| `HDBSCAN` | dense regions, **varying** density | ✗ | ✓ |
| `AgglomerativeClustering` | hierarchical | ✓ | ✗ |
| `SpectralClustering` | connected in a similarity graph | ✓ | ✗ |

## Anomaly detection

| Estimator | Theory |
|---|---|
| `IsolationForest` | anomalies are easy to isolate — **default choice** |
| `LocalOutlierFactor` | lower local density than neighbours (`novelty=True` for new data) |
| `OneClassSVM` | outside a learned boundary; O(n²)+ |
| `EllipticEnvelope` | far in Mahalanobis distance from one Gaussian |

## Where scikit-learn stops

| Need | Go to |
|---|---|
| deep learning, GPUs | PyTorch, JAX |
| data larger than RAM | Dask-ML, Spark, `partial_fit` |
| time-series models | statsmodels, sktime |
| Bayesian inference | PyMC, NumPyro |
| causal inference | DoWhy, EconML |
| low-latency serving | ONNX Runtime, Triton |

These are **estimators inside your sklearn harness**, not replacements for it.
