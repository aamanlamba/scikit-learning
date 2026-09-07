# Cheat sheet 07 — Performance, GPU and Apple Silicon

## The effort/reward ordering for tabular ML

Work down this list. Stop when it's fast enough.

| # | Change | Effort | Typical gain |
|---|---|---|---|
| 1 | Fix nested thread pools (`n_jobs=1` on inner estimators) | minutes | **up to 40×** |
| 2 | `HistGradientBoosting` instead of `GradientBoosting`/`RandomForest` | minutes | 10–100× |
| 3 | Subsample during development | free | 10× |
| 4 | `Pipeline(memory=...)` to cache preprocessing in a search | minutes | 2–5× |
| 5 | Shrink the search budget (Module 07) | free | 4× |
| 6 | `float32` instead of `float64` where precision allows | minutes | ~2× memory |
| 7 | Sparse matrices for one-hot / text | minutes | 10–1000× memory |
| 8 | `newton-cholesky` solver when n ≫ p | seconds | up to 10× |
| 9 | Move dense linear algebra to the GPU | a day | 1–3×, on 13 estimators |

**Number 1 is the one people miss and the largest single win.**

## Threading

```python
from threadpoolctl import threadpool_info, threadpool_limits
threadpool_info()                                  # what pools exist, how many threads

with threadpool_limits(limits=1, user_api="blas"): # escape hatch for libs without n_jobs
    ...
```

Rules:

1. **Parallelise at exactly one level.** Outer `n_jobs=-1` → inner `n_jobs=1`.
   Both at `-1` spawns *n_cores²* threads for *n_cores* cores.
2. **The boosting libraries default to all cores.** `LGBMClassifier`,
   `XGBClassifier` and `CatBoostClassifier` all use every core unless told
   otherwise. Inside a parallel search, set `n_jobs=1` on the estimator.
3. **Prefer parallelising the outer loop.** CV folds are embarrassingly parallel
   and equally sized; tree building is neither.
4. **On Apple Silicon, `-1` is not always best.** `os.cpu_count()` counts
   efficiency cores too, and a parallel loop runs at the speed of its slowest
   worker. Try `n_jobs` = performance-core count:

   ```bash
   sysctl hw.perflevel0.logicalcpu   # P-cores
   sysctl hw.perflevel1.logicalcpu   # E-cores
   ```

Environment variables (set **before** importing numpy):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## GPU: what is actually possible

**scikit-learn has no GPU backend.** It has Array API dispatch — hand an
estimator a PyTorch tensor and, if that estimator opts in, the maths happens
wherever the tensor lives.

```python
import os
os.environ["SCIPY_ARRAY_API"] = "1"      # BEFORE importing scipy or sklearn

import torch
from sklearn import config_context
from sklearn.decomposition import PCA

Xt = torch.asarray(X.astype("float32"), device="mps")   # or "cuda"
with config_context(array_api_dispatch=True):
    Z = PCA(n_components=50).fit_transform(Xt)          # returns a torch tensor
```

Forgetting the env var gives a `RuntimeError` when you enable dispatch, not at
import. If sklearn is already imported, restart the kernel.

**The ~13 estimators that dispatch** (sklearn 1.8): `Binarizer`, `GaussianNB`,
`KernelCenterer`, `LabelEncoder`, `LinearDiscriminantAnalysis`, `MinMaxScaler`,
`Normalizer`, `PCA`, `PolynomialFeatures`, `Ridge`, `RidgeCV`,
`RidgeClassifierCV`, `StandardScaler`. Plus many `sklearn.metrics` functions.

Check for yourself — the list grows each release:

```python
[n for n, c in all_estimators() if c().__sklearn_tags__().array_api_support]
```

**Nothing tree-based is on that list, and nothing will be.** Trees have
data-dependent branching, irregular memory access, and sequential dependence
between boosting rounds — all three are fatal on a GPU.

## Benchmarking a GPU honestly

```python
def sync():
    torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()

fn(); sync()                    # warm up — the first call compiles kernels
t0 = perf_counter(); fn(); sync(); elapsed = perf_counter() - t0
```

Without the sync you time how fast you can *queue* work. That is where fake
100× speed-ups come from.

There is always a **crossover size** below which the GPU loses to fixed
overhead. Typical tabular problems sit below it.

## Apple Silicon specifics

| | Detail |
|---|---|
| Backend | MPS (Metal Performance Shaders), `device="mps"` |
| **float64** | **Not supported.** `torch.asarray(x_f64, device="mps")` raises. Cast to float32 explicitly. |
| Memory | Unified — `.to("mps")` doesn't cross a PCIe bus, so transfers are cheap and the crossover sits lower than on CUDA |
| XGBoost GPU | ❌ CUDA only, no Metal build |
| LightGBM GPU | ❌ OpenCL, effectively unavailable on macOS |
| CatBoost GPU | ❌ CUDA only |
| PyTorch / MLX | ✅ this is the real Apple Silicon story |

**The float32 constraint is not only a performance question.** On an
ill-conditioned design (collinear features, Module 05) single precision gave
~75× the coefficient error of float64 at a condition number of only 2e4. If
`alpha` is near zero, moving to the GPU can change your answer. Regularise, or
keep that model on the CPU.

## Where the GPU does pay on a Mac

| Workload | Benefit | How |
|---|---|---|
| **Sentence embeddings** | **5–20×** | `SentenceTransformer(..., device="mps")` |
| Fine-tuning a transformer | large | PyTorch MPS / MLX |
| Any neural network | large | PyTorch MPS / MLX |
| Large dense PCA / SVD | moderate, above crossover | Array API + torch |
| **Gradient boosting** | **none** | no Metal backend exists |
| Tree ensembles | **none** | not GPU-shaped |
| CV of small models | **none** | use CPU cores |

## Memory

```python
X.astype(np.float32)                   # half the memory; trees use float32 anyway
df[col].astype("category")             # often 10-50× on repeated strings
OneHotEncoder(sparse_output=True)      # required above a few thousand columns
```

Estimators that **cannot** take sparse input: `HistGradientBoosting*`, `PCA`
(use `TruncatedSVD`), `GaussianNB`, most of `sklearn.manifold`.

## Out-of-core

When the data does not fit:

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vec = HashingVectorizer(n_features=2**18)     # stateless, no vocabulary
clf = SGDClassifier(loss="log_loss")
for batch_X, batch_y in stream:
    clf.partial_fit(vec.transform(batch_X), batch_y, classes=classes)
```

Estimators with `partial_fit`: `SGDClassifier`/`SGDRegressor`, `MultinomialNB`,
`BernoulliNB`, `Perceptron`, `MiniBatchKMeans`, `IncrementalPCA`,
`MLPClassifier`.

## The consulting version

> "We need GPUs for our ML platform" is a claim to test, not accept. For tabular
> modelling — which is most enterprise ML — it is usually wrong, and the honest
> version is "we need enough RAM to hold the feature store and enough cores to
> cross-validate in parallel". Where GPUs *are* required is the embedding and
> LLM layer, and that is a different budget line with a different justification.
