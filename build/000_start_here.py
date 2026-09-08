# %% [markdown]
# # Start here — the ground beneath scikit-learn
#
# This chapter comes before Module 00. It exists because most people who "know
# some scikit-learn" have a gap one layer down: they can call `fit` and
# `predict`, and they are less sure what a numpy array *is*, why a DataFrame and
# an array behave differently at the boundary of an estimator, or where any of it
# sits in a world of large language models.
#
# By the end of this notebook you should be able to answer four questions:
#
# 1. **What is the numeric Python stack, and what is each layer actually for?**
# 2. **Why is vectorised code fast, and what does that mean for how I write it?**
# 3. **Where does scikit-learn sit, and what is it *for* — in 2026, with LLMs
#    available?**
# 4. **What is the shape of the whole curriculum, and why is it ordered this way?**
#
# It is short, everything runs, and nothing here is prerequisite reading you can
# skip and hope for the best — the confusions it clears up are the ones that
# resurface in Module 04 as a validation bug you cannot explain.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import platform
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from skmastery import set_plot_style

set_plot_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)
rng = np.random.default_rng(0)

print(f"python       {platform.python_version()}")
print(f"numpy        {np.__version__}")
print(f"pandas       {pd.__version__}")
print(f"scipy        {scipy.__version__}")
print(f"scikit-learn {sklearn.__version__}")

# %% [markdown]
# ## 1. The stack, one layer at a time
#
# Four libraries, and they are not alternatives to each other — they are floors
# of the same building.
#
# | Layer | Library | What it gives you | The one-sentence version |
# |---|---|---|---|
# | 4 | **scikit-learn** | estimators, pipelines, model selection, metrics | *one interface for ~200 algorithms* |
# | 3 | **pandas** | labelled, heterogeneous tables | *a spreadsheet with a type system* |
# | 2 | **scipy** | optimisation, sparse matrices, statistics, linear algebra | *the numerical methods numpy leaves out* |
# | 1 | **numpy** | the n-dimensional array, and fast operations on it | *contiguous typed memory, plus loops written in C* |
#
# Everything above rests on layer 1. A pandas column is a numpy array with a
# label; a scikit-learn estimator, when you look inside it, is holding numpy
# arrays with trailing underscores in their names. **When something behaves
# strangely at the top of the stack, the explanation is almost always one or two
# floors down**, which is why this chapter exists.

# %% [markdown]
# ### Layer 1 — numpy, and why it is fast
#
# A Python list is an array of *pointers* to objects scattered through memory,
# each carrying its own type information. A numpy array is one contiguous block
# of memory of a single type. That difference is the whole performance story.

# %%
py_list = list(range(1_000_000))
np_arr = np.arange(1_000_000)


def timed(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = perf_counter()
        fn()
        ts.append(perf_counter() - t0)
    return min(ts)


t_loop = timed(lambda: sum(x * x for x in py_list))
t_vec = timed(lambda: np.sum(np_arr * np_arr))

print(f"python loop  : {t_loop * 1000:8.2f} ms")
print(f"numpy vector : {t_vec * 1000:8.2f} ms")
print(f"speed-up     : {t_loop / t_vec:8.0f}x")
# sys.getsizeof on a list counts the POINTERS only. The integers they point to
# are separate objects, each with its own header -- so measure both.
list_bytes = sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list[:1000]) / 1000 * len(py_list)
print(f"\nmemory: list {list_bytes / 1e6:6.1f} MB   "
      f"({sys.getsizeof(py_list) / 1e6:.0f} MB of pointers + the int objects themselves)")
print(f"        array {np_arr.nbytes / 1e6:6.1f} MB   "
      f"(just the numbers) -> {list_bytes / np_arr.nbytes:.1f}x smaller")

# %% [markdown]
# **What actually happened.** The loop executed a million iterations of the
# Python interpreter: unbox an object, look up `__mul__`, allocate a result, box
# it again. The numpy version passed a pointer and a length to a compiled C loop
# that ran over contiguous `int64` values with no interpreter involved.
#
# > **The habit to build: if you have written a `for` loop over rows of data, ask
# > what the array expression is.** It is usually shorter, always faster, and
# > often clearer. This applies right through the curriculum — Module 05 builds
# > logistic regression from scratch, and it is fifteen lines precisely because
# > none of them loop over samples.

# %%
# The three ideas that make array code work. Everything else is detail.

# (a) ELEMENTWISE OPERATIONS -- no loop, no indexing.
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([10.0, 20.0, 30.0, 40.0])
print("a + b        :", a + b)
print("a * b        :", a * b)
print("a > 2        :", a > 2)          # a boolean ARRAY, not a single bool

# (b) BOOLEAN MASKING -- selection expressed as an array.
print("a[a > 2]     :", a[a > 2])
print("count > 2    :", (a > 2).sum())   # True is 1: counting is summing

# (c) BROADCASTING -- the rule that makes shapes line up.
X = np.arange(12).reshape(3, 4).astype(float)
col_means = X.mean(axis=0)               # shape (4,)
print(f"\nX shape {X.shape}, col_means shape {col_means.shape}")
print("X - col_means:\n", X - col_means) # (3,4) - (4,) works: the (4,) is reused per row

# %% [markdown]
# **Broadcasting is the one to understand properly**, because it is the source of
# both numpy's elegance and its most confusing bugs. The rule: align shapes from
# the *right*; a dimension of size 1 (or missing) is stretched to match.
#
# `(3, 4) - (4,)` works — the row vector is reused for every row. `(3, 4) - (3,)`
# does **not** — 3 does not match 4 on the right. You fix it by making the
# intent explicit with `(3, 1)`:

# %%
row_means = X.mean(axis=1)               # shape (3,)
try:
    X - row_means
except ValueError as e:
    print(f"X - row_means -> ValueError: {e}")

print("\nX - row_means[:, None] works:")
print(X - row_means[:, None])            # (3,4) - (3,1): stretched across columns
print("\n`[:, None]` adds an axis. In sklearn code you will see it constantly,")
print("usually to turn a 1-D vector of predictions into a column.")

# %% [markdown]
# ### The shape errors you will actually hit
#
# Three of them account for most of the confusion, and all three are about the
# difference between a **1-D vector** and a **2-D column**:

# %%
v = np.array([1.0, 2.0, 3.0])
print(f"v.shape            {v.shape}   <- 1-D: 'three numbers'")
print(f"v.reshape(-1, 1)   {v.reshape(-1, 1).shape}   <- 2-D: 'three rows, one column'")
print(f"v.reshape(1, -1)   {v.reshape(1, -1).shape}   <- 2-D: 'one row, three columns'")
print()
print("scikit-learn's convention, and it is worth memorising:")
print("  X is ALWAYS 2-D  (n_samples, n_features)")
print("  y is ALWAYS 1-D  (n_samples,)")
print()
print("So a single sample is X.reshape(1, -1) -- one row, many columns.")
print("A single feature is X.reshape(-1, 1)   -- many rows, one column.")
print("Getting these the wrong way round is the most common beginner error, and")
print("the error message names both: 'Expected 2D array, got 1D array instead'.")

# %% [markdown]
# ### Layer 2 — scipy, in one paragraph
#
# numpy gives you the array; scipy gives you the numerical methods over it. In
# this curriculum you will meet four corners of it, and it is useful to know they
# are all the same library:
#
# - **`scipy.sparse`** — matrices that store only their non-zeros. Text
#   vectorisation (Module 10) produces matrices that are 99.9% zeros; storing
#   them densely would need more memory than exists.
# - **`scipy.optimize`** — the solvers underneath `LogisticRegression` and most
#   linear models (Module 05).
# - **`scipy.stats`** — distributions, used for hypothesis tests and for
#   `RandomizedSearchCV`'s parameter distributions (Module 07).
# - **`scipy.linalg`** — decompositions; PCA is an SVD in a wrapper (Module 09).

# %%
from scipy import sparse

dense = np.zeros((1000, 1000))
dense[rng.integers(0, 1000, 500), rng.integers(0, 1000, 500)] = 1.0
sp = sparse.csr_matrix(dense)
print(f"dense : {dense.nbytes / 1e6:.2f} MB")
print(f"sparse: {(sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes) / 1e6:.4f} MB "
      f"({dense.nbytes / (sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes):.0f}x smaller)")
print(f"density: {sp.nnz / dense.size:.3%}")

# %% [markdown]
# ### Layer 3 — pandas, and the boundary that catches people
#
# pandas adds two things numpy does not have: **labels** (column names, an index)
# and **heterogeneous types** (one table with integers, floats, strings and
# dates). That is exactly what real tabular data needs, and it is why every
# dataset in this curriculum is a DataFrame.
#
# The subtlety worth internalising now: **scikit-learn estimators accept
# DataFrames, but most of them work in arrays internally.** Where the labels
# survive that round trip, and where they are silently dropped, is the source of
# a lot of confusion later.

# %%
df = pd.DataFrame({
    "age": [34, 51, 28, 45],
    "income": [42_000.0, 88_000.0, 31_500.0, np.nan],
    "region": ["north", "south", "north", "west"],
    "prior_customer": [True, False, True, True],
})
print(df.dtypes.to_string())
print()
print("A DataFrame is a dict of numpy arrays with an index bolted on:")
print(f"  df['age'].to_numpy() -> {df['age'].to_numpy()}  dtype {df['age'].dtype}")
print(f"  df.to_numpy().dtype  -> {df.to_numpy().dtype}   <- ALL columns become object!")
print()
print("That last line is the boundary. Mixing strings and numbers forces numpy to")
print("the 'object' dtype, which is a pointer array again -- slow, and unusable by")
print("most estimators. Encoding the strings first is not tidiness; it is what")
print("makes the data representable at all. Module 02 and 03 are about doing that")
print("correctly, and Module 03 is about doing it WITHOUT leaking information.")

# %%
# The single most useful pandas idiom in this curriculum: split-apply-combine.
sales = pd.DataFrame({
    "region": ["north", "south", "north", "west", "south", "north"],
    "amount": [100.0, 250.0, 175.0, 90.0, 300.0, 210.0],
    "defaulted": [0, 1, 0, 0, 1, 0],
})
print(sales.groupby("region").agg(n=("amount", "size"),
                                  total=("amount", "sum"),
                                  default_rate=("defaulted", "mean")).round(3).to_string())
print("\nIf you can read that, you can read 80% of the data-handling code in this")
print("curriculum. Almost every diagnostic in it is a groupby with a metric.")

# %% [markdown]
# ## 2. What scikit-learn actually is
#
# Here is the whole library in one idea, and Module 00 spends a chapter on it
# because everything else follows:
#
# > **scikit-learn is one interface, implemented about two hundred times.**
#
# Every estimator — a scaler, a random forest, a clustering algorithm, a
# dimensionality reducer — supports some subset of the same four methods:
#
# | Method | Meaning | Who has it |
# |---|---|---|
# | `fit(X, y)` | learn from data; store what you learned with a trailing underscore | everything |
# | `transform(X)` | apply what you learned to produce new features | transformers |
# | `predict(X)` | apply what you learned to produce an answer | predictors |
# | `score(X, y)` | how good was that | predictors |
#
# Because the interface is uniform, tools that know *nothing* about your
# particular algorithm can operate on it: `Pipeline`, `GridSearchCV`,
# `cross_validate`, `CalibratedClassifierCV`. **Learning the interface is worth
# more than learning any individual algorithm**, and it is why Module 11 (writing
# your own estimator that passes the library's own conformance suite) is a more
# important module than it sounds.

# %%
# The whole thing, end to end, in ten lines.
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skmastery import load_credit_risk

credit = load_credit_risk()
y = credit["default"]
X = credit.drop(columns=["default", "application_id", "collections_flag", "application_month"])

model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_include=["object", "string"])),
    ])),
    ("clf", LogisticRegression(max_iter=2000)),
])
scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"5-fold ROC AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

# %% [markdown]
# **Read what just happened, because it is the shape of everything to come.**
#
# - `ColumnTransformer` routed numeric and text columns down different paths.
# - `Pipeline` chained the whole thing into a single object with a `fit` method.
# - `cross_val_score` split the data five ways and, crucially, **re-fitted the
#   entire pipeline on each training fold** — including the imputer's median and
#   the scaler's mean.
#
# That last point is the reason pipelines exist, and it is not a convenience. If
# you had imputed and scaled *before* splitting, the medians and means would have
# been computed using the validation rows, and your score would be optimistic by
# an amount you cannot estimate. **The pipeline is a correctness mechanism.**
# Module 03 is about that; Module 04 measures what it costs you when you get it
# wrong.

# %%
# What "fitted" means, concretely: state with a trailing underscore.
model.fit(X, y)
scaler = model.named_steps["prep"].named_transformers_["num"].named_steps["sc"]
print("Before fit, a fresh StandardScaler has no mean_ attribute at all.")
print(f"After fit, it does: mean_[:4] = {np.round(scaler.mean_[:4], 3)}")
print(f"                    scale_[:4] = {np.round(scaler.scale_[:4], 3)}")
print()
print("THE convention: `alpha` (no underscore) is something YOU set.")
print("                `mean_` (trailing underscore) is something the DATA set.")
print("Everything in Module 00 follows from taking that seriously.")

# %% [markdown]
# ## 3. Why this matters in 2026
#
# It is a fair question. Large language models can write this code, and in some
# framings can do the task directly. Three honest answers.
#
# ### (a) Most enterprise machine learning is still tabular, and tabular is still
# gradient-boosted trees
#
# Credit decisions, fraud, claims, churn, pricing, collections, capital models,
# demand forecasting — these are rows and columns with a numeric outcome, and on
# that shape of problem a well-tuned boosted tree remains at or near the state of
# the art. Not because the field is behind, but because the inductive bias fits:
# tabular features are heterogeneous, low-dimensional, and full of sharp
# thresholds, which is exactly what trees represent well and what
# attention-based models represent awkwardly.
#
# ### (b) scikit-learn is the harness around the LLM, not a competitor to it
#
# Module 10 makes this concrete. In a modern LLM system, scikit-learn is:
#
# - the **classifier head** on frozen embeddings — embed once, fit a logistic
#   regression, get a calibrated, explainable, millisecond classifier;
# - the **router** that decides which requests need the expensive model at all,
#   which is often the single largest cost line in an LLM system;
# - the **retrieval index** (`NearestNeighbors` over embeddings is a perfectly
#   good vector store up to about a million documents);
# - the **guardrail**, a fast classifier on inputs and outputs at a threshold
#   tuned to an explicit cost matrix;
# - and **the entire evaluation harness**. Cross-validation, metrics,
#   calibration, significance testing — none of that changed because the features
#   got denser.
#
# > **The harness outlives the representation.** Module 10 demonstrates this by
# > swapping TF-IDF for embeddings behind an unchanged pipeline: every number in
# > the section recomputes from a one-line change. That property is what makes
# > the sklearn skills durable rather than dated.
#
# ### (c) The part an LLM cannot do for you is the part that carries the risk
#
# An LLM will write you a `GridSearchCV` in seconds. It will not, unprompted:
#
# - notice that your random split leaks because fraud clusters on cards
#   (Module 04 measures that at a 34% inflation);
# - notice that a feature is recorded *after* the outcome it predicts
#   (Module 12's leakage exercise — no interpretability tool catches it either);
# - tell you that your model is already at the irreducible accuracy ceiling and
#   further tuning is not available (Module 10.2);
# - tell you that the fairness metric your committee is optimising moves by half
#   its range on threshold alone (Module 12.4);
# - or tell you that the model your team is proudest of would be rejected by
#   model risk for a reason that has nothing to do with its accuracy.
#
# **Those judgements are the job.** The code is the easy part, and it has been
# the easy part for a while. This curriculum is arranged around the judgement.

# %% [markdown]
# ### A note on where the real leverage is
#
# > 💼 **Consulting lens.** The most valuable output of a modelling engagement is
# > often a recommendation to *stop modelling*. Module 10 computes an accuracy
# > ceiling and finds the model already at it. Module 14 recommends a logistic
# > regression over a boosted ensemble and then, in the validator's review,
# > attacks the grounds on which that recommendation was made. Module 13
# > establishes that no model monitor would have caught a data incident that a
# > four-line range check would have caught on day one.
# >
# > None of those conclusions is a modelling result. All of them require knowing
# > the modelling well enough to be sure — which is the reason to learn it
# > properly rather than by prompt.

# %% [markdown]
# ## 4. How the curriculum is arranged
#
# Sixteen modules in three parts, plus this chapter.
#
# ### Part I — Foundations (00–05)
# **The idea: most errors happen before any learning does.**
# The estimator contract, data representation, transformers, pipelines,
# validation design, and linear models built from scratch. Module 04 is the one
# most people should not skip: a wrong split moves your metric by twenty points,
# invisibly.
#
# ### Part II — The model families (06–10)
# **The idea: choose the family from the problem, then measure the prize before
# spending the budget.**
# Trees and ensembles, hyperparameter search, metrics and thresholds,
# unsupervised learning, and text through to embeddings. Module 08 is the one
# people skip most and benefit from most: the model outputs a score, the business
# needs a decision, and those are different problems.
#
# ### Part III — Practice and performance (11–15)
# **The idea: the work that survives contact with an organisation.**
# Writing your own estimators, interpretability and fairness, production,
# a full end-to-end capstone, and what actually makes tabular ML fast.
#
# ### How to work through it
#
# 1. **Run every cell, and guess the output before you read it.** The guessing is
#    the learning; the running only confirms it.
# 2. **Do the exercises before opening the solutions.** They are where the
#    material moves from recognisable to usable, and the solutions are written as
#    worked arguments rather than answer keys — several of them reach the
#    *opposite* conclusion to the one the exercise implies, because that is what
#    the data said.
# 3. **Break things on purpose.** Most modules contain a "here is the wrong way,
#    measured" section. Extend those.
# 4. **Keep the cheat sheets open.** They are meant to be used while working, not
#    read once.
#
# Modules 00–05 are sequential and worth doing in order. From 06 you can jump to
# what you need.

# %% [markdown]
# ## 5. A five-minute self-check
#
# If you can answer these without running anything, start at Module 04. If any of
# them is uncomfortable, start at Module 00 — it will take an afternoon and save
# you a week.

# %%
QUESTIONS = [
    ("What shape must X be for an sklearn estimator, and what shape must y be?",
     "X is 2-D (n_samples, n_features); y is 1-D (n_samples,)."),
    ("What does a trailing underscore on an attribute mean?",
     "It was learned from data during fit, not set by you in __init__."),
    ("Why does `np.array([1,2,3]) - np.array([[1],[2]])` produce a 2x3 array?",
     "Broadcasting: shapes (3,) and (2,1) align from the right to (2,3)."),
    ("Why fit a scaler inside a pipeline rather than before splitting?",
     "Fitting before the split computes the mean using validation rows, so the "
     "score is optimistic by an unknown amount."),
    ("What does cross_val_score do that train_test_split does not?",
     "It refits the whole estimator on each of k training folds, giving k "
     "estimates and therefore a spread, not one number."),
    ("A model has 95% accuracy on a problem with a 5% positive rate. Good?",
     "Unknown -- predicting the majority class always would score 95%. "
     "Compare against that baseline first (Module 08)."),
]
for i, (q, a) in enumerate(QUESTIONS, 1):
    print(f"{i}. {q}")
print("\n" + "-" * 70)
print("Answers:")
for i, (q, a) in enumerate(QUESTIONS, 1):
    print(f"{i}. {a}")

# %% [markdown]
# ---
# ## Takeaways
#
# - **numpy is contiguous typed memory plus C loops.** Vectorised code is fast
#   because the interpreter is not in it. If you wrote a `for` loop over rows,
#   there is an array expression that is shorter and faster.
# - **Broadcasting aligns shapes from the right**, stretching size-1 dimensions.
#   `[:, None]` is how you make the intent explicit.
# - **X is 2-D, y is 1-D.** Most shape errors are this, and the message says so.
# - **pandas is labels and mixed types over numpy.** `df.to_numpy()` on a mixed
#   table gives you `object` dtype — which is why encoding comes first.
# - **scikit-learn is one interface implemented ~200 times.** Learn the
#   interface; the algorithms are interchangeable behind it.
# - **The trailing underscore is the whole state convention.** `alpha` is yours,
#   `mean_` is the data's.
# - **A pipeline is a correctness mechanism, not a convenience.** It is what makes
#   cross-validation honest.
# - **In an LLM stack, sklearn is the head, the router, the index, the guardrail
#   and the entire evaluation harness.** The harness outlives the representation.
#
# **Next:** Module 00 — the estimator contract, in detail. Everything in the
# curriculum composes because of it.
