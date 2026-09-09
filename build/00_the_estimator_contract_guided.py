# %% [markdown]
# # Guided — Module 00: The Estimator Contract
#
# This sits between the module and the solutions. The solutions show you a
# finished answer; this shows you the **route to one**, and it stops short of
# the answer until you ask for it.
#
# **How to use it.** Every exercise below has the same five parts:
#
# 1. **What this is really testing** — the one idea, so you know what "done"
#    means.
# 2. **Before you write code** — a question to answer in your head first. If you
#    can answer it, the code is usually ten lines.
# 3. **Hints** — three of them, each behind a ▸ you have to click. Take the
#    smallest one that unsticks you.
# 4. **A self-check** — a function you run against *your* answer that tells you
#    whether it is right, without showing you the right answer.
# 5. **The worked answer**, last, also behind a ▸.
#
# The self-checks are the part worth using. Being stuck is usually not "I cannot
# write the code" — it is "I wrote something and I do not know whether it is
# right." That is a fixable problem and it is what these solve.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions", "guided"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skmastery import set_plot_style

set_plot_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.base import clone, is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

print("ready")


def reveal(title, body):
    """Render a click-to-open block. Jupyter renders HTML inside markdown, and
    this notebook uses that so a hint you have not opened cannot spoil you."""
    from IPython.display import Markdown, display
    display(Markdown(f"<details><summary><b>{title}</b></summary>\n\n{body}\n\n</details>"))

# %% [markdown]
# ---
# # Exercise 0.4 — Quantify the leak
#
# **Starting here, because this is where you said you were stuck.** The others
# follow in order below.

# %% [markdown]
# ## What this is really testing
#
# Not the loop. The loop is eight lines and you can probably already write it.
#
# It is testing whether you can survive a **null result**. You are being asked to
# build an instrument, point it at a known-bad practice, and read a number very
# close to zero — and then say something true about what that means. Almost
# everyone who does this exercise writes the code correctly, sees a gap of about
# 0.0004, and concludes they have made a mistake.
#
# You have not made a mistake. **The gap really is that small, and the exercise
# is about why.**
#
# So the skill being built is: *when a measurement says "no effect", separate
# "my instrument is broken" from "the effect is genuinely absent here" — and
# then explain the absence rather than filing it away as a curiosity.*

# %% [markdown]
# ## Before you write code
#
# Answer this in your head. It takes thirty seconds and it changes what you
# write:
#
# > **When `StandardScaler` leaks, what exactly is the leaked information?**
#
# Not "the test set". Be specific. `StandardScaler.fit` stores exactly two
# arrays. Name them. Then ask: *if those two arrays were slightly wrong, would a
# logistic regression's predictions change?*
#
# If your instinct is "obviously yes, the inputs would be scaled wrongly" —
# that instinct is the thing this exercise is about to correct.

# %%
reveal("▸ Hint 1 — what is actually leaking",
"""`StandardScaler.fit` learns two arrays: `mean_` and `scale_`, one number per
feature. That is the entire learned state — thirty features, sixty numbers.

So the leak is: *those sixty numbers were computed using rows that later became
the test set.* Nothing else crosses the boundary. No labels, no row-level
information, no structure.

Now the useful question, and the one most people skip: **how wrong are those
sixty numbers, and does anything downstream care?**

Two separate things have to both be true for a leak to show up in a score:

1. The leaked information has to *change the fitted transform* by a meaningful
   amount, and
2. the downstream model's answer has to *depend* on that part of the transform.

Write those two down. You are going to test both of them.""")

# %%
reveal("▸ Hint 2 — making the two arms comparable",
"""The trap here is an unpaired comparison. If arm A and arm B see different
splits, you are measuring the leak *plus* split-to-split noise — and on this
dataset the noise is far larger than the leak, so you will get a number that
bounces around and means nothing.

**Use the same `random_state` for both arms inside each iteration**, so the two
workflows see the identical train/test partition. Then the per-seed difference
`leaky[i] - clean[i]` isolates the only thing you changed.

Report the *paired* difference and its standard deviation, not two separate
means with two separate error bars. `leaky.mean() - clean.mean()` is the same
number either way, but `np.std(leaky - clean)` is a much smaller and much more
honest error bar than `leaky.std()`.

That is a general habit worth forming: **when you can pair, pair.**""")

# %%
reveal("▸ Hint 3 — the skeleton",
"""```python
from sklearn.datasets import load_breast_cancer

X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)

leaky, clean = [], []
for seed in range(30):
    # ARM A -- leaky: scale the WHOLE X first, then split.
    X_all = ...                     # TODO: StandardScaler().fit_transform on all of X_bc
    a_tr, a_te, ya_tr, ya_te = train_test_split(
        X_all, y_bc, test_size=0.3, random_state=seed, stratify=y_bc)
    leaky.append(...)               # TODO: fit LogisticRegression(max_iter=5000), .score on test

    # ARM B -- clean: split FIRST, then a pipeline that fits the scaler on train only.
    b_tr, b_te, yb_tr, yb_te = train_test_split(
        X_bc, y_bc, test_size=0.3, random_state=seed, stratify=y_bc)   # same seed!
    pipe = Pipeline([...])          # TODO: scaler + logistic regression
    clean.append(...)               # TODO: fit on train, score on test

leaky, clean = np.array(leaky), np.array(clean)
print(f"A (leaky) : {leaky.mean():.4f}")
print(f"B (clean) : {clean.mean():.4f}")
print(f"paired difference: {(leaky - clean).mean():+.5f} +/- {(leaky - clean).std():.5f}")
```

Fill in the four TODOs, run it, and then run the self-check below. **Expect a
difference near zero** — that is the correct answer, not a bug.""")

# %% [markdown]
# ## Self-check
#
# Run your version, then pass your two arrays to this. It checks the shape of
# your result without telling you the answer.

# %%
def check_04(leaky, clean):
    """Tell the reader whether their 0.4 harness is built correctly."""
    leaky, clean = np.asarray(leaky, float), np.asarray(clean, float)
    ok = True
    if leaky.shape != clean.shape or leaky.size < 10:
        print("✗ Both arrays should hold one score per seed, 30 of them.")
        return False
    if np.allclose(leaky, clean):
        print("✗ The two arms are IDENTICAL, seed for seed. Arm A is not leaking —")
        print("  the usual cause is fitting the scaler on the training split rather")
        print("  than on the whole of X. Arm A must call fit_transform BEFORE the split.")
        return False
    paired_sd = np.std(leaky - clean)
    unpaired_sd = np.hypot(leaky.std(), clean.std())
    if paired_sd > 0.9 * unpaired_sd:
        print("✗ Your paired differences are as noisy as two independent runs, which")
        print("  means the two arms are probably seeing DIFFERENT splits. Use the same")
        print("  random_state for both arms inside each iteration.")
        ok = False
    gap = (leaky - clean).mean()
    if abs(gap) > 0.02:
        print(f"✗ A paired gap of {gap:+.4f} is far larger than this pairing can produce.")
        print("  Check that arm B really is fitting the scaler inside the pipeline.")
        ok = False
    if ok:
        print(f"✓ Harness looks right. Paired gap {gap:+.5f} (sd {paired_sd:.5f}) over "
              f"{leaky.size} seeds.")
        print("  A gap this small is the EXPECTED result. Keep going — the next two")
        print("  cells are where the exercise actually pays off.")
    return ok


# Demonstrated on a correct run so you can see what a pass looks like.
X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)
_leaky, _clean = [], []
for seed in range(30):
    X_all = StandardScaler().fit_transform(X_bc)
    a_tr, a_te, ya_tr, ya_te = train_test_split(X_all, y_bc, test_size=0.3,
                                                random_state=seed, stratify=y_bc)
    _leaky.append(LogisticRegression(max_iter=5000).fit(a_tr, ya_tr).score(a_te, ya_te))
    b_tr, b_te, yb_tr, yb_te = train_test_split(X_bc, y_bc, test_size=0.3,
                                                random_state=seed, stratify=y_bc)
    _clean.append(Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=5000))]
                           ).fit(b_tr, yb_tr).score(b_te, yb_te))
check_04(_leaky, _clean)

# %% [markdown]
# ## The part that unsticks you: is the instrument broken?
#
# You have a number near zero. Before explaining it, **prove your harness can
# detect a leak at all.** Point the identical experiment at a transformer that
# leaks badly, and see whether the needle moves.
#
# This is a *positive control*, and it is the single most useful habit in this
# notebook. It converts "did I do it wrong?" into a question you can answer.

# %%
rs = np.random.default_rng(0)
n_rows, n_cols = 300, 3000
X_noise = rs.normal(size=(n_rows, n_cols))
y_noise = rs.integers(0, 2, n_rows)          # LABELS ARE RANDOM. Truth = 0.50.

pc_leaky, pc_clean = [], []
for seed in range(15):
    # Same shape of mistake: fit the transformer on EVERYTHING, then split.
    sel = SelectKBest(f_classif, k=20).fit(X_noise, y_noise)
    Xa = sel.transform(X_noise)
    a_tr, a_te, ya_tr, ya_te = train_test_split(Xa, y_noise, test_size=0.3,
                                                random_state=seed, stratify=y_noise)
    pc_leaky.append(LogisticRegression(max_iter=2000).fit(a_tr, ya_tr).score(a_te, ya_te))

    b_tr, b_te, yb_tr, yb_te = train_test_split(X_noise, y_noise, test_size=0.3,
                                                random_state=seed, stratify=y_noise)
    pc_clean.append(Pipeline([("sel", SelectKBest(f_classif, k=20)),
                              ("lr", LogisticRegression(max_iter=2000))]
                             ).fit(b_tr, yb_tr).score(b_te, yb_te))

pc_leaky, pc_clean = np.array(pc_leaky), np.array(pc_clean)
print("POSITIVE CONTROL — the same experiment, with SelectKBest instead of StandardScaler")
print("Data is pure noise: the honest accuracy is 0.50 by construction.\n")
print(f"  leaky : {pc_leaky.mean():.4f}")
print(f"  clean : {pc_clean.mean():.4f}")
print(f"  gap   : {pc_leaky.mean() - pc_clean.mean():+.4f}")
print()
print("The harness detects a leak of a quarter of accuracy on data with NO signal.")
print("So it is not broken. The StandardScaler gap really is ~0.000.")

# %% [markdown]
# ## So why is it zero? Test the two candidate explanations
#
# From Hint 1, a leak needs **both** of these to be true:
#
# 1. the leaked rows **change the fitted transform** appreciably, and
# 2. the downstream model's answer **depends on** that part of the transform.
#
# The usual explanation given for this exercise is (1): 171 test rows barely move
# a mean computed over 569, so the constants hardly change. That is true, and it
# is *not the main reason*. Test (2) directly — deliberately sabotage the
# scaler's learned constants, far more violently than any leak could, and see
# whether the score notices.

# %%
Xtr, Xte, ytr, yte = train_test_split(X_bc, y_bc, test_size=0.3, random_state=0, stratify=y_bc)


def sabotaged_scaler(sigma):
    """A StandardScaler whose learned constants are deliberately corrupted."""
    class Sabotaged(StandardScaler):
        def fit(self, X, y=None, **kw):
            super().fit(X, y, **kw)
            r = np.random.default_rng(1)
            self.mean_ = self.mean_ * (1 + r.normal(0, sigma, self.mean_.shape))
            self.scale_ = self.scale_ * np.exp(r.normal(0, sigma, self.scale_.shape))
            return self
    return Sabotaged()


rows = []
for sigma in [0.0, 0.05, 0.25, 1.0, 4.0]:
    sc = StandardScaler() if sigma == 0 else sabotaged_scaler(sigma)
    acc = Pipeline([("sc", sc), ("lr", LogisticRegression(max_iter=5000))]
                   ).fit(Xtr, ytr).score(Xte, yte)
    rows.append({"corruption of mean_/scale_": "none (correct)" if sigma == 0 else f"sigma = {sigma}",
                 "test accuracy": round(acc, 4)})
sab = pd.DataFrame(rows).set_index("corruption of mean_/scale_")
print(sab.to_string())
print()
print("Read the bottom row. The scaler's constants were corrupted by orders of")
print("magnitude -- far beyond anything a 30% test set could do -- and accuracy")
print("moved by a couple of points. At realistic corruption levels it does not")
print("move at all.")

# %% [markdown]
# ## The answer, and it is better than "the statistic barely moves"
#
# **A logistic regression is almost invariant to how you scale its inputs.**
# Rescaling feature *j* by a constant is undone by the model rescaling
# coefficient *j* by its reciprocal; shifting it is absorbed by the intercept.
# The only thing that survives is the interaction with the **regularisation
# penalty**, which is why the accuracy moves a little at extreme corruption and
# not at all at realistic ones.
#
# So the two factors score like this:
#
# | | Does the leak change the fitted transform? | Does the model's answer depend on it? | Leak visible? |
# |---|---|---|---|
# | `StandardScaler` → logistic regression | barely (2 numbers per feature, 569 rows) | **almost not at all** | no |
# | `SelectKBest` → anything | **enormously** — it picks *which columns exist* using the labels | totally | **+0.24 on pure noise** |
#
# **The generalisable rule, which is what to carry out of this exercise:**
#
# > The size of a leak is *(how much the leaked information changes the fitted
# > transform)* × *(how much the model's answer depends on that transform)*.
# > A near-zero reading means one of those factors was near zero **for this
# > pairing** — it says nothing whatsoever about the next one.
#
# And the practical consequence, which is the reason the module bothers:
#
# > **You cannot audit for leakage by measuring it.** Measuring gives you one
# > number for one pairing on one dataset. What protects you is the *structural*
# > guarantee that `fit` only ever sees training rows — which is what a
# > `Pipeline` is, and why the module calls it a correctness mechanism rather
# > than a convenience.
#
# Module 03 puts a number on the other end of the range: feature selection before
# cross-validation reaches **AUC 0.82 on 3,000 columns of pure noise**, where the
# truth is 0.50. Same mistake, same shape of code, four hundred times the damage.

# %%
reveal("▸ The worked answer for 0.4",
"""```python
from sklearn.datasets import load_breast_cancer

X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)

leaky, clean = [], []
for seed in range(30):
    # A: scale everything, THEN split -- the test rows contributed to mean_/scale_
    X_all = StandardScaler().fit_transform(X_bc)
    a_tr, a_te, ya_tr, ya_te = train_test_split(
        X_all, y_bc, test_size=0.3, random_state=seed, stratify=y_bc)
    leaky.append(LogisticRegression(max_iter=5000).fit(a_tr, ya_tr).score(a_te, ya_te))

    # B: split FIRST; the pipeline fits the scaler on the training rows only
    b_tr, b_te, yb_tr, yb_te = train_test_split(
        X_bc, y_bc, test_size=0.3, random_state=seed, stratify=y_bc)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=5000))])
    clean.append(pipe.fit(b_tr, yb_tr).score(b_te, yb_te))

leaky, clean = np.array(leaky), np.array(clean)
print(f"A (leaky) : {leaky.mean():.4f}")
print(f"B (clean) : {clean.mean():.4f}")
print(f"paired difference: {(leaky - clean).mean():+.5f} +/- {(leaky - clean).std():.5f}")
```

**Two sentences on why the gap is small**, which is what the exercise asks for:

> `StandardScaler` learns only a per-feature mean and standard deviation, and a
> logistic regression is very nearly invariant to an affine rescaling of its
> inputs — it simply rescales its coefficients — so even badly wrong constants
> barely move the decision boundary. The leak is real and the instrument
> detects leaks (it finds +0.24 with `SelectKBest` on pure noise); it reads zero
> here because this particular transformer carries almost no information that
> this particular model uses.

**Transformers that would move it a lot:** anything that sees the target
(`SelectKBest`, `TargetEncoder`, `SelectFromModel`), anything that learns a
high-dimensional object from few rows (`PCA` on wide data,
`QuantileTransformer` with few samples per quantile), and anything that changes
*which columns exist* rather than their units.""")

# %% [markdown]
# ---
# # Exercise 0.1 — Classify the library

# %% [markdown]
# ## What this is really testing
#
# That you can interrogate an estimator's **role** without knowing its name.
# Every tool in the library — `Pipeline`, `GridSearchCV`, `cross_validate` —
# does exactly this internally, because it has to work with estimators that did
# not exist when it was written. You are writing a tiny version of that.
#
# ## Before you write code
#
# > **What distinguishes a transformer from a predictor?** Not the class name.
# > What can you actually ask an *object* to find out?

# %%
reveal("▸ Hint 1 — the question to ask an object",
"""Roles are defined by **which methods exist**, so the test is `hasattr`:

- a **transformer** has `transform`
- a **predictor** has `predict`
- some estimators are both (`PCA` has `transform`; `KMeans` has both)

For classifier-vs-regressor, do **not** use `hasattr` and do not read the class
name — scikit-learn exposes `is_classifier(est)` and `is_regressor(est)`, which
read the estimator's tags. That is the supported way and it keeps working for
estimators you did not write.

For fitted-vs-unfitted, `check_is_fitted(est)` raises `NotFittedError` when the
estimator is unfitted, so wrap it in a try/except and return a bool.""")

# %%
reveal("▸ Hint 2 — the shape of the function",
"""```python
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

def describe_object(est):
    try:
        check_is_fitted(est)
        fitted = True
    except NotFittedError:
        fitted = False
    return {
        "is_transformer": hasattr(est, "transform"),
        "is_predictor":   hasattr(est, "predict"),
        "is_classifier":  is_classifier(est),
        "is_regressor":   is_regressor(est),
        "is_fitted":      fitted,
    }
```

Then build the table with a dict comprehension keyed on
`type(est).__name__`, and `pd.DataFrame(...).T` to get one row per estimator.""")

# %%
reveal("▸ Hint 3 — the row that surprises people",
"""Look carefully at `KMeans`. It is a transformer **and** a predictor, and it is
neither a classifier nor a regressor.

That is not a quirk. `KMeans.transform` returns distances to each centroid — a
legitimate feature extractor — and `KMeans.predict` returns a cluster
assignment, which is not a supervised label. The four flags are genuinely
independent, which is exactly why the library asks four separate questions
rather than one `type` check.

If your table has `KMeans` looking like a classifier, you used the class name
somewhere.""")

# %%
def check_01(fn):
    """Run the reader's describe_object against a few known cases."""
    from sklearn.cluster import KMeans
    cases = [
        (StandardScaler(), dict(is_transformer=True, is_predictor=False,
                                is_classifier=False, is_regressor=False, is_fitted=False)),
        (LogisticRegression(), dict(is_transformer=False, is_predictor=True,
                                    is_classifier=True, is_regressor=False, is_fitted=False)),
        (Ridge(), dict(is_transformer=False, is_predictor=True,
                       is_classifier=False, is_regressor=True, is_fitted=False)),
        (KMeans(n_init="auto"), dict(is_transformer=True, is_predictor=True,
                                     is_classifier=False, is_regressor=False, is_fitted=False)),
        (StandardScaler().fit(np.arange(20).reshape(-1, 2).astype(float)),
         dict(is_transformer=True, is_predictor=False, is_classifier=False,
              is_regressor=False, is_fitted=True)),
    ]
    bad = []
    for est, want in cases:
        got = fn(est)
        for k, v in want.items():
            if bool(got.get(k)) != v:
                bad.append(f"{type(est).__name__}.{k}: got {got.get(k)!r}, expected {v!r}")
    if bad:
        print("✗ " + "\n✗ ".join(bad))
        return False
    print("✓ describe_object agrees on all five reference objects, fitted and unfitted.")
    return True


reveal("▸ The worked answer for 0.1",
"""```python
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

def describe_object(est):
    try:
        check_is_fitted(est); fitted = True
    except NotFittedError:
        fitted = False
    return {"is_transformer": hasattr(est, "transform"),
            "is_predictor":   hasattr(est, "predict"),
            "is_classifier":  is_classifier(est),
            "is_regressor":   is_regressor(est),
            "is_fitted":      fitted}

objs = [StandardScaler(), PCA(), KMeans(n_init="auto"), LogisticRegression(),
        Ridge(), StandardScaler().fit(np.random.default_rng(0).normal(size=(10, 3)))]
pd.DataFrame({f"{type(o).__name__}{' (fitted)' if describe_object(o)['is_fitted'] else ''}":
              describe_object(o) for o in objs}).T
```

The point to take away: **the library asks objects what they can do, never what
they are called.** That is the whole reason a `Pipeline` written in 2013 works
with an estimator written in 2026.""")

# %% [markdown]
# ---
# # Exercise 0.2 — Derive the parameter path

# %% [markdown]
# ## What this is really testing
#
# Whether you can read a nested estimator as a **tree of named steps**, and turn
# a path through that tree into a string. Every hyperparameter search you ever
# write depends on getting these right, and guessing them wastes an hour at a
# time.
#
# ## Before you write code
#
# > **What is the separator, and what are the names on either side of it?**
# > Say the path out loud in English first: *"the pipeline's `prep` step, its
# > `num` branch, its `poly` step, its `degree` parameter."* Now write that with
# > the separator between each pair.

# %%
reveal("▸ Hint 1 — the rule, in one line",
"""The separator is a **double underscore**, `__`, and the names are the ones
**you** gave the steps when you built the object — not the class names.

```
<step name>__<step name>__...__<parameter name>
```

A `ColumnTransformer`'s branches are named the same way its transformers are, so
`prep__num` reaches the inner `Pipeline`, and `prep__num__poly` reaches the
`PolynomialFeatures` inside it.

Say the English sentence, replace every "'s" with `__`, and you have the key.""")

# %%
reveal("▸ Hint 2 — write these four down before checking",
"""Predict all four, in writing, *then* verify. Predicting first is the exercise;
looking them up is not.

1. the polynomial degree
2. the number of features `SelectKBest` keeps
3. the forest's `max_depth`
4. the one-hot encoder's `min_frequency`

Then:

```python
keys = set(pipe.get_params(deep=True))
for k in ["...", "...", "...", "..."]:
    print(f"{k:45s} {'FOUND' if k in keys else 'NOT A KEY'}")
```

If one is wrong, do not just fix it — work out *why* your path was wrong. The
usual cause is using a class name (`polynomialfeatures`) where a step name
(`poly`) belongs.""")

# %%
reveal("▸ Hint 3 — how to search when you are lost",
"""`get_params(deep=True)` returns a flat dict of every reachable key. Filter it:

```python
[k for k in pipe.get_params(deep=True) if k.endswith("degree")]
[k for k in pipe.get_params(deep=True) if "poly" in k]
```

This is the answer to "how do I find the key for X" in general, and it is why
the module says *never guess them*. Two seconds of filtering beats ten minutes
of a `ValueError: Invalid parameter` loop.

Worth noticing while you are in there: `len(pipe.get_params(deep=True))` is
large. Every one of those is a legal thing to put in a search space.""")

# %%
def check_02(answers):
    """answers: a dict {'degree': key, 'k': key, 'max_depth': key, 'min_frequency': key}"""
    from sklearn.impute import SimpleImputer
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer()), ("poly", PolynomialFeatures())]), ["a"]),
            ("cat", OneHotEncoder(), ["b"]),
        ])),
        ("sel", SelectKBest()),
        ("clf", RandomForestClassifier()),
    ])
    keys = set(pipe.get_params(deep=True))
    truth = {"degree": "prep__num__poly__degree", "k": "sel__k",
             "max_depth": "clf__max_depth", "min_frequency": "prep__cat__min_frequency"}
    ok = True
    for name, want in truth.items():
        got = answers.get(name)
        if got == want:
            print(f"✓ {name:14s} {got}")
        elif got in keys:
            print(f"~ {name:14s} {got}  — a real key, but not the one asked for")
            ok = False
        else:
            print(f"✗ {name:14s} {got!r} is not a key on this estimator")
            ok = False
    return ok


reveal("▸ The worked answer for 0.2",
"""```
polynomial degree      prep__num__poly__degree
SelectKBest's k        sel__k
forest max_depth       clf__max_depth
encoder min_frequency  prep__cat__min_frequency
```

Verify with:

```python
keys = pipe.get_params(deep=True)
for k in ["prep__num__poly__degree", "sel__k", "clf__max_depth",
          "prep__cat__min_frequency"]:
    print(f"{k:32s} {'ok' if k in keys else 'MISSING'}  -> {keys.get(k)}")
```

The one people get wrong is the fourth: a `ColumnTransformer` branch is reached
by *its own name*, so it is `prep__cat__min_frequency` and not
`prep__onehotencoder__min_frequency`. Names you chose, every time.""")

# %% [markdown]
# ---
# # Exercise 0.3 — Prove that `clone` protects you

# %% [markdown]
# ## What this is really testing
#
# That you can see `clone` *working*, on an object you broke yourself. It is a
# three-line experiment and its value is entirely in having run it: after this,
# "estimators must not accumulate state" stops being a rule you were told and
# becomes a thing you watched happen.
#
# ## Before you write code
#
# > **What does `clone` actually copy?** Not the object. It reads
# > `get_params()` and calls the constructor again. So what survives a clone,
# > and what does not?

# %%
reveal("▸ Hint 1 — what clone does, precisely",
"""`clone(est)` does approximately this:

```python
type(est)(**est.get_params(deep=False))
```

It builds a **new, unfitted** object from the same hyperparameters. Nothing
learned survives — no attribute with a trailing underscore, and no attribute you
added by hand during `fit`.

That is the whole protection. `cross_validate` and `GridSearchCV` clone before
every fit, so a fold cannot inherit anything from the fold before it.

Your broken estimator needs to violate exactly that: keep something across
fits.""")

# %%
reveal("▸ Hint 2 — the smallest broken estimator that shows it",
"""```python
from sklearn.base import BaseEstimator

class Accumulator(BaseEstimator):
    def __init__(self, tag="x"):
        self.tag = tag
    def fit(self, X, y=None):
        if not hasattr(self, "seen_"):     # <-- the bug: only initialise once
            self.seen_ = []
        self.seen_.append(len(X))          # <-- and then append forever
        return self
```

`if not hasattr(...)` is the tell. Real estimator code should assign learned
state **unconditionally** on every `fit`.

Now fit it three times on different-sized data, twice: once reusing the object,
once through `clone`. Print `seen_` each time.""")

# %%
reveal("▸ Hint 3 — the loop",
"""```python
sizes = [10, 20, 30]

est = Accumulator()
for n in sizes:
    est.fit(np.zeros((n, 2)))
    print("reused :", est.seen_)

print()
for n in sizes:
    fresh = clone(est).fit(np.zeros((n, 2)))
    print("cloned :", fresh.seen_)
```

Read the two blocks side by side. In the first, fold three sees the sizes of
folds one and two. In the second, each fit sees only its own data — which is
what every cross-validation you have ever run has been quietly doing for you.""")

# %%
def check_03(cls):
    """Confirm the reader's broken class really does accumulate, and that clone stops it."""
    try:
        est = cls()
        for n in (10, 20, 30):
            est.fit(np.zeros((n, 2)))
        state = [v for k, v in vars(est).items() if isinstance(v, list)]
        if not state or len(state[0]) < 3:
            print("✗ Refitting the same object three times did not accumulate anything.")
            print("  The class needs a list attribute it APPENDS to in fit(), initialised")
            print("  only when absent.")
            return False
        fresh = clone(est).fit(np.zeros((10, 2)))
        fstate = [v for k, v in vars(fresh).items() if isinstance(v, list)]
        if fstate and len(fstate[0]) == 1:
            print(f"✓ Reused object accumulated {state[0]}; a clone starts clean at "
                  f"{fstate[0]}.")
            return True
        print("✗ The clone also carried state — check that the accumulating attribute is")
        print("  created inside fit(), not in __init__ (clone re-runs __init__).")
        return False
    except Exception as e:
        print(f"✗ {type(e).__name__}: {e}")
        return False


reveal("▸ The worked answer for 0.3",
"""```python
from sklearn.base import BaseEstimator

class Accumulator(BaseEstimator):
    def __init__(self, tag="x"):
        self.tag = tag
    def fit(self, X, y=None):
        if not hasattr(self, "seen_"):
            self.seen_ = []
        self.seen_.append(len(X))
        return self

sizes = [10, 20, 30]
est = Accumulator()
for n in sizes:
    print("reused :", est.fit(np.zeros((n, 2))).seen_)
print()
for n in sizes:
    print("cloned :", clone(est).fit(np.zeros((n, 2))).seen_)
```

Output:

```
reused : [10]
reused : [10, 20]
reused : [10, 20, 30]

cloned : [10]
cloned : [20]
cloned : [30]
```

**The sting in the tail**, which Module 11 measures: an estimator with this bug
passes all 47 of scikit-learn's own `check_estimator` conformance checks. The
suite fits each estimator once, so it never looks. `clone` protects you from the
bug; nothing in the library detects it. That is why Module 11 writes the missing
test by hand.""")

# %% [markdown]
# ---
# # Exercise 0.5 — Navigate without Google

# %% [markdown]
# ## What this is really testing
#
# Whether you can answer a "which estimator should I use?" question **from the
# library itself** in under two minutes. This is a real working skill: the
# library gains estimators faster than any tutorial is updated, and the
# introspection route never goes stale.
#
# ## Before you write code
#
# > `all_estimators()` gives you every estimator class. You want the ones with a
# > particular *method*. How do you check for a method on a class you have not
# > instantiated?

# %%
reveal("▸ Hint 1 — the three tools",
"""```python
from sklearn.utils.discovery import all_estimators

all_estimators(type_filter="classifier")   # [(name, class), ...]
```

`hasattr` works on the **class**, so you do not need to construct anything:

```python
[n for n, c in all_estimators(type_filter="classifier") if hasattr(c, "partial_fit")]
```

For part 2, filter on the name; for part 3, `print(cls.__doc__[:600])` is your
documentation.""")

# %%
reveal("▸ Hint 2 — part 3 is the one that matters",
"""Parts 1 and 2 are mechanical. Part 3 asks you to **choose and justify**, which
is the actual skill.

Outlier detection on unlabelled tabular data: the candidates are
`IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `EllipticEnvelope`.
Pick one and reject two *with a reason tied to your data*, not to a benchmark:

- How many rows and columns do you have? (`OneClassSVM` is O(n²)-ish;
  `EllipticEnvelope` assumes roughly Gaussian data.)
- Do you need to score **new** points later, or only flag the ones you have?
  This is the decisive question, and it is answered by whether the estimator has
  a `predict` method at all — `LocalOutlierFactor` does not, unless you set
  `novelty=True`.

That last point is worth finding for yourself: it is the estimator contract
doing real work in a real decision.""")

# %%
reveal("▸ Hint 3 — a way to lay the comparison out",
"""```python
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

for cls in [IsolationForest, LocalOutlierFactor, OneClassSVM, EllipticEnvelope]:
    est = cls()
    print(f"{cls.__name__:20s} predict={hasattr(est, 'predict')}  "
          f"score_samples={hasattr(est, 'score_samples')}  "
          f"fit_predict={hasattr(est, 'fit_predict')}")
```

Build that table, then write your three sentences against what it shows.""")

# %%
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils.discovery import all_estimators

partial = [n for n, c in all_estimators(type_filter="classifier") if hasattr(c, "partial_fit")]
encoders = [n for n, c in all_estimators(type_filter="transformer") if "Encoder" in n]
print(f"classifiers with partial_fit ({len(partial)}):")
print("  " + ", ".join(partial))
print(f"\ntransformers named *Encoder* ({len(encoders)}):")
print("  " + ", ".join(encoders))

print("\noutlier detectors, by what they can do:")
rows = []
for cls in [IsolationForest, LocalOutlierFactor, OneClassSVM, EllipticEnvelope]:
    est = cls()
    rows.append({"estimator": cls.__name__,
                 "predict (score NEW points)": hasattr(est, "predict"),
                 "fit_predict": hasattr(est, "fit_predict"),
                 "score_samples": hasattr(est, "score_samples")})
print(pd.DataFrame(rows).set_index("estimator").to_string())

# %%
reveal("▸ The worked answer for 0.5",
"""**1. Out-of-core classifiers** — the ones with `partial_fit`, listed above.
The pattern they enable is Module 13's streaming loop: a stateless
`HashingVectorizer` feeding an `SGDClassifier` a batch at a time, so the data
never has to fit in memory at once.

**2. The encoders**, and when each is right:

- `OneHotEncoder` — nominal categories, low cardinality; the default, and the
  only one that is safe without care about leakage.
- `OrdinalEncoder` — categories with a real order, **or** any categorical feed
  into a tree model, which does not care about the spacing.
- `TargetEncoder` — high cardinality where one-hot would explode. It uses the
  target, so it leaks unless cross-fitted — and it ships with cross-fitting
  built in for exactly that reason (Module 02 measures the leak at AUC 1.000 on
  a pure-noise ID column).
- `LabelEncoder` — **for `y` only.** It takes a 1-D array and it is not a
  feature transformer. Using it on `X` is a common early mistake.
- `LabelBinarizer` / `MultiLabelBinarizer` — also target-side: one-vs-rest and
  multi-label respectively. **Note that the name filter above misses both**,
  because neither contains the string "Encoder" — they turn up under
  `all_estimators(type_filter="transformer")` but not under a search for
  "Encoder". That is worth noticing: searching by name is fast and it silently
  omits things. When the answer matters, filter on *capability* (`hasattr`) or
  on the type filter, not on the spelling of the class.

**3. Outlier detection: `IsolationForest`.**

- It scales to many rows and columns, needs almost no tuning, makes no
  distributional assumption, and — read the table above — it has `predict`, so
  it can score **new** points after fitting. That last property is what makes it
  deployable rather than merely analytical.
- **Rejected `LocalOutlierFactor`**: by default it has no `predict` at all. It
  is designed to flag outliers *within the set it was fitted on*, which is a
  different job. (`novelty=True` swaps that behaviour, and then it loses
  `fit_predict` — the two modes are mutually exclusive, which the table shows.)
- **Rejected `EllipticEnvelope`**: it fits a robust covariance and therefore
  assumes the inliers are roughly Gaussian and elliptical. Tabular BFSI features
  — incomes, balances, counts — are skewed and heavy-tailed, so the assumption
  is wrong before you start.

Note how much of that reasoning came from the *contract* — which methods exist —
rather than from any benchmark. That is Module 00's whole thesis, used in
anger.""")

# %% [markdown]
# ---
# ## Where to go next
#
# If 0.4 landed, the thing to carry forward is the **positive control**: when a
# measurement says "no effect", point the same instrument at a case where the
# effect is known to be large. It costs five minutes, and it is the difference
# between "I think my code is fine" and "I know my code is fine".
#
# You will want it again almost immediately:
#
# - **Module 03** builds the other end of this range — the same mistake, with a
#   transformer that can learn a lot, reaching AUC 0.82 on pure noise.
# - **Module 04** is the one to slow down on. It is where a wrong split moves a
#   metric by twenty points with nothing to warn you, and it is the module
#   everything after it depends on.
#
# One habit from this notebook worth keeping permanently: **when you can pair,
# pair.** Two arms, same seed, report the paired difference. It shrinks your
# error bars for free and it isolates the thing you actually changed.
