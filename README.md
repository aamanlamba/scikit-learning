# scikit-learn, properly

A self-paced curriculum for going from "I've used it in notebooks" to fluent —
able to build, evaluate, defend and critique tabular ML work from a blank file.

Built for scikit-learn **1.8**, Python 3.11+.

---

## How this is organised

```
scikit-learning/
├── notebooks/       the modules — run these, in order
├── guided/          stuck? hints that unfold, plus self-checks, before the answer
├── solutions/       worked answers to every exercise
├── cheatsheets/     reference cards to keep open while you work
├── data/            generated BFSI datasets (regenerate with make_data.py)
├── package/         skcredit — the installable package built in exercise 11.5
├── service/         the FastAPI scoring service built in exercise 14.5
├── artifacts/       model bundles and drift dashboards written by module 13
├── src/skmastery/   shared helpers and dataset generators
├── build/           jupytext sources the notebooks are generated from
└── build.py         regenerate + execute all notebooks
```

Each module is a runnable notebook containing worked material, then a set of
exercises at the end with empty cells. The matching file in `solutions/` has
worked answers. Resist it until you have something that runs.

**If you get stuck**, `guided/` sits between the two: for each exercise it gives
you what the exercise is really testing, a question to answer before writing any
code, three hints behind click-to-open blocks, and a **self-check function** you
run against your own attempt. The answer is last, also collapsed. Module 00 has
one; ask for others as you need them.

> **Status.** Complete. The orientation chapter, all 16 module notebooks, all 16
> worked solution sets and all 7 cheat sheets execute end to end against
> scikit-learn 1.8.
>
> The solutions are written as worked arguments, not answer keys. Several reach
> the *opposite* conclusion to the one the exercise implies, because that is
> what the data said — every numeric claim in them was checked against actual
> notebook output and the prose rewritten where reality disagreed.
>
> **Module 15 is hardware-dependent by design.** It detects what accelerator you
> have and adapts. It was built in a CPU-only container, so the GPU columns are
> empty in the shipped output — run it on your Mac to fill them in.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

The notebooks add `src/` to `sys.path` themselves, so no install step is needed.
If you prefer, `pip install -e .` also works.

The bundled CSVs are generated, not collected — regenerate them any time with:

```bash
python -m skmastery.datasets      # or: python make_data.py
```

A few modules also pull real datasets (`fetch_california_housing`,
`fetch_20newsgroups`, `fetch_openml`). Those need network access on first run
and then cache to `~/scikit_learn_data`.

---

## The curriculum

### Before you start

| # | Module | The idea it exists to install |
|---|---|---|
| — | [Start here](notebooks/000_start_here.ipynb) | numpy, pandas, scipy and where scikit-learn sits — including why any of it still matters in an LLM world. |

### Part I — Foundations

| # | Module | The idea it exists to install |
|---|---|---|
| 00 | The Estimator Contract | scikit-learn is one interface implemented ~200 times. Learn the interface. |
| 01 | Data Representation | Most errors are shape, dtype and target-type errors, before any learning happens. |
| 02 | Transformers | Scaling is geometry, not tidiness. |
| 03 | Pipelines and Composition | The pipeline is a correctness mechanism, not a convenience. |
| 04 | Validation Strategy | A wrong split moves your metric by twenty points, invisibly. |
| 05 | Linear Models | Build OLS and logistic regression from scratch; everything else stops being mysterious. |

### Part II — The model families

| # | Module | The idea it exists to install |
|---|---|---|
| 06 | Trees and Ensembles | Bagging and boosting are *opposite* arguments about bias and variance. |
| 07 | Hyperparameter Search | Measure the size of the prize before spending the budget. |
| 08 | Metrics, Calibration, Thresholds | The model outputs a score; the business needs a decision. Those are different problems. |
| 09 | Unsupervised Learning | Clustering always returns clusters. Whether they mean anything is your job. |
| 10 | Text, Features and Embeddings | Where classical feature extraction ends and the LLM era begins. |

### Part III — Practice and performance

| # | Module | The idea it exists to install |
|---|---|---|
| 11 | Extending scikit-learn | Custom estimators that pass `check_estimator` and compose properly. |
| 12 | Interpretability and Fairness | Explanations that survive a model-risk committee. |
| 13 | Production | Persistence, drift, monitoring, and what breaks between notebook and service. |
| 14 | Capstone | End-to-end BFSI credit decisioning, from raw table to signed-off model. |
| 15 | GPU and Acceleration | What actually makes tabular ML fast — and why, on Apple Silicon, it isn't the GPU. |

---

## Datasets

All five are generated from a fixed seed by `src/skmastery/datasets.py`. Each one
was built to teach a specific failure mode rather than to be clean.

| Dataset | Rows | Target | What it is for |
|---|---|---|---|
| `credit_risk` | 12,000 | 13% default | The workhorse. MNAR missingness, a post-outcome leakage column, macro drift in the final year. |
| `card_fraud` | 60,000 | 0.7% fraud | Extreme imbalance; fraud concentrates on compromised cards, so a random split inflates the score ~34%. |
| `insurance_claims` | 15,000 | claim cost | 91% zeros, heavy tail. Squared error predicts negative claim costs; Tweedie does not. |
| `telco_churn` | 8,000 | 18% churn | Categorical-heavy, with the classic blank `total_charges` on brand-new accounts. |
| `support_tickets` | 6,000 | 6 classes | Short text with overlapping vocabulary and 5% label noise, so there is a real accuracy ceiling. |

No real customer data is involved. The causal structures are invented — treat
the coefficients as pedagogy, not as domain truth.

---

## How to actually work through it

1. **Run every cell.** Before reading a cell's output, guess it. The guessing is
   the learning; the running only confirms it.
2. **Do the exercises before opening the solutions.** They are where the
   material moves from recognisable to usable.
3. **Break things on purpose.** Most modules contain a "here is the wrong way,
   measured" section. Extend those.
4. **Keep the cheat sheets open.** They are meant to be used while working, not
   read once.

Modules 00–05 are sequential and worth doing in order. From 06 onwards you can
jump to what you need, though 08 (metrics and thresholds) is the one most people
benefit from most and skip most often.

---

## Regenerating the notebooks

The notebooks are generated from jupytext percent-format sources in `build/`,
which is what makes them diffable and reviewable.

```bash
python build.py            # convert all + execute
python build.py --no-exec  # convert only, fast
python build.py 08 12      # just those modules
```
