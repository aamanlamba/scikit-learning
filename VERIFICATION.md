# Verification record

Every numeric claim in this curriculum's prose is supposed to be backed by output
in the notebook that makes it. This file records the audits that have been run
against that promise, and — more usefully — the claims that failed one.

It exists because a curriculum that spends fifteen modules on measuring things
honestly should be able to show its own working.

---

## Audit — full rebuild and cross-reference sweep

**Scope.** All 17 notebooks and 16 solution sets rebuilt and executed from
source in one pass. Every prose claim that cites another module by number *and*
carries a figure (167 of them) extracted and checked against executed output.
Both test suites run.

**Result.** 33 of 33 notebooks execute clean. 107 tests pass (98 in
`package/`, 9 in `service/`). Seven claims were wrong and are corrected below.

### 1. Notebooks 00–07 had been executed against a superseded dataset

`data/card_fraud.csv` was regenerated at 01:06; notebooks 00–05 carried outputs
timestamped 00:54–01:10. Module 04's shipped output reported a **0.547% fraud
rate** where the dataset holds **0.695%**, and consequently reported the
grouped/random split gap as **−8.2%**.

Re-executed against the shipped data, the same cell reports **+34.0%** — which
is the figure the README, the syllabus and three modules had been citing all
along. **The prose was right and the notebook was stale**, which is the more
dangerous direction: a reader checking the claim against the shipped output
would have concluded the prose was invented.

*Fix:* full rebuild. *Standing control:* a rebuild after any change to
`src/skmastery/datasets.py`, not just after a change to a notebook.

### 2. Module 04 and Module 14 appeared to contradict each other

Module 04 measures the random-split inflation on the fraud data at **+34%**;
exercise 14.1 measures **+8.8%** on the same dataset with the same estimator
family. Both are correct: **Module 04 deliberately leaves `card_id` in the
feature matrix** and 14.1 removes it, so the two measure different leaks — the
identity of compromised cards, versus the shared structure among one card's
transactions.

*Fix:* 14.1 now carries an explicit reconciliation. The generalisable point —
that leakage magnitude is a property of the feature set as much as of the split
— is worth more than either number.

### 3. Module 03's leakage figure was understated in two citations

Module 03 measures feature-selection-before-CV at **AUC 0.818** on pure noise
(0.825 at 3,000 features). Two solutions cited it as "0.5 → 0.75".

*Fix:* both corrected to 0.82.

### 4. Module 07 compared an AP spread against an AUC noise floor

The Pareto tolerance was justified by citing "the fold-to-fold noise measured in
Module 04 (±0.03)". Module 04's noise floor is **ROC AUC** (sd 0.0135 over 30
resamples); Module 07 optimises **average precision**. The two do not share a
scale — measured in place, the AP noise floor for the same estimator on the same
data is **sd 0.0234**, nearly double.

*Fix:* Module 07 now measures its own noise floor and derives the tolerance from
it. The exercise is stronger for it: the selected model gives up less than one
standard deviation of AP.

### 5. Module 07's Pareto front is not reproducible, and did not say so

Its second objective is a wall-clock timing, so the front moves with machine
load. The selected knee, the saving and the front's membership all changed
between two runs on the same container.

*Fix:* stated in the module. **Any objective that includes a timing is a noisy
objective**, and a multi-objective study built on one inherits that noise.

### 6. The "up to 40×" threading figure was never measured

Four places cited a 40× oversubscription penalty. Its origin was a code comment
in Module 06 recording a development observation that no notebook had ever
produced.

*Fix:* Module 15 now measures it — **2.3–2.6× on this two-core container** — and
says plainly that the number depends on core count and nesting depth, so you
should quote the one you measured. The unverified figure is removed from the
cheat sheet, from Module 15's tables and from the syllabus.

This was the worst of the seven. An unrecorded number had propagated into a
reference card, which is exactly the failure Module 12.5 is about.

### 7. Six syllabus findings were attributed to the wrong module

The renamed-column `ValueError` is Module 00's, not 01's. The float32 precision
result is Module 15's, not 05's. Four others were restatements of a neighbouring
module's work.

*Fix:* every syllabus finding now names a result produced by the module it sits
under.

---

## Claims re-confirmed against fresh output

| Claim | Where | Measured |
|---|---|---|
| Random split inflates fraud AP | 04 | **+34.0%** |
| Naive target encoding on a noise ID column | 02 | **AUC 1.000** in-sample, 0.500 cross-fitted |
| Feature selection before CV on pure noise | 03 | **0.818** vs 0.469 inside the pipeline |
| float32 coefficient error at cond 2e4 | 15 | **74×** float64's |
| Monotonic constraint's cost | 06 | **−0.009 AUC**, i.e. a gain |
| Gower + agglomerative cluster sizes | 09 | **1492 / 2 / 2 / 4** |
| Ticket error decomposition vs the generator | 10 | 12.40% observed vs **12.52% expected** ambiguity; 3.27% vs 3.50% label noise |
| Routing: the LLM tier is squeezed out | 10.5 | 83.7% auto / **0% LLM** / 16.3% human |
| A no-reset-on-refit estimator | 11 | passes **47 of 47** checks |
| Selection-rate ratio moves on threshold alone | 12 | **0.41 → 0.96**, one model |
| Reliance concentration, leaky model | 12.5 | **92%** on one feature |
| Corrupted feed under the stated cost matrix | 13.5 | **cheaper** by ~£21k/month |
| Capstone winner across 8 configurations | 14.4 | wins **8/8**; margin £4.12 vs £15.03 spread |
| Tree building's share of the search profile | 15 | **69%** |
| Estimators with Array API support | cheat 07 | exactly **13**, list matches |
| Dataset shapes and rates | README | all five confirmed |

---

## How to re-run this audit

```bash
python build.py                                   # rebuild + execute everything
cd package  && PYTHONPATH=$PWD/src pytest -q      # 98 tests
cd ../service && PYTHONPATH=$PWD pytest -q        #  9 tests
```

Then check that no prose cites a number the notebooks do not produce. The
extraction that found these seven:

```python
import re, glob
mod = re.compile(r"Module\s+\d{2}", re.I)
num = re.compile(r"\d+(?:\.\d+)?\s*(?:%|x\b|×)|£[\d,]+|\b0\.\d{2,4}\b")
for f in glob.glob("build/*.py"):
    for i, line in enumerate(open(f), 1):
        if line.lstrip().startswith("#") and mod.search(line) and num.search(line):
            print(f"{f}:{i}: {line.strip()}")
```

**The check that matters is not the regex — it is opening the cited notebook and
reading the number.** Four of the seven errors above would have passed any
automated consistency check, because the citing prose and the cited notebook
were each internally consistent and disagreed only with reality.
