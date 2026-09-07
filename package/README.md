# skcredit

scikit-learn estimators for credit risk modelling. Every estimator passes
`sklearn.utils.estimator_checks.check_estimator`, so they compose with
`Pipeline`, `GridSearchCV`, `cross_validate` and `set_output` unchanged.

## Install

```bash
pip install -e ".[dev]"
```

## Use

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skcredit import TimeSeriesGroupSplit, Winsorizer

pipe = Pipeline([
    ("clip", Winsorizer(lower=0.01, upper=0.99, add_indicator=True)),
    ("lr", LogisticRegression(max_iter=2000)),
])

cv = TimeSeriesGroupSplit(n_splits=5)
# cross_validate(pipe, X, y, cv=cv, groups=customer_id)
```

`Winsorizer` clips each feature to learned percentiles and optionally appends a
flag per feature. `TimeSeriesGroupSplit` gives expanding-window splits in which
no group spans train and test — the splitter you need when the same customer
appears many times and time order matters.

## Develop

```bash
pytest -q            # conformance suite, behaviour tests and doctests
```

## Why these two

Both exist because the library has no equivalent and both encode a rule a credit
team already follows. That is the bar for adding an estimator: not "it would be
neat", but "the alternative is a copy-pasted helper in six notebooks".
