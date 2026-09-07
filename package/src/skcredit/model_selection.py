"""Cross-validation splitters for grouped time series."""

import numpy as np

__all__ = ["TimeSeriesGroupSplit"]


class TimeSeriesGroupSplit:
    """Expanding-window splits in which no group spans train and test.

    Rows are assumed to be in time order. Any group straddling the cut point is
    assigned wholly to training, which can only shrink the test set and never
    leaks future information into it.

    Parameters
    ----------
    n_splits : int, default=5
    gap : int, default=0
        Rows dropped between the end of train and the start of test.

    Examples
    --------
import numpy as np
from skcredit import TimeSeriesGroupSplit
X = np.arange(20).reshape(-1, 1)
groups = np.repeat(np.arange(10), 2)
cv = TimeSeriesGroupSplit(n_splits=2)
for tr, te in cv.split(X, groups=groups):
    assert not (set(groups[tr]) & set(groups[te]))
cv.get_n_splits()
    2
    """

    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        """Number of splitting iterations."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate (train, test) index arrays."""
        if groups is None:
            raise ValueError("TimeSeriesGroupSplit requires `groups`")
        n = len(X)
        order = np.arange(n)
        groups = np.asarray(groups)
        fold = n // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            cut = i * fold
            train_idx = order[: cut - self.gap]
            test_idx = order[cut : cut + fold]
            test_idx = test_idx[~np.isin(groups[test_idx], list(set(groups[train_idx])))]
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx
