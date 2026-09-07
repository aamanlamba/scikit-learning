"""Preprocessing transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted, validate_data

__all__ = ["Winsorizer"]


class Winsorizer(TransformerMixin, BaseEstimator):
    """Clip each feature to learned lower and upper percentiles.

    Parameters
    ----------
    lower : float, default=0.01
        Lower percentile, in [0, 1).
    upper : float, default=0.99
        Upper percentile, in (0, 1].
    add_indicator : bool, default=False
        If True, append one binary column per feature flagging clipped rows.

    Attributes
    ----------
    lower_bounds_ : ndarray of shape (n_features,)
        Learned lower clip points.
    upper_bounds_ : ndarray of shape (n_features,)
        Learned upper clip points.

    Examples
    --------
import numpy as np
from skcredit import Winsorizer
X = np.array([[1.0], [2.0], [3.0], [100.0]])
w = Winsorizer(lower=0.0, upper=0.75).fit(X)
float(w.upper_bounds_[0])
    3.0
w.transform(X).ravel().tolist()
    [1.0, 2.0, 3.0, 3.0]
    """

    def __init__(self, lower=0.01, upper=0.99, add_indicator=False):
        self.lower = lower
        self.upper = upper
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        """Learn the clip points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError(
                f"need 0 <= lower < upper <= 1, got {self.lower}, {self.upper}"
            )
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.nanpercentile(X, self.lower * 100, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X):
        """Clip `X` to the learned bounds."""
        check_is_fitted(self)
        X = validate_data(
            self, X, dtype=np.float64, reset=False, ensure_all_finite="allow-nan"
        )
        clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        if self.add_indicator:
            flags = ((X < self.lower_bounds_) | (X > self.upper_bounds_)).astype(np.float64)
            clipped = np.hstack([clipped, flags])
        return clipped

    def get_feature_names_out(self, input_features=None):
        """Output feature names."""
        names = _check_feature_names_in(self, input_features)
        if self.add_indicator:
            names = np.concatenate([names, [f"{n}_clipped" for n in names]])
        return np.asarray(names, dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
