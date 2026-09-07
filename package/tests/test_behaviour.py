"""Behaviour the conformance suite does not check."""

import numpy as np
import pytest
from sklearn.base import clone

from skcredit import TimeSeriesGroupSplit, Winsorizer


def test_fit_resets_state():
    """Refit on different data must equal a fresh fit — see Module 11.1."""
    r = np.random.default_rng(0)
    A, B = r.normal(0, 1, (120, 4)), r.normal(5, 1, (120, 4))
    contaminated = clone(Winsorizer()).fit(A).fit(B).transform(B)
    fresh = clone(Winsorizer()).fit(B).transform(B)
    np.testing.assert_allclose(contaminated, fresh)


def test_invalid_percentiles_raise_in_fit_not_init():
    """Validation belongs in fit, so clone() of a bad config still constructs."""
    est = Winsorizer(lower=0.9, upper=0.1)     # must NOT raise here
    clone(est)                                  # nor here
    with pytest.raises(ValueError, match="lower < upper"):
        est.fit(np.zeros((5, 2)))


def test_nan_passthrough():
    X = np.array([[1.0], [np.nan], [3.0], [100.0]])
    out = Winsorizer(lower=0.0, upper=0.75).fit(X).transform(X)
    assert np.isnan(out[1, 0])


def test_indicator_columns_and_names():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [100.0, 4.0]])
    w = Winsorizer(lower=0.0, upper=0.75, add_indicator=True).fit(X)
    assert w.transform(X).shape == (4, 4)
    assert list(w.get_feature_names_out(["a", "b"])) == ["a", "b", "a_clipped", "b_clipped"]


def test_splitter_never_shares_groups():
    X = np.arange(200).reshape(-1, 1)
    groups = np.repeat(np.arange(50), 4)
    for tr, te in TimeSeriesGroupSplit(n_splits=4).split(X, groups=groups):
        assert not set(groups[tr]) & set(groups[te])
        assert tr.max() < te.min()          # train is strictly in the past


def test_splitter_requires_groups():
    with pytest.raises(ValueError, match="requires"):
        list(TimeSeriesGroupSplit().split(np.zeros((10, 1))))
