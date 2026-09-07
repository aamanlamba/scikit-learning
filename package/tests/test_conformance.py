"""The whole conformance suite, in four lines."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from skcredit import Winsorizer


@parametrize_with_checks([Winsorizer(), Winsorizer(add_indicator=True)])
def test_sklearn_compatible(estimator, check):
    check(estimator)
