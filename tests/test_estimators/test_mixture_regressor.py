import numpy as np
import pytest

from sklego.common import flatten
from sklego.mixture import GMMRegressor, BayesianGMMRegressor
from tests.conftest import general_checks, nonmeta_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_non_transformer_estimators_n_iter",
        ],
    ),
)
def test_estimator_checks(test_fn):
    reg = GMMRegressor()
    test_fn(GMMRegressor.__name__, reg)
    reg = BayesianGMMRegressor()
    test_fn(BayesianGMMRegressor.__name__, reg)


def test_obvious_usecase():
    X = np.concatenate(
        [np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))]
    )
    y = 2 * X + 1
    assert GMMRegressor().fit(X, y).score(X, y) >= 0.99
    assert BayesianGMMRegressor().fit(X, y).score(X, y) >= 0.99
