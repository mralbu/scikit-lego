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
    assert GMMRegressor(n_components=2, covariance_type='tied').fit(X, y).score(X, y) >= 0.9
    assert GMMRegressor(n_components=2, covariance_type='diag').fit(X, y).score(X, y) >= 0.9
    assert GMMRegressor(n_components=4, covariance_type='spherical').fit(X, y).score(X, y) >= 0.9

    assert BayesianGMMRegressor().fit(X, y).score(X, y) >= 0.99
    assert BayesianGMMRegressor(n_components=2, covariance_type='tied').fit(X, y).score(X, y) >= 0.9
    assert BayesianGMMRegressor(n_components=2, covariance_type='diag').fit(X, y).score(X, y) >= 0.9
    assert BayesianGMMRegressor(n_components=4, covariance_type='spherical').fit(X, y).score(X, y) >= 0.9

    for Regressor in [GMMRegressor, BayesianGMMRegressor]:
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            gmr = Regressor(n_components=2, covariance_type=covariance_type)
            gmr.fit(X, y)
            
            n_samples = 2
            yhat, components = gmr.sample(X, n_samples=2)

            assert yhat.shape == (y.shape[0], n_samples, y.shape[1])
            assert components.shape == (y.shape[0], n_samples)

    assert np.isreal(GMMRegressor(n_components=2).fit(X, y).aic(X, y))
    assert np.isreal(GMMRegressor(n_components=2).fit(X, y).bic(X, y))