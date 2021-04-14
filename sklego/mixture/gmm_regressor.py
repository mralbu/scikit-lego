import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class GMMRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        """
        The GMMRegressor trains a Gaussian Mixture Model on a dataset containing both X and y columns.
        Predictions are evaluated conditioning the fitted Multivariate Gaussian Mixture on the known
        X variables. All parameters of the model are an exact copy of the parameters in scikit-learn.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.array, y: np.array) -> "GMMRegressor":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES, multi_output=True)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            weights_init=self.weights_init,
            means_init=self.means_init,
            precisions_init=self.precisions_init,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )

        id_X = slice(0, X.shape[1])
        id_y = slice(X.shape[1], None)

        self.gmm_.fit(np.hstack((X, y)))

        covYX = self.gmm_.covariances_[:, id_y, id_X]
        precXX = np.einsum(
            "klm,knm->kln",
            self.gmm_.precisions_cholesky_[:, id_X, id_X],
            self.gmm_.precisions_cholesky_[:, id_X, id_X],
        )
        
        self.coef_ = np.einsum("klm,knm->kln", covYX, precXX)
        self.intercept_ = self.gmm_.means_[:, id_y] - np.einsum(
            "klm,km->kl", self.coef_, self.gmm_.means_[:, id_X]
        )

        return self

    def predict(self, X):
        check_is_fitted(self, ["gmm_", "coef_", "intercept_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        id_X = slice(0, X.shape[1])
        id_y = slice(X.shape[1], None)

        # evaluate weights based on N(X|mean_x,sigma_x) for each component
        gmmX_ = GaussianMixture(n_components=self.n_components)
        gmmX_.weights_ = self.gmm_.weights_
        gmmX_.means_ = self.gmm_.means_[:, id_X]
        gmmX_.covariances_ = self.gmm_.covariances_[:, id_X, id_X]
        gmmX_.precisions_ = self.gmm_.precisions_[:, id_X, id_X]
        gmmX_.precisions_cholesky_ = self.gmm_.precisions_cholesky_[:, id_X, id_X]

        weights_ = gmmX_.predict_proba(X).T

        # posterior_means = mean_y + sigma_xx^-1 . sigma_xy . (x - mean_x)
        posterior_means = self.gmm_.means_[:, id_y][:, :, np.newaxis] + np.einsum(
            "ijk,lik->ijl", self.coef_, (X[:, np.newaxis] - self.gmm_.means_[:, id_X])
        )

        return (posterior_means * weights_[:, np.newaxis]).sum(axis=0).T
