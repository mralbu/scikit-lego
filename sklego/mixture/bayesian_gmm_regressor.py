import numpy as np
from scipy.linalg import pinvh
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class BayesianGMMRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        """
        The BayesianGMMRegressor trains a Gaussian Mixture Model on a dataset containing both X and y columns.
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
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.array, y: np.array) -> "BayesianGMMRegressor":
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

        self.gmm_ = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            weight_concentration_prior_type=self.weight_concentration_prior_type,
            weight_concentration_prior=self.weight_concentration_prior,
            mean_precision_prior=self.mean_precision_prior,
            mean_prior=self.mean_prior,
            degrees_of_freedom_prior=self.degrees_of_freedom_prior,
            covariance_prior=self.covariance_prior,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )

        id_X = slice(0, X.shape[1])
        id_y = slice(X.shape[1], None)

        self.gmm_.fit(np.hstack((X, y)))
        self.intercept_ = np.zeros((self.n_components, y.shape[1]))
        self.coef_ = np.zeros((self.n_components, y.shape[1], X.shape[1]))
        for k in range(self.n_components):
            covYX = self.gmm_.covariances_[k, id_y, id_X]
            precXX = pinvh(self.gmm_.covariances_[k, id_X, id_X])
            # precXX = self.gmm_.precision[k, id_X, id_X]
            self.coef_[k] = covYX.dot(precXX)
            self.intercept_[k] = (
                self.gmm_.means_[k, id_y] - self.coef_[k].dot(self.gmm_.means_[k, id_X].T)
            )

        return self

    def predict(self, X):
        check_is_fitted(self, ["gmm_", "coef_", "intercept_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        id_X = slice(0, X.shape[1])
        id_y = slice(X.shape[1], None)

        # evaluate weights based on N(X|mean_x,sigma_x) for each component
        gmmX_ = BayesianGaussianMixture(n_components=self.n_components)
        gmmX_.weights_ = self.gmm_.weights_
        gmmX_.means_ = self.gmm_.means_[:, id_X]
        gmmX_.covariances_ = self.gmm_.covariances_[:, id_X, id_X]
        gmmX_.precisions_ = self.gmm_.precisions_[:, id_X, id_X]
        gmmX_.precisions_cholesky_ = self.gmm_.precisions_cholesky_[:, id_X, id_X]
        gmmX_.degrees_of_freedom_ = self.gmm_.degrees_of_freedom_
        gmmX_.mean_precision_ = self.gmm_.mean_precision_
        gmmX_.weight_concentration_ = self.gmm_.weight_concentration_
        gmmX_.mean_prior_ = self.gmm_.mean_prior_[id_X]

        weights_ = gmmX_.predict_proba(X).T

        # posterior_means = mean_y + sigma_xx^-1 . sigma_xy . (x - mean_x)
        posterior_means = self.gmm_.means_[:, id_y][:, :, np.newaxis] + np.einsum(
            "ijk,lik->ijl", self.coef_, (X[:, np.newaxis] - self.gmm_.means_[:, id_X])
        )

        return (posterior_means * weights_[:, np.newaxis]).sum(axis=0).T
