import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
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
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        self.joint_gmm_ = GaussianMixture(
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

        self.joint_gmm_.fit(np.hstack((X, y)))

        covYX = self.joint_gmm_.covariances_[:, id_y, id_X]
        precXX = np.einsum("klm,knm->kln",
            self.joint_gmm_.precisions_cholesky_[:, id_X, id_X],
            self.joint_gmm_.precisions_cholesky_[:, id_X, id_X],
        )

        self.coef_ = np.einsum("klm,knm->kln", covYX, precXX)
        self.intercept_ = self.joint_gmm_.means_[:, id_y] - np.einsum(
            "klm,km->kl", self.coef_, self.joint_gmm_.means_[:, id_X]
        )

        return self

    def predict(self, X):
        """
        Predict posterior mean.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: Returns posterior mean, array-like, shape=(n_samples, n_columns_y).
        """

        weights, posterior_means = self.condition(X, covariances=False)

        return (posterior_means * weights[:, np.newaxis]).sum(axis=0).T

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: Returns the probability each Gaussian (state) in the model given each sample.
        """
        check_is_fitted(self, ["joint_gmm_", "coef_", "intercept_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        id_X = slice(0, X.shape[1])

        # evaluate weights based on N(X|mean_x,sigma_x) for each component
        marginal_x_gmm = GaussianMixture(n_components=self.n_components)
        marginal_x_gmm.weights_ = self.joint_gmm_.weights_
        marginal_x_gmm.means_ = self.joint_gmm_.means_[:, id_X]
        marginal_x_gmm.covariances_ = self.joint_gmm_.covariances_[:, id_X, id_X]
        marginal_x_gmm.precisions_ = self.joint_gmm_.precisions_[:, id_X, id_X]
        marginal_x_gmm.precisions_cholesky_ = self.joint_gmm_.precisions_cholesky_[:, id_X, id_X]

        return marginal_x_gmm.predict_proba(X)

    def condition(self, X, covariances=False):
        """
        Condition joint distribution on X.

        :param X: array-like, shape=(n_samples, n_columns) training data.
        :param covariances: bool, return posterior covariances? (default=False)
        :return: Returns posterior_weights, posterior_means, posterior_covariances
        """
        check_is_fitted(self, ["joint_gmm_", "coef_", "intercept_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        id_X = slice(0, X.shape[1])
        id_y = slice(X.shape[1], None)

        posterior_weights = self.predict_proba(X).T

        # posterior_means = mean_y + sigma_xx^-1 . sigma_xy . (x - mean_x)
        posterior_means = self.joint_gmm_.means_[:, id_y][:, :, np.newaxis] + np.einsum(
            "ijk,lik->ijl", self.coef_, (X[:, np.newaxis] - self.joint_gmm_.means_[:, id_X]),
        )

        if not covariances:
            return posterior_weights, posterior_means
        else:
            # posterior_covariances = sigma_yy - sigma_xx^-1 . sigma_xy .sigma_yx.T
            posterior_covariances = self.joint_gmm_.covariances_[:, id_y, id_y] - np.einsum(
                "klm,kmn->kln", self.coef_, self.joint_gmm_.covariances_[:, id_X, id_y]
            )

            return posterior_weights, posterior_means, posterior_covariances

    def sample(self, X, n_samples=1):
        """
        Sample conditional/posterior distribution given X.

        :param X: array-like, shape=(n_samples, n_columns) training data.
        :param covariances: bool, return posterior covariances? (default=False)
        :return: Returns samples, component labels
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        posterior_weights, posterior_means, posterior_covariances = self.condition(X, covariances=True)
        posterior_precisions_cholesky = _compute_precision_cholesky(posterior_covariances, self.joint_gmm_.covariance_type)
        posterior_precisions = np.einsum("klm,knm->kln", posterior_precisions_cholesky, posterior_precisions_cholesky)

        y = np.zeros((X.shape[0], n_samples, posterior_means.shape[1]))
        c = np.zeros((X.shape[0], n_samples))
        for ix, _ in enumerate(X):
            posterior_gmm = GaussianMixture(n_components=self.n_components)
            posterior_gmm.weights_ = posterior_weights[:, ix]
            posterior_gmm.means_ = posterior_means[:, :, ix]
            posterior_gmm.covariances_ = posterior_covariances
            posterior_gmm.precisions_ = posterior_precisions
            posterior_gmm.precisions_cholesky_ = posterior_precisions_cholesky

            y[ix], c[ix] = posterior_gmm.sample(n_samples)

        return y, c

    def aic(self, X, y):
        check_is_fitted(self, ["joint_gmm_", "coef_", "intercept_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if y.ndim == 1:
            y = np.expand_dims(y, 1)
        return self.joint_gmm_.aic(np.hstack((X, y)))

    def bic(self, X, y):
        check_is_fitted(self, ["joint_gmm_", "coef_", "intercept_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if y.ndim == 1:
            y = np.expand_dims(y, 1)
        return self.joint_gmm_.bic(np.hstack((X, y)))
