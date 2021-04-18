import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._bayesian_mixture import _compute_precision_cholesky
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
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        self.joint_gmm_ = BayesianGaussianMixture(
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

        self.joint_gmm_.fit(np.hstack((X, y)))

        covariances = _get_full_matrix(self.joint_gmm_.covariances_, self.covariance_type, self.n_components, self.joint_gmm_.n_features_in_)
        precisions_cholesky = _get_full_matrix(self.joint_gmm_.precisions_cholesky_, self.covariance_type, self.n_components, self.joint_gmm_.n_features_in_)

        covYX = covariances[:, id_y, id_X]
        precXX = np.einsum("klm,knm->kln", precisions_cholesky[:, id_X, id_X], precisions_cholesky[:, id_X, id_X])

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

        covariances = _get_full_matrix(self.joint_gmm_.covariances_, self.covariance_type, self.n_components, self.joint_gmm_.n_features_in_)
        precisions = _get_full_matrix(self.joint_gmm_.precisions_, self.covariance_type, self.n_components, self.joint_gmm_.n_features_in_)
        precisions_cholesky = _get_full_matrix(self.joint_gmm_.precisions_cholesky_, self.covariance_type, self.n_components, self.joint_gmm_.n_features_in_)

        # evaluate weights based on N(X|mean_x,sigma_x) for each component
        marginal_x_gmm = BayesianGaussianMixture(n_components=self.n_components)
        marginal_x_gmm.weights_ = self.joint_gmm_.weights_
        marginal_x_gmm.means_ = self.joint_gmm_.means_[:, id_X]
        marginal_x_gmm.covariances_ = covariances[:, id_X, id_X]
        marginal_x_gmm.precisions_ = precisions[:, id_X, id_X]
        marginal_x_gmm.precisions_cholesky_ = precisions_cholesky[:, id_X, id_X]
        marginal_x_gmm.degrees_of_freedom_ = self.joint_gmm_.degrees_of_freedom_
        marginal_x_gmm.mean_precision_ = self.joint_gmm_.mean_precision_
        marginal_x_gmm.weight_concentration_ = self.joint_gmm_.weight_concentration_
        marginal_x_gmm.mean_prior_ = self.joint_gmm_.mean_prior_[id_X]

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
            full_covariances = _get_full_matrix(self.joint_gmm_.covariances_, self.covariance_type, self.n_components, self.joint_gmm_.n_features_in_)

            # posterior_covariances = sigma_yy - sigma_xx^-1 . sigma_xy .sigma_yx.T
            posterior_covariances = full_covariances[:, id_y, id_y] - np.einsum(
                "klm,kmn->kln", self.coef_, full_covariances[:, id_X, id_y]
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
        posterior_precisions_cholesky = _compute_precision_cholesky(posterior_covariances, 'full')
        posterior_precisions = np.einsum("klm,knm->kln", posterior_precisions_cholesky, posterior_precisions_cholesky)

        y = np.zeros((X.shape[0], n_samples, posterior_means.shape[1]))
        c = np.zeros((X.shape[0], n_samples))
        for ix, _ in enumerate(X):
            posterior_gmm = BayesianGaussianMixture(n_components=self.n_components)
            posterior_gmm.weights_ = posterior_weights[:, ix]
            posterior_gmm.means_ = posterior_means[:, :, ix]
            posterior_gmm.covariances_ = posterior_covariances
            posterior_gmm.precisions_ = posterior_precisions
            posterior_gmm.precisions_cholesky_ = posterior_precisions_cholesky
            # posterior_gmm.degrees_of_freedom_ = self.joint_gmm_.degrees_of_freedom_
            # posterior_gmm.mean_precision_ = self.joint_gmm_.mean_precision_
            # posterior_gmm.weight_concentration_ = self.joint_gmm_.weight_concentration_
            # posterior_gmm.mean_prior_ = self.joint_gmm_.mean_prior_[id_X]

            y[ix], c[ix] = posterior_gmm.sample(n_samples)

        return y, c

def _get_full_matrix(m, covariance_type, n_components_, n_features_in_):
    if covariance_type == 'full':
        return m
    elif covariance_type == 'tied':
        return np.repeat(m[np.newaxis, :], n_components_, axis=0)
    elif covariance_type == 'diag':
        full_m = np.zeros((n_components_, n_features_in_, n_features_in_))
        for k in range(n_components_):
            full_m[k] = np.diag(m[k])
        return full_m
    elif covariance_type == 'spherical':
        full_m = np.zeros((n_components_, n_features_in_, n_features_in_))
        for k in range(n_components_):
            full_m[k] = np.diag(np.repeat(m[k], n_features_in_))
        return full_m