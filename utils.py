"""Utilities modules."""

import numba
import numpy as np
import seaborn as sns
from scipy import special
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification, fetch_openml
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state, multiclass


RATIO_TEST_TRAIN = 3


def is_container(obj):
    """Simple container type test."""
    return isinstance(obj, (list, set, tuple, dict, range, bytearray, bytes)) or hasattr(obj, "__array__")


def args_2_str(args):
    """Parse and conver to a single string the 'argparse' arguments."""
    return '___'.join([arg_name + '_' +  '_'.join([str(v) for v in val])
                       if is_container(val) else arg_name + '_' +  str(val)
                       for arg_name, val in vars(args).items()])


def format_duration(seconds):
    """Converts a duration in seconds to HH:MM:SS format."""
    s, ns = divmod(seconds, 1)
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s {int(1000.0 * ns):03d}ms"


def type_1_error(pred, y, signed_target=False):
    if signed_target:
        y = check_signed_target(y)
    return np.mean(pred != y)


@numba.njit
def compute_delta_Qbar(l_Sigma, l_mu, n_per_class, lbda, tol=1e-6, n_iter=1000):
    """Find the fix point."""
    d = len(l_Sigma[0])
    lbda_Id = lbda * np.eye(d)

    n = np.sum(n_per_class)
    c1, c2 = n_per_class / np.sum(n_per_class)

    mu_1, mu_2 = l_mu[0].reshape((-1, 1)), l_mu[1].reshape((-1, 1))
    C1, C2 = l_Sigma[0] + mu_1 @ mu_1.T, l_Sigma[1] + mu_2 @ mu_2.T

    delta, delta_old = np.zeros((2, )), np.ones((2, ))
    for _ in range(n_iter):

        XXt = c1 * C1 / (1 + delta[0]) + c2 * C2 / (1 + delta[1])
        Qbar = np.linalg.inv(XXt + lbda_Id)
        delta = np.array([np.trace(C1 @ Qbar) / n,
                          np.trace(C2 @ Qbar) / n,
                          ])

        if np.linalg.norm(delta - delta_old) < tol:
            break

        delta_old = delta.copy()

    return delta, Qbar


@numba.njit
def compute_QCQ(Qbar, Sigma_1, C1, C2, n_per_class, delta):
    """Compute QCQ."""
    I2 = np.eye(2)
    n = np.sum(n_per_class)
    c1, c2 = n_per_class / np.sum(n_per_class)

    Vtilde = np.array([[np.trace(C1 @ Qbar @ C1 @ Qbar) / n,
                        np.trace(C1 @ Qbar @ C2 @ Qbar) / n],
                       [np.trace(C2 @ Qbar @ C1 @ Qbar) / n,
                        np.trace(C2 @ Qbar @ C2 @ Qbar) / n],
                       ])

    Tbar = np.array([np.trace(C1 @ Qbar @ Sigma_1 @ Qbar) / n,
                     np.trace(C2 @ Qbar @ Sigma_1 @ Qbar) / n,
                     ])

    Atilde = np.diag(np.array([c1 / (1 + delta[0])**2,
                               c2 / (1 + delta[1])**2,
                               ]))

    d1, d2 = np.linalg.inv(I2 - Vtilde @ Atilde) @ Tbar

    QCQbar = Qbar @ Sigma_1 @ Qbar
    QCQbar += (c1 * d1 / (1 + delta[0])**2) * Qbar @ C1 @ Qbar
    QCQbar += (c2 * d2 / (1 + delta[1])**2) * Qbar @ C2 @ Qbar

    return QCQbar


@numba.njit
def compute_J(n_per_class):
    """Compute the J matrix."""
    J = np.zeros((np.sum(n_per_class), 2))

    J[:n_per_class[0], 0] = np.ones((n_per_class[0], ))
    J[n_per_class[0]:, 1] = np.ones((n_per_class[1], ))

    return J


def compute_th_type_1_err(mean_pred_per_class, variance):
    """Compute the theoretical classification type 1 error."""
    z = np.abs(mean_pred_per_class[0] - mean_pred_per_class[1])
    z /= (2.0 * np.sqrt(2.0 * variance))
    return 0.5 * special.erfc(z)


def theoretical_error(l_Sigma, l_mu, n_per_class, m, lbda):
    """Theoretical errors."""
    y = np.hstack([c * np.ones((n,)) for c, n in zip([1, -1], n_per_class)])
    n = np.sum(n_per_class)

    mu_1, mu_2 = l_mu[0].reshape((-1, 1)), l_mu[1].reshape((-1, 1))
    C1, C2 = l_Sigma[0] + mu_1 @ mu_1.T, l_Sigma[1] + mu_2 @ mu_2.T

    J = compute_J(n_per_class)
    delta, Qbar = compute_delta_Qbar(l_Sigma, l_mu, n_per_class, lbda)
    QCQ1 = compute_QCQ(Qbar, l_Sigma[0], C1, C2, n_per_class, delta)

    deltap = [np.trace(l_Sigma[0] @ QCQ1) / n,
              np.trace(l_Sigma[1] @ QCQ1) / n,
              ]
    M_delta = np.diag(1 / (1 + delta)) @ l_mu
    M_deltap = np.diag(deltap / (1 + delta)**2) @ l_mu

    mean_pred_per_class = y.T @ J @ M_delta @ Qbar @ l_mu.T

    v = np.hstack([np.trace(l_Sigma[0] @ QCQ1) / ((1 + delta[0])**2) * np.ones((n_per_class[0],)),
                   np.trace(l_Sigma[1] @ QCQ1) / ((1 + delta[1])**2) * np.ones((n_per_class[1],)),
                   ])

    term_1 = y.T @ np.diag(v) @ y
    term_2 = y.T @ J @ M_delta @ QCQ1 @ M_delta.T @ J.T @ y
    term_3 = y.T @ J @ M_deltap @ Qbar @ M_delta.T @ J.T @ y
    term_4 = y.T @ J @ M_delta @ Qbar @ l_Sigma[0] @ Qbar @ M_delta.T @ J.T @ y

    var_pred_class_1 = term_1 / m + term_2 / m - 2 * term_3 / m + m * (m - 1) * term_4 / m**2

    return compute_th_type_1_err(mean_pred_per_class, var_pred_class_1)


def fetch_datasets(n_samples=1000):
    """Return a dict of binary 7 classification datatset."""
    datasets = {}

    # Synthetic-simple
    datasets['Synthetic\nsimple'] = make_classification(n_samples=n_samples, flip_y=0.0)

    # Synthetic-hard
    datasets['Synthetic\nhard'] = make_classification(n_samples=n_samples, flip_y=0.2)

    # Breast Cancer
    datasets['Breast\nCancer'] = load_breast_cancer(return_X_y=True)

    # Bank Diabetes
    data = fetch_openml(name="diabetes", version=1, as_frame=True)
    datasets['Diabetes'] = (np.array(data.data)[:n_samples, :],
                            np.array(data.target.map({'tested_negative': 0, 'tested_positive': 1}))[:n_samples])

    # Ionosphere
    data = fetch_openml(name="ionosphere", version=1, as_frame=True)
    datasets['Ionosphere'] = (np.array(data.data)[:n_samples, :],
                              np.array(data.target.map({'b': 0, 'g': 1}))[:n_samples])

    # Titanic
    data = sns.load_dataset("titanic")
    datasets['Titanic'] = (np.array(data[["age", "fare"]].fillna(0))[:n_samples, :], np.array(data["survived"])[:n_samples])

    # Spams
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    columns = [f"Feature{i}" for i in range(1, 58)] + ["Spam"]
    data = pd.read_csv(url, names=columns)
    datasets['Spams'] = (np.array(data.iloc[:, :-1])[:n_samples, :], np.array(data["Spam"])[:n_samples])

    return datasets


def make_classif(n_samples_train_per_class, l_Sigma, l_mu, random_state=None):
    """Return a simple binary classification dataset."""
    assert len(l_mu) == 2
    assert len(l_Sigma) == 2

    random_state = check_random_state(random_state)
    n_per_class = RATIO_TEST_TRAIN * n_samples_train_per_class + n_samples_train_per_class

    l_X, l_y = [], []
    for mu, sigma, label in zip(l_mu, l_Sigma, [-1, 1]):
        l_X.append(random_state.multivariate_normal(mu, sigma, size=n_per_class))
        l_y.append(label * np.ones(n_per_class))

    X_train = np.r_[l_X[0][:n_samples_train_per_class, :], l_X[1][:n_samples_train_per_class, :]]
    X_test = np.r_[l_X[0][n_samples_train_per_class:, :], l_X[1][n_samples_train_per_class:, :]]
    y_train = np.r_[l_y[0][:n_samples_train_per_class], l_y[1][:n_samples_train_per_class]]
    y_test = np.r_[l_y[0][n_samples_train_per_class:], l_y[1][n_samples_train_per_class:]]

    return X_train, X_test, y_train, y_test


def check_signed_target(y):
    """Check that the binary target is -1/1."""
    if set(np.unique(y)) == {0, 1}:
        return np.where(y == 0, -1, 1)
    else:
        return y


class AveragingLinearBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_clfs=None, n_clfs_max=100, random_state=None):
        assert hasattr(base_model, 'alpha')

        self.n_clfs = n_clfs
        self.n_clfs_max = n_clfs_max

        self.base_model = base_model
        self.models_ = None
        self.lbda = self.base_model.alpha

        self.fitted_ = False
        self.classes_ = None

        self.random_state = check_random_state(random_state)

    def _get_opt_m(self, X, y):
        """Compute the optimal number of models ."""
        mask_1, mask_2 = y == -1, y == 1

        n_per_class = np.array([np.sum(mask_1), np.sum(mask_2)])
        l_Sigma = np.array([X[mask_1].T @ X[mask_1], X[mask_2].T @ X[mask_2]])
        l_mu = np.array([np.mean(X[mask_1], axis=0), np.mean(X[mask_2], axis=0)])

        kwargs = dict(l_Sigma=l_Sigma, l_mu=l_mu, lbda=self.lbda)
        th_error = [theoretical_error(m=m, n_per_class=n_per_class//m, **kwargs)
                    for m in np.arange(1, self.n_clfs_max + 1)]

        return np.argmax(th_error)

    def fit(self, X, y):
        self.fitted_ = True
        self.classes_ = multiclass.unique_labels(y)
        y = check_signed_target(y)

        if self.n_clfs is None:
            self.n_clfs = self._get_opt_m(X, y)

        self.models_ = [clone(self.base_model) for _ in range(self.n_clfs)]

        split_indices = np.array_split(self.random_state.permutation(np.arange(len(X))), self.n_clfs)
        self.fitted_models_ = [model.fit(X[idx, :], y[idx]) for idx, model in zip(split_indices, self.models_)]
        return self

    def predict(self, X):
        assert self.fitted_
        mean_pred = np.mean([model.predict(X) for model in self.fitted_models_], axis=0)
        return np.where(mean_pred >= 0, 1, -1)

    def score(self, X, y):
        assert self.fitted_
        y = check_signed_target(y)
        return type_1_error(self.predict(X), y)


class Ridge(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.fitted_ = False
        self.classes_ = None

    def fit(self, X, y):
        self.fitted_ = True

        y = check_signed_target(y)

        self.classes_ = multiclass.unique_labels(y)
        n, d = X.shape

        if self.fit_intercept:
            X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
            X, y = X - X_mean, y - y_mean

        self.coef_ = np.linalg.solve(X.T @ X / n + self.alpha * np.eye(d), X.T @ y)
        self.intercept_ = 0.0

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_

        return self

    def predict(self, X):
        assert self.fitted_
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        assert self.fitted_
        y = check_signed_target(y)
        return type_1_error(self.predict(X), y)
