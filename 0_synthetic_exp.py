#%%
"""Simple synthetic experiments."""

import os
import time
import numba
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import threadpoolctl
from bandpy.utils import format_duration

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}" r"\usepackage{amssymb}"
mpl.rcParams['figure.dpi'] = 400

t0_global = time.perf_counter()


###############################################################################
# Globals

def make_classif(n_samples, l_Sigma, l_mu, test_size=0.2, random_state=None):
    """Return a simple binary classification dataset."""
    assert len(l_mu) == 2
    assert len(l_Sigma) == 2

    random_state = check_random_state(random_state)

    d = len(l_mu[0])
    n_samples_per_class = int(n_samples / 2)

    l_X, l_y = [], []
    for mu, sigma, label in zip(l_mu, l_Sigma, [0, 1]):
        Z = random_state.randn(n_samples_per_class, d)
        l_X.append(mu + Z.dot(np.linalg.cholesky(sigma).T))
        l_y.append(label * np.ones(n_samples_per_class))

    X, y = np.r_[l_X[0], l_X[1]], np.r_[l_y[0], l_y[1]]

    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)


class ModelAveragingClassif(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_clfs=1, random_state=None):
        self.n_clfs = n_clfs
        self.models = [clone(base_model) for _ in range(self.n_clfs)]
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        split_indices = np.array_split(self.random_state.permutation(np.arange(len(X))), self.n_clfs)
        self.fitted_models_ = [model.fit(X[idx, :], y[idx]) for idx, model in zip(split_indices, self.models)]
        return self

    def predict(self, X):
        mean_pred = np.mean([model.predict(X) for model in self.fitted_models_], axis=0)
        return np.round(mean_pred).astype(int)


class LinearClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        Xb = self._add_bias(X)
        y = np.where(y == 0, -1, 1)
        alpha_Id = self.alpha * np.eye(X.shape[1] + 1)
        alpha_Id[0, 0] = 0
        self.w_ = np.linalg.solve(Xb.T @ Xb + alpha_Id, Xb.T @ y)
        return self

    def predict(self, X):
        return (self._add_bias(X) @ self.w_ >= 0).astype(int)


def run_one_scenario(n_clfs, n_samples, lbda, l_Sigma, l_mu, metric_func,
                     test_size=0.95, n_trials=100, random_state=None):
    """Run on experiment on the givne scenario."""
    n_samples = int(n_samples / (1.0 - test_size))
    base_model = LinearClassifier(alpha=lbda)

    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):

        l_acc = []
        for _ in range(n_trials):
            X_train, X_test, y_train, y_test = make_classif(n_samples, l_Sigma, l_mu, test_size=test_size)
            clf = ModelAveragingClassif(base_model=base_model, n_clfs=n_clfs, random_state=random_state)
            l_acc.append(metric_func(clf.fit(X_train, y_train).predict(X_test), y_test))

    return np.mean(l_acc), np.std(l_acc)


@numba.njit
def acc(pred, y):
    return np.mean(pred == y)


@numba.njit
def mse(pred, y):
    return 0.5 * np.mean(np.square(y - pred))


###############################################################################
# Main

if __name__ == '__main__':

    plot_dir = "figures__0_synthetic_exp"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    nrows, ncols = 1, 1
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols*3.0, nrows*3.0))

    fontsize = 11
    seed = 0
    random_state = check_random_state(seed)
    n_trials = 100
    verbose = 10
    n_jobs = 30

    metric_func = mse
    metric_func_name = r'$\mathbf{E}\left[\frac{1}{2}\left(\underset{m}{\sum} f_m(X) - Y\right)^2\right]$'
    # metric_func = acc
    # metric_func_name = r'$\mathbf{E}\left[1_{\{f_m(X) \neq Y\}}\right]$'

    d = 1000
    n = d
    test_size = 0.9
    l_lbda = [1e-6, 1e3, 1e6]
    colors = ['tab:orange', 'tab:blue']

    m_max = d - 1
    m_min = 1
    l_m = np.linspace(m_min, m_max, m_max - m_min, dtype=int)

    l_Sigma = [X @ X.T / np.linalg.norm(X @ X.T) for X in [random_state.randn(d, d) for _ in range(2)]]
    # l_Sigma = [np.eye(d) for _ in range(2)]

    beta = 2.0
    delta_mu = beta / np.sqrt(d) * np.max([np.linalg.eigvals(S) for S in l_Sigma])
    l_mu = [np.zeros(d), delta_mu * np.ones(d)]

    print("[Main] Main loop:")

    for color, lbda in zip(colors, l_lbda):

        print(f"[Main] lbda={lbda:.1e}...")

        t0 = time.perf_counter()

        l_mean_acc, l_std_acc = zip(*Parallel(verbose=verbose, n_jobs=n_jobs)(delayed(run_one_scenario)(
                                                    n_clfs=m, n_samples=n, lbda=lbda, l_Sigma=l_Sigma,
                                                    l_mu=l_mu, test_size=test_size, metric_func=metric_func,
                                                    random_state=random_state, n_trials=n_trials)
                                                    for m in l_m))

        mean_acc = np.array(l_mean_acc)
        std_acc = np.array(l_std_acc)

        axis[0, 0].plot(l_m, mean_acc, color=color, label=r'$\lambda=' + f'{lbda:.1e}' + r'$', lw=1.0, alpha=0.75)
        axis[0, 0].fill_between(l_m, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.1)

    axis[0, 0].set_xlabel(r"$m$ ($d=$" + f"{d}" + r",$n=$" + f"{n}" + r")", fontsize=fontsize)
    axis[0, 0].set_ylabel(metric_func_name, fontsize=fontsize)
    axis[0, 0].legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize=int(0.75*fontsize))

    fig.tight_layout()

    plot_filename = os.path.join(plot_dir, 'score_evolution')
    fig.savefig(plot_filename + ".png", dpi=300)
    fig.savefig(plot_filename + ".pdf", dpi=300)
    fig.show()

###############################################################################
# Timing
print(f"[Main] Experiment runtime: {format_duration(time.perf_counter() - t0_global)}")

#%%
