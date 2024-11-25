"""Synthetic data experiments."""

import os
import time
import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from scipy import linalg
from sklearn.utils import check_random_state
import threadpoolctl
from utils import (format_duration, args_2_str, Ridge, AveragingLinearBinaryClassifier,
                   make_classif, theoretical_error, type_1_error)


mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}" r"\usepackage{amssymb}"
mpl.rcParams['figure.dpi'] = 400


t0_global = time.perf_counter()


###############################################################################
# Globals
fontsize = 10
seed = 0
random_state = check_random_state(seed)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def empirical_error(n_clfs, lbda, err_func, X_train, X_test, y_train, y_test,
                    n_trials=100, random_state=None):
    """Run on experiment on the givne scenario."""
    base_model = Ridge(alpha=lbda)

    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):

        l_err = []
        for _ in range(n_trials):
            clf = AveragingLinearBinaryClassifier(base_model=base_model, n_clfs=n_clfs,
                                                  random_state=random_state)
            l_err.append(err_func(clf.fit(X_train, y_train).predict(X_test), y_test))

    return np.mean(l_err), np.std(l_err)


def get_sigma(d, cov_mat_type='identity', equal=True, high=1.0, random_state=None):

    random_state = check_random_state(random_state)

    if cov_mat_type == 'identity':
        Sigma = np.eye(d)
        return np.array([Sigma, Sigma])

    elif cov_mat_type == 'toeplitz':
        Sigma = linalg.toeplitz(0.0**np.arange(d))
        return np.array([Sigma, Sigma])

    elif cov_mat_type == 'diag_random':
        if equal:
            Sigma = random_state.uniform(low=0.0, high=high, size=d) * np.eye(d)
            return np.array([Sigma, Sigma])

        else:
            Sigma_1 = random_state.uniform(low=0.0, high=high, size=d) * np.eye(d)
            Sigma_2 = random_state.uniform(low=0.0, high=high, size=d) * np.eye(d)
            return np.array([Sigma_1, Sigma_2])

    elif cov_mat_type == 'full_random':
        if equal:
            X = random_state.randn(d, 5000)
            Sigma = X @ X.T / np.linalg.norm(X @ X.T)
            return np.array([Sigma, Sigma])

        else:
            X_1 = random_state.randn(d, 500)
            Sigma_1 = X_1 @ X_1.T / np.linalg.norm(X_1 @ X_1.T)
            X_2 = random_state.randn(d, 5000)
            Sigma_2 = X_2 @ X_2.T / np.linalg.norm(X_2 @ X_2.T)
            return np.array([Sigma_1, Sigma_2])

    else:
        raise ValueError(f"'cov_mat_type' not understood, got {cov_mat_type}")


def e_(i, d):
    """Return the first canonical vector."""
    e_ = np.zeros((d,))
    e_[i-1] = 1.0
    return e_


###############################################################################
# Main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, help="Dimension.", default=100)
    parser.add_argument('--n-samples', type=int, help="Nb of train samples for each class.", default=500)
    parser.add_argument('--m-max', type=int, help="Maximal number of models to ensemble.", default=100)
    parser.add_argument('--l-lbda', type=float, nargs="+", help="List of the regularization (e.g., --l-lbda 0.1 0.01 0.001).", default=[1e-1, 1e1])
    parser.add_argument('--cov-mat-type', type=str, help="Type of covariance matrix.", default='identity')
    parser.add_argument('--n-trials', type=int, help="Nb of trials.", default=10)
    parser.add_argument('--n-jobs', type=int, help="Nb of CPUs.", default=1)
    args = parser.parse_args()

    main_figures_dir = "figures__0_synthetic_exp"
    if not os.path.isdir(main_figures_dir):
        os.makedirs(main_figures_dir)

    sub_figure_dir = os.path.join(main_figures_dir, 'figure____' + args_2_str(args))
    if not os.path.isdir(sub_figure_dir):
        os.makedirs(sub_figure_dir)

    n_per_class = np.array([args.n_samples, args.n_samples])
    l_m = np.arange(1, args.m_max + 1)

    l_mu = np.array([0.9 * e_(1, args.dim), -0.9 * e_(1, args.dim)])
    l_Sigma = get_sigma(args.dim, cov_mat_type=args.cov_mat_type, equal=True, random_state=random_state)

    X_train, X_test, y_train, y_test = make_classif(args.n_samples, l_Sigma, l_mu)

    ###########################################################################
    # Main loop

    nrows, ncols = len(args.l_lbda), 1
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             figsize=(ncols*2.75, nrows*2.5))

    for i in tqdm.tqdm(range(len(args.l_lbda)), desc="[Main] Main loop"):

        color, lbda = colors[i], args.l_lbda[i]

        # empirical results
        kwargs = dict(lbda=lbda, err_func=type_1_error, X_train=X_train, X_test=X_test,
                      y_train=y_train, y_test=y_test, random_state=random_state,
                      n_trials=args.n_trials)
        delayed_runs = (delayed(empirical_error)(n_clfs=m, **kwargs) for m in l_m)
        l_mean_err, l_std_err = zip(*Parallel(n_jobs=args.n_jobs)(delayed_runs))
        mean_err = np.array(l_mean_err)
        std_err = np.array(l_std_err)

        # theoretical results
        kwargs = dict(l_Sigma=l_Sigma, l_mu=l_mu, lbda=lbda)
        th_err = [theoretical_error(m=m, n_per_class=n_per_class//m, **kwargs) for m in l_m]

        label = r'$\lambda=' + f'{lbda:.2f}' + r'$ ($m^*=' + f'{np.argmin(mean_err)}'
        label += r'$, $\mathrm{err}^*=' + f'{np.min(mean_err):.3f}' + r'$)'
        axis[i, 0].plot(l_m, mean_err, linestyle='solid', color=color, label=label, lw=1.0, alpha=0.75)
        axis[i, 0].plot(l_m, th_err, linestyle='dashed', color='tab:gray', lw=1.0, alpha=0.75)
        axis[i, 0].fill_between(l_m, mean_err - std_err, mean_err + std_err, color=color, alpha=0.1)

        xlabel = r"\begin{center}$m$\\$d=$" + f"{args.dim}" + r", $n^{\mathrm{train}}_{\mathrm{class}}=$" + f"{args.n_samples}"
        xlabel += ", $\Sigma=$" + f"'{args.cov_mat_type}'" + r"\end{center}"
        axis[i, 0].set_xlabel(xlabel, fontsize=fontsize)
        ylabel = r'$\mathrm{err} = \mathbf{E}\left[1_{\{f_m(X) \neq Y\}}\right]$'
        axis[i, 0].set_ylabel(ylabel, fontsize=fontsize)
        axis[i, 0].legend(ncol=1, loc='lower center', bbox_to_anchor=(0.35, 1.1),
                        fontsize=int(0.8*fontsize))

    fig.tight_layout()

    figure_filename = os.path.join(sub_figure_dir, 'score_evolution')
    fig.savefig(figure_filename + ".png", dpi=300)
    fig.savefig(figure_filename + ".pdf", dpi=300)

###############################################################################
# Timing
print(f"[Main] Experiment runtime: {format_duration(time.perf_counter() - t0_global)}")
