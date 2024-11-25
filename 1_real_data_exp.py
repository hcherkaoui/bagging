"""Real data experiments."""

import os
import time
import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from utils import (format_duration, args_2_str, Ridge, AveragingLinearBinaryClassifier,
                   type_1_error, fetch_datasets)


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

n_samples = 3000
m_max = 100
test_size = 0.4

markers = {"m=m_max": 'o',
           "m=1": 's',
           "Grid-search-m": '^',
           "Random-search-m": 'X',
           }

datasets = fetch_datasets(n_samples=n_samples)

def set_up_m_max(lbda, m_max, random_state, n_jobs):
    base_model = Ridge(alpha=lbda)
    clf = AveragingLinearBinaryClassifier(base_model=base_model, n_clfs=m_max, random_state=random_state)
    return markers["m=m_max"], "m=m_max", clf


def set_up_m_1(lbda, m_max, random_state, n_jobs):
    base_model = Ridge(alpha=lbda)
    clf = AveragingLinearBinaryClassifier(base_model=base_model, n_clfs=1, random_state=random_state)
    return markers["m=1"], "m=1", clf


def set_up_grid_search(lbda, m_max, random_state, n_jobs):
    base_model = Ridge(alpha=lbda)
    base_ensemble = AveragingLinearBinaryClassifier(base_model=base_model, random_state=random_state)
    clf = GridSearchCV(estimator=base_ensemble, param_grid={'n_clfs': range(1, m_max+1)}, cv=10,
                       scoring=make_scorer(type_1_error, greater_is_better=False, response_method='predict', signed_target=True),
                       n_jobs=n_jobs)
    return markers["Grid-search-m"], "Grid-search-m", clf


def set_up_random_search(lbda, m_max, random_state, n_jobs):
    base_model = Ridge(alpha=lbda)
    base_ensemble = AveragingLinearBinaryClassifier(base_model=base_model, random_state=random_state)
    clf = RandomizedSearchCV(estimator=base_ensemble, param_distributions={'n_clfs': range(1, m_max+1)}, n_iter=10, cv=10,
                             scoring=make_scorer(type_1_error, greater_is_better=False, response_method='predict', signed_target=True),
                             n_jobs=n_jobs)
    return markers["Random-search-m"], "Random-search-m", clf


###############################################################################
# Main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--l-lbda', type=float, nargs="+", help="List of the regularization (e.g., --l-lbda 0.1 0.01 0.001).", default=[1e-1, 1e1])
    parser.add_argument('--n-trials', type=int, help="Nb of trials.", default=10)
    parser.add_argument('--n-jobs', type=int, help="Nb of CPUs.", default=1)
    args = parser.parse_args()

    main_figures_dir = "figures__1_real_data_exp"
    if not os.path.isdir(main_figures_dir):
        os.makedirs(main_figures_dir)

    sub_figure_dir = os.path.join(main_figures_dir, 'figure____' + args_2_str(args))
    if not os.path.isdir(sub_figure_dir):
        os.makedirs(sub_figure_dir)

    ###########################################################################
    # Main loop

    # iterate on lbda
    perf_pd = pd.DataFrame(columns=['dataset', 'trial', 'lbda', 'model', 'error', 'perf'])
    for lbda in args.l_lbda:

        # iterate on dataset
        for k, (dataset_name, (X, y)) in enumerate(datasets.items()):

            clean_dataset_name = dataset_name.replace('\n', '-')

            X -= np.mean(X, axis=0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # iterate on trials
            for i in tqdm.tqdm(range(args.n_trials), desc=f"[Main] Reg. level '{lbda:.3f}' Dataset '{clean_dataset_name:<16}'"):

                clf = AveragingLinearBinaryClassifier(base_model=Ridge(alpha=lbda), random_state=random_state)
                err_baseline = type_1_error(clf.fit(X_train, y_train).predict(X_test), y_test, signed_target=True)
                new_row = {'dataset': dataset_name, 'trial': i, 'lbda': lbda, 'model': 'Opt-m', 'error': err_baseline, 'perf': 0.0}
                perf_pd.loc[len(perf_pd)] = new_row

                # iterate on models
                for marker, clf_name, clf in [set_up_m_1(lbda, m_max, random_state, args.n_jobs),
                                              set_up_m_max(lbda, m_max, random_state, args.n_jobs),
                                              set_up_grid_search(lbda, m_max, random_state, args.n_jobs),
                                              set_up_random_search(lbda, m_max, random_state, args.n_jobs)]:

                    err = type_1_error(clf.fit(X_train, y_train).predict(X_test), y_test, signed_target=True)
                    perf = 100. * (err_baseline - err) / (err_baseline + 1e-6)
                    new_row = {'dataset': dataset_name, 'trial': i, 'lbda': lbda, 'model': clf_name, 'error': err, 'perf': perf}
                    perf_pd.loc[len(perf_pd)] = new_row

    ###########################################################################
    # Plotting

    nrows, ncols = len(datasets.keys()), 1
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             figsize=(ncols*2.0, nrows*1.75))

    for n, dataset_name in enumerate(datasets.keys()):

        added_labels = set()
        for lbda in args.l_lbda:
            for i in range(args.n_trials):
                for model_name in ['m=1', 'm=m_max', 'Grid-search-m', 'Random-search-m']:

                    mask_x = ((perf_pd['lbda'] == lbda) & (perf_pd['dataset'] == dataset_name) &
                              (perf_pd['trial'] == i) & (perf_pd['model'] == 'Opt-m'))

                    mask_y = ((perf_pd['lbda'] == lbda) & (perf_pd['dataset'] == dataset_name) &
                              (perf_pd['trial'] == i) & (perf_pd['model'] == model_name))

                    x = float(perf_pd[mask_x]['error'].iloc[0])
                    y = float(perf_pd[mask_y]['error'].iloc[0])

                    label = model_name if (model_name not in added_labels) and (n == 0) else None
                    axis[n, 0].scatter(x, y, s=20.0, marker=markers[model_name], color='tab:blue', label=label,
                                       alpha=0.75)

                    added_labels.add(model_name)

        axis[n, 0].plot(axis[n, 0].get_xlim(), axis[n, 0].get_xlim(), lw=2.0, linestyle='dashed', color='black',
                        alpha=0.5)

        axis[n, 0].set_xlabel("'Opt-m' error", fontsize=fontsize)
        axis[n, 0].set_ylabel("Error", fontsize=fontsize)
        if n == 0:
            axis[n, 0].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.3, 1.5), fontsize=int(0.65*fontsize))
        axis[n, 0].set_title(f"Dataset '{dataset_name}'", fontsize=fontsize)

    fig.tight_layout()

    figure_filename = os.path.join(sub_figure_dir, 'score_map_comparison')
    fig.savefig(figure_filename + ".png", dpi=300)
    fig.savefig(figure_filename + ".pdf", dpi=300)

    nrows, ncols = 1, len(args.l_lbda)
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             figsize=(ncols*2.0, nrows*2.0))

    for n, lbda in enumerate(args.l_lbda):

        sns.barplot(data=perf_pd[perf_pd['lbda'] == lbda], x='dataset', y='error', hue='model', ax=axis[0, n],
                    width=0.9,  errorbar='sd', errwidth=1, errcolor='tab:gray')
        axis[0, n].axhline(0.0, linestyle='dashed', lw=1.0, color='black', alpha=0.5)

        axis[0, n].set_xticks(range(len(datasets)), list(datasets.keys()), fontsize=int(0.4*fontsize), rotation=60)
        axis[0, n].set_xlabel(r'$\lambda = ' + f"{lbda:.3f}" + '$', fontsize=fontsize)
        axis[0, n].set_ylabel(r'$\mathbf{E}\left[1_{\{f_m(X) \neq Y\}}\right]$', fontsize=fontsize)
        axis[0, n].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.45, 1.1), fontsize=int(0.5*fontsize))

    fig.tight_layout()

    figure_filename = os.path.join(sub_figure_dir, 'score_barplot_comparison')
    fig.savefig(figure_filename + ".png", dpi=300)
    fig.savefig(figure_filename + ".pdf", dpi=300)


###############################################################################
# Timing
print(f"[Main] Experiment runtime: {format_duration(time.perf_counter() - t0_global)}")
