import numpy as np
import metrics
import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from hypothesis import TreeHypothesis
from consistency import MCConsistentTree
from hypothesis_search import MCHypothesisSearch
from algorithms import crembo_oracle
from nn_train import train_oracle_and_predict


def compress_tree(max_tree_depth, X_train, y_train, p_ic, weight=None, X_val=None, y_val=None, score=None, delta=1):
    X_train = np.concatenate([X_train, X_val], axis=0) if X_val is not None else X_train
    y_train = np.concatenate([y_train, y_val], axis=0) if X_val is not None else y_train

    # train benchmark tree
    b_tree = DecisionTreeClassifier(max_depth=max_tree_depth, class_weight=weight)
    b_tree.fit(X_train, y_train)
    f_b = TreeHypothesis('Tree_bench', b_tree)

    consistency = MCConsistentTree(depth=max_tree_depth, class_weight=weight)
    a = MCHypothesisSearch(consistency, X_val=X_val, y_val=y_val, score=score)

    f_med, depth, y = crembo_oracle(s=X_train, p_ic=p_ic, a=a, delta=delta)

    # train a tree with all the data labels from M
    tree = DecisionTreeClassifier(max_depth=max_tree_depth, class_weight=weight)
    tree.fit(X_train, y)
    f_voting = TreeHypothesis('Tree_voting', tree)

    return f_med, f_voting, f_b


def robustness_exp(max_tree_depth, num_experiments, dataset, score='accuracy',
                   weight=None, n_splits=10, device='cpu', use_agreement=False, delta=1):

    kf_scores = []
    kf = KFold(n_splits=n_splits)
    x, y, X_test, y_test = datasets.prepare_data(dataset, return_test=True)

    score_func, score_metric = metrics.get_socre_foncs(score)
    c = datasets.get_number_of_classes(dataset)

    trees = []
    for train, test in kf.split(x):
        X_train, _, y_train, _ = x[train], x[test], y[train], y[test]
        X_train, X_val, y_train, y_val = datasets.prepare_val(X_train, y_train)
        X_nn = np.concatenate([X_train, X_val], axis=0)
        y_nn = np.concatenate([y_train, y_val], axis=0)
        k_scores = []
        nn_score = []
        for k in range(num_experiments):
            p_ic, nn_test_score = train_oracle_and_predict(dataset, X_nn, y_nn, X_test, y_test, c, device)
            f_med, f_voting, f_m = compress_tree(max_tree_depth, X_train, y_train, p_ic, weight=weight, X_val=X_val,
                                                 y_val=y_val, score=score_metric, delta=delta)
            k_scores.append(score_func(None, None, f_med, f_voting, f_m, X_train, y_train, X_test, y_test))
            nn_score.append(nn_test_score)

        trees.append((f_m, f_voting, f_med))
        kf_scores.append(metrics.average_scores(k_scores, num_experiments))

    means = metrics.mean_and_std(kf_scores, mean_only=True, show_rf=False)
    output = metrics.agreement_score(trees, X_test) if use_agreement else None

    nn_av = np.mean(nn_score)
    return output, means, nn_av


def robustness_agreement_exp(max_tree_depth, num_experiments, dataset, score='accuracy',
                   weight=None, n_splits=10, device='cpu', delta=1):
    scores = []
    means = []
    nn_av = []
    for i in range(num_experiments):
        print(f'Experiment number {i+1}')
        agreement, pred_scores, nn_score = robustness_exp(max_tree_depth, num_experiments=1, dataset=dataset, score=score,
                   weight=weight, n_splits=n_splits, device=device, use_agreement=True, delta=delta)
        means.append(pred_scores)
        scores.append(agreement)
        nn_av.append(nn_score)

    print('\nFinal results:')
    scores = np.asarray(scores)
    scores = scores.mean(axis=0)
    print(f'Average Agreement score: BM {scores[0]}, VT {scores[1]}, MED {scores[2]}')


def generalization_exp(max_tree_depth, num_experiments=1, dataset='mnist',
              score='accuracy', weight=None, device='cpu', delta=1, n_splits=10):

    kf_scores = []
    kf = KFold(n_splits=n_splits)
    x, y, _, _,  = datasets.prepare_data(dataset, return_test=False)

    c = datasets.get_number_of_classes(dataset)
    score_func, score_metric = metrics.get_socre_foncs(score)

    nn_score = []
    for k in range(num_experiments):
        print(f'Experiment number {k+1}')
        f_scores = []
        for train, test in kf.split(x):
            X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
            X_train, X_val, y_train, y_val = datasets.prepare_val(X_train, y_train)
            X_nn = np.concatenate([X_train, X_val], axis=0)
            y_nn = np.concatenate([y_train, y_val], axis=0)

            p_ic, nn_test_score = train_oracle_and_predict(dataset, X_nn, y_nn, X_test, y_test, c, device)

            f_med, f_all, f_m = compress_tree(max_tree_depth, X_train, y_train, p_ic, weight=weight, X_val=X_val,
                                                 y_val=y_val, score=score_metric, delta=delta)
            f_scores.append(score_func(None, None, f_med, f_all, f_m, X_train, y_train, X_test, y_test))
            nn_score.append(nn_test_score)

        mean_var_win = metrics.mean_and_std(f_scores, mean_only=False, show_rf=False, nn_score=nn_score)
        kf_scores.append(mean_var_win)

    print('\nFinal results:')
    print(f'Average NN mean {sum([score[0] for score in kf_scores]) / num_experiments}, std {sum([score[1] for score in kf_scores]) / num_experiments}')
    idx = 2
    for t in ('BM', 'VT', 'MED'):
        t_mean = sum([score[idx] for score in kf_scores]) / num_experiments
        t_wins = sum([score[idx + 2] for score in kf_scores]) / num_experiments
        idx += 3
        print(f'Average {t} mean {t_mean}, wins {t_wins}')

    return
