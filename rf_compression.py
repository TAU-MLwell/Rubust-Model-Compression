import numpy as np
import metrics
import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from hypothesis import TreeHypothesis
from consistency import MCConsistentTree
from hypothesis_search import MCHypothesisSearch
from algorithms import crembo


def compress_tree(n_estimators, max_tree_depth, max_forest_depth, X_train, y_train, c, weight=None, X_val=None,
                  y_val=None, score=None, delta=1):
    # All but MED are trained on x_val as well
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    # train benchmark tree
    b_tree = DecisionTreeClassifier(max_depth=max_tree_depth, class_weight=weight)
    b_tree.fit(X_train, y_train)
    f_b = TreeHypothesis('Tree_bench', b_tree)

    # train random forest to create a collection of hypotheses
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_forest_depth, class_weight=weight)
    rf.fit(X_train, y_train)

    # get trees from forest
    hypotheses = []
    for i in range(len(rf.estimators_)):
        name = f'Tree_{i}'
        hypotheses.append(TreeHypothesis(name, f=rf.estimators_[i]))

    # define consistency and hypothesis search algorithms
    consistency = MCConsistentTree(depth=max_tree_depth, class_weight=weight)
    a = MCHypothesisSearch(consistency, X_val=X_val, y_val=y_val, score=score)

    f_med, depth, y1 = crembo(s=X_train, t=hypotheses, c=c, a=a, delta=delta)

    # train a tree with all the data labels from MCMA
    tree = DecisionTreeClassifier(max_depth=max_tree_depth, class_weight=weight)
    tree.fit(X_train, y1)
    f_voting = TreeHypothesis('Tree_voting', tree)

    return rf, hypotheses, f_med, f_voting, f_b


def robustness_agreement_exp(dataset, max_tree_depth, forest_depth, num_trees, num_experiments=1,
                      score='accuracy', weight=None, n_splits=10, delta=1):
    scores = []
    means = []
    for i in range(num_experiments):
        agreement, mean_score = robustness_exp(dataset, max_tree_depth, forest_depth, num_trees, num_experiments=1,
                           score=score, weight=weight, n_splits=n_splits, use_agreement=True, delta=delta)
        means.append(mean_score)
        scores.append(agreement)

    print('\nFinal results:')
    scores = np.asarray(scores)
    scores = scores.mean(axis=0)
    print(f'Average Agreement score: RF {scores[0]}, BM {scores[1]}, VT {scores[2]}, MED {scores[3]}')


def robustness_exp(dataset, max_tree_depth, forest_depth, num_trees, num_experiments=1,
                      score='accuracy', weight=None, n_splits=10, use_agreement=False, delta=1):

    kf_scores = []
    kf = KFold(n_splits=n_splits)
    x, y, X_test, y_test = datasets.prepare_data(dataset, return_test=True)
    c = datasets.get_number_of_classes(dataset)
    score_func, score_metric = metrics.get_socre_foncs(score)

    trees = []
    for train, test in kf.split(x):
        X_train, _, y_train, _ = x[train], x[test], y[train], y[test]
        X_train, X_val, y_train, y_val = datasets.prepare_val(X_train, y_train)
        k_scores = []
        for k in range(num_experiments):
            rf, _, f_med, f_all, f_m = compress_tree(num_trees, max_tree_depth, forest_depth, X_train, y_train,
                                                     c, weight=weight, X_val=X_val, y_val=y_val,
                                                     score=score_metric, delta=delta)

            k_scores.append(score_func(rf, None, f_med, f_all, f_m, X_train, y_train, X_test, y_test))

        trees.append((rf, f_m, f_all, f_med))
        kf_scores.append(metrics.average_scores(k_scores, num_experiments))

    means = metrics.mean_and_std(kf_scores, mean_only=True)
    output = metrics.agreement_score(trees, X_test) if use_agreement else None
    return output, means


def generalization_exp(dataset, max_tree_depth, forest_depth, num_trees, num_experiments=1,
                      score='accuracy', weight=None, n_splits=10, delta=1):

    kf_scores = []
    kf = KFold(n_splits=n_splits)
    x, y, _, _, = datasets.prepare_data(dataset, return_test=False)
    c = datasets.get_number_of_classes(dataset)
    score_func, score_metric = metrics.get_socre_foncs(score)

    for k in range(num_experiments):
        f_scores = []
        for train, test in kf.split(x):
            X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
            X_train, X_val, y_train, y_val = datasets.prepare_val(X_train, y_train)
            rf, _, f_med, f_all, f_m = compress_tree(num_trees, max_tree_depth, forest_depth, X_train, y_train,
                                                     c, weight=weight, X_val=X_val, y_val=y_val,
                                                     score=score_metric, delta=delta)

            f_scores.append(score_func(rf, None, f_med, f_all, f_m, X_train, y_train, X_test, y_test))

        mean_var_win = metrics.mean_and_std(f_scores, mean_only=False)
        kf_scores.append(mean_var_win)

    print('\nFinal results:')
    print(f'Average RF mean {sum([score[0] for score in kf_scores]) / num_experiments}, var {sum([score[1] for score in kf_scores]) / num_experiments}')
    idx = 2
    for t in ('BM', 'VT', 'MED'):
        t_mean = sum([score[idx] for score in kf_scores]) / num_experiments
        t_wins = sum([score[idx + 2] for score in kf_scores]) / num_experiments
        idx += 3
        print(f'Average {t} mean {t_mean}, wins {t_wins}')

    return
