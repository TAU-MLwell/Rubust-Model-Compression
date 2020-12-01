import numpy as np


def average_scores(k_scores, k):
    num_trees = len(k_scores[0][0])
    hypotheses_train_av_scores = np.zeros((1, num_trees))
    hypotheses_test_av_scores = np.zeros((1, num_trees))
    rf_train_av = 0
    rf_test_av = 0
    med_score_train_av = 0
    med_score_test_av = 0
    score_av_train_all = 0
    score_av_test_all = 0
    f_b_train_score_av = 0
    f_b_test_score_av = 0
    for i in range(k):
        hypotheses_train_scores, hypotheses_test_scores, rf_train_score, rf_test_score, med_score_train, med_score_test, score_train_all, score_test_all, f_b_train_score, f_b_test_score = k_scores[i]
        if rf_train_score is not None:
            rf_train_av += (rf_train_score / k)
            rf_test_av += (rf_test_score / k)
        med_score_train_av += (med_score_train / k)
        med_score_test_av += (med_score_test / k)
        score_av_train_all += (score_train_all / k)
        score_av_test_all += (score_test_all / k)
        f_b_train_score_av += (f_b_train_score / k)
        f_b_test_score_av += (f_b_test_score / k)

    hypotheses_train_av_scores = np.divide(hypotheses_train_av_scores, k)
    hypotheses_test_av_scores = np.divide(hypotheses_test_av_scores, k)

    return hypotheses_train_av_scores, hypotheses_test_av_scores, rf_train_av, rf_test_av, med_score_train_av, med_score_test_av, score_av_train_all, score_av_test_all, f_b_train_score_av, f_b_test_score_av


def mean_and_std(score_list, mean_only=True, show_rf=True, nn_score=None):
    # train is -1
    med_test_scores = [score[5] for score in score_list]
    voting_test_scores = [score[7] for score in score_list]
    bm_test_scores = [score[9] for score in score_list]

    bm_mean, voting_mean, med_mean = np.mean(bm_test_scores), np.mean(voting_test_scores), np.mean(med_test_scores)
    bm_std, voting_std, med_std = np.std(bm_test_scores), np.std(voting_test_scores), np.std(med_test_scores)
    mean_print = 'Mean Test Score: '
    std_print = 'std Test Score: '
    model_mean = 0
    if show_rf:
        rf_test_scores = [score[3] for score in score_list]
        model_mean = np.mean(rf_test_scores)
        model_std = np.std(rf_test_scores)
        mean_print += f'RF {model_mean}, '
        std_print += f'RF {model_std}, '
    if nn_score:
        model_mean = np.mean(nn_score)
        model_std =np.std(nn_score)
        mean_print += f'NN {model_mean}, '
        std_print += f'NN {model_std}, '

    mean_print += f'BM {bm_mean}, VT {voting_mean}, MED {med_mean}'
    std_print += f'BM {bm_std}, VT {voting_std}, MED {med_std}'



    print(mean_print)
    if mean_only:
        return model_mean, bm_mean, voting_mean, med_mean
    print(std_print)


    wins = np.argmax(np.vstack([med_test_scores, voting_test_scores, bm_test_scores]), axis=0)
    med_wins = sum(wins == 0)
    voting_wins = sum(wins == 1)
    bm_wins = sum(wins == 2)
    print(f'Wins: BM {bm_wins}, VT {voting_wins}, MED {med_mean}')
    return model_mean, model_std, bm_mean, bm_std, bm_wins, voting_mean, voting_std, voting_wins, med_mean, med_std, med_wins


def acc_score(rf, hypotheses, f_med, f_all, f_b, X_train, y_train, X_test, y_test):
    # individual tree score
    hypotheses_train_scores = []
    hypotheses_test_scores = []
    if hypotheses is not None:
        for f in hypotheses:
            hypotheses_train_scores.append(f.score(X_train, y_train))
            hypotheses_test_scores.append(f.score(X_test, y_test))

    # benchmark score
    f_b_train_score = f_b.score(X_train, y_train)
    f_b_test_score = f_b.score(X_test, y_test)

    # random forest score
    rf_train_score = None
    rf_test_score = None
    if rf is not None:
        rf_train_score = rf.score(X_train, y_train)
        rf_test_score = rf.score(X_test, y_test)

    # median score
    med_score_train = f_med.score(X_train, y_train)
    med_score_test = f_med.score(X_test, y_test)

    # all points score
    score_train_all = f_all.score(X_train, y_train)
    score_test_all = f_all.score(X_test, y_test)

    return hypotheses_train_scores, hypotheses_test_scores, rf_train_score, rf_test_score, med_score_train, med_score_test, score_train_all, score_test_all, f_b_train_score, f_b_test_score


def score(score_func, rf, hypotheses, f_med, f_all, f_b, X_train, y_train, X_test, y_test):
    # individual tree score
    hypotheses_train_scores = []
    hypotheses_test_scores = []

    # benchmark score
    f_b_train_score = score_func(f_b, X_train, y_train)
    f_b_test_score = score_func(f_b, X_test, y_test)

    # random forest score
    rf_train_score = score_func(rf, X_train, y_train)
    rf_test_score = score_func(rf, X_test, y_test)

    # median score
    med_score_train = score_func(f_med, X_train, y_train)
    med_score_test = score_func(f_med, X_test, y_test)

    # all points score
    score_train_all = score_func(f_all, X_train, y_train)
    score_test_all = score_func(f_all, X_test, y_test)

    return hypotheses_train_scores, hypotheses_test_scores, rf_train_score, rf_test_score, med_score_train, med_score_test, score_train_all, score_test_all, f_b_train_score, f_b_test_score


def get_socre_foncs(score_name):
    if score_name == 'accuracy':
        return acc_score, None
    else:
        raise NotImplementedError(f'Not implemented for {score_name} score')


def agreement_score(hypotheses, X):
    final_score = []
    for i in range(len(hypotheses)):
        agreement = []
        for j in range(len(hypotheses)):
            if i == j:
                continue

            score = []
            for k in range(len(hypotheses[i])):
                pred_i = hypotheses[i][k].predict(X)
                pred_j = hypotheses[j][k].predict(X)
                score.append(sum(pred_i == pred_j) / len(X))

            agreement.append(score)
        agreement = np.asarray(agreement)
        agreement = agreement.mean(axis=0)
        final_score.append(agreement)

    final_score = np.asarray(final_score)
    final_score = final_score.mean(axis=0)
    return final_score
