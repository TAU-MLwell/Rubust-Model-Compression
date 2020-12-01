import numpy as np


def de(s, t, f):
    n = len(t)
    est_predictions = np.asarray([est.predict(s) for est in t])
    predictions = f.predict(s)
    predictions = np.tile(predictions, (n, 1))
    d = (predictions == est_predictions).sum(axis=0) / n

    depth = d.min()
    return depth


def memo(s, t, c, a):
    """
    :param s: S = {x1,..,xu} data sample
    :param t: T = {f1,..,fn} hypotheses sample
    :param c: number of classes
    :param a: A(x, y, Y) is a function that given a sample,
    labels and allowed labels returns a consistent hypothesis
    :return: Median hypothesis, belief, thresholds, index of selected depth and predictions for student tree
    """

    # predictions for every estimator
    predictions = np.asarray([est.predict(s) for est in t], dtype=np.int64).T

    # agreement (voting)
    p_bin = np.apply_along_axis(
        lambda k: np.bincount(k, minlength=c),
        axis=1, arr=predictions)

    # calculate probabilities (p_ic)
    p_ic = np.divide(p_bin, len(t))

    # training labels for voting tree
    y = np.argmax(p_ic, axis=1)

    # remove validation samples
    val_size = a.X_val.shape[0]
    p_ic = p_ic[:-val_size, :]
    s = s[:-val_size, :]

    thresholds = p_ic.flatten()
    thresholds = np.unique(thresholds)
    thresholds.sort()

    f, idx = a.search(s, p_ic, thresholds)
    depth = thresholds[idx]
    print(f'MEMO: index {idx}, depth {depth}')

    return f, p_ic, thresholds, idx, y


def crembo(s, t, c, a, delta=1):
    f, p_ic, thresholds, idx, y = memo(s, t, c, a)

    # remove validation samples
    val_size = a.X_val.shape[0]
    s = s[:-val_size, :]

    f, idx = a.depth_opt(s, p_ic, thresholds, f, idx, delta=delta)
    depth = thresholds[idx]
    print(f'CREMBO: index {idx}, depth {depth}')
    return f, depth, y


def memo_oracle(s, p_ic, a):
    """
    :param s: S = {x1,..,xu} data sample
    :param p_ic: oracle agreement probabilities
    :param a: A(x, y, Y) is a function that given a sample,
    labels and allowed labels returns a consistent hypothesis
    :return: Median hypothesis, thresholds, index of selected depth and predictions for student tree
    """
    # get training labels
    y = np.argmax(p_ic, axis=1)

    # remove validation samples
    val_size = a.X_val.shape[0]
    p_ic = p_ic[:-val_size, :]
    s = s[:-val_size, :]

    thresholds = p_ic.flatten()
    thresholds = np.unique(thresholds)
    thresholds.sort()

    f, idx = a.search(s, p_ic, thresholds)
    depth = thresholds[idx]
    print(f'MEMO: index {idx}, depth {depth}')

    return f, thresholds, idx, y


def crembo_oracle(s, p_ic, a, delta=1):
    f, thresholds, idx, y = memo_oracle(s, p_ic, a)

    # remove validation samples
    val_size = a.X_val.shape[0]
    s = s[:-val_size, :]
    p_ic = p_ic[:-val_size, :]

    f, idx = a.depth_opt(s, p_ic, thresholds, f, idx, delta=delta)
    depth = thresholds[idx]
    print(f'CREMBO: index {idx}, depth {depth}')

    return f, depth, y
