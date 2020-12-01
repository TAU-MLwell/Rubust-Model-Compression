import torch
import torch.nn.functional as F
import numpy as np

def preds2lables(preds, d, soft=False, normlize=False):
    """
    :param preds: M predictions
    :param d: depth
    :param soft: soft predictions
    :param normlize: normalize predictions
    :return: Y, allowed labels that corresponds with the predictions
    """
    preds_shape = preds.shape
    ones = preds if soft else torch.ones(preds_shape).to(preds.device)
    zeroes = torch.zeros(preds_shape).to(preds.device)
    Y = torch.where(preds >= d, ones, zeroes)

    if normlize:
        Y = F.normalize(Y, dim=1, p=1)

    return Y


def calc_allowed_labels(preds, d):
    """
    Calculates allowed labels for SKLEARN models
    :param preds: M predictions
    :param d: depths
    :return: Y, allowed labels that corresponds with the predictions
    """
    Y = []
    y = np.zeros(preds.shape)
    for i in range(preds.shape[0]):
        allowed_targets = np.where(preds[i, :] >= d)[0]
        y[i, allowed_targets] = 1
        Y.append(allowed_targets)

    return Y, y