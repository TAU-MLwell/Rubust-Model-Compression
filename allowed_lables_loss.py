import torch.nn as nn
import torch.nn.functional as F
from utils import preds2lables


def allowed_labels_loss(outputs, M_outputs, d, device):
    """
    :param outputs: outputs of small model (m)
    :param M_outputs: outputs of large model (M)
    :param d: depth value
    :param device: device to run on (cpu/cuda)
    :return:
    """
    M_preds = F.softmax(M_outputs, dim=1)
    Y = preds2lables(M_preds, d).to(device)
    loss = nn.BCEWithLogitsLoss()(outputs, Y)
    return loss
