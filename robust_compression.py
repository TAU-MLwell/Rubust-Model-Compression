import torch
import torch.nn.functional as F
from utils import preds2lables, calc_allowed_labels
import numpy as np

class CREMBO:
    def __init__(self, create_model, train_hypothesis, eval_model, args, delta=50):
        """
        :param create_model: Function that returns the small model. signature: create_model(args) -> torch.nn.Module
        :param train_hypothesis: Training function for training small model. Must use allowed_labels_loss.
        Signature: train_hypothesis(args, model, M, train_dataloader, test_dataloader, device) -> torch.nn.Module
        :param eval_model: Function for evaluating model on a validation set.
        Signature: eval_model(model, val_dataloader, device)
        :param args: Optional arguments for create_model and train_hypothesis
        :param delta: step size in loop
        """
        self.create_model = create_model
        self.train_hypothesis = train_hypothesis
        self.eval_model = eval_model
        self.args = args
        self.delta = delta
        self.memo = MEMO(create_model, train_hypothesis, args)

    def __call__(self, M, train_dataloader, test_dataloader, val_dataloader, device='cpu'):
        """
        :param M: Large model (must return probabilities)
        :param train_dataloader: Train set dataloader or (x,y) tuple in case of sklearn models
        :param test_dataloader:  Test set dataLoader or (x,y) tuple in case of sklearn models
        :param val_dataloader: Validation set dataLoader or (x,y) tuple in case of sklearn models
        :param device: device to run on (cpu/cuda)
        :return: f which is the selected hypothesis
        """

        f, thresholds, idx = self.memo(M, train_dataloader, test_dataloader, device)

        start = idx
        end = len(thresholds) - 1
        best = self.eval_model(f, val_dataloader, device)

        for m in range(start, end, self.delta):
            model = self.create_model(self.args)
            d = thresholds[m]
            h = self.train_hypothesis(self.args, model, M, d, train_dataloader, test_dataloader, device)

            # evaluate score on validation set
            score = self.eval_model(h, val_dataloader, device)

            if score > best:
                best = score
                f = h

        return f


class MEMO:
    def __init__(self, create_model, train_hypothesis, args):
        """
        :param create_model: Function that returns the small model. signature: create_model(args) -> torch.nn.Module
        :param train_hypothesis: Training function for training small model. Must use allowed_labels_loss.
        Signature: train_hypothesis(args, model, M, train_dataloader, test_dataloader, device) -> torch.nn.Module
        :param args: Optional arguments for create_model and train_hypothesis
        """
        self.create_model = create_model
        self.train_hypothesis = train_hypothesis
        self.args = args

        if 'sklearn' in args and args['sklearn']:
            self.get_depths = sklearn_get_depths
            self.is_consistent = sklearn_is_consistent
        else:
            self.get_depths = get_depths
            self.is_consistent = is_consistent

    def __call__(self, M, train_dataloader, test_dataloader, device='cpu'):
        """
        :param M: Large model (must return probabilities)
        :param train_dataloader: Train set dataloader or (x,y) tuple in case of sklearn models
        :param test_dataloader:  Test set dataLoader or (x,y) tuple in case of sklearn models
        :param device: device to run on (cpu/cuda)
        :return: f_out, thresholds, idx which are the median hypothesis, sorted thresholds (depths) list and index of
        the median hypothesis depth respectively.
        """
        thresholds = self.get_depths(M, train_dataloader, device=device)

        start = 0
        end = len(thresholds) - 1
        f_out = None
        idx = 0
        while start <= end:
            m = (end + start) // 2
            d = thresholds[m]
            model = self.create_model(self.args)
            f = self.train_hypothesis(self.args, model, M, d, train_dataloader, test_dataloader, device)

            # check for consistency
            consistent = self.is_consistent(f, M, train_dataloader, d, device)

            if consistent:
                f_out = f
                idx = m
                start = m + 1
            else:
                end = m - 1

        if f_out is None:
            f_out = model

        return f_out, thresholds, idx


def get_depths(M, train_dataloader, device='cpu'):
    """
    :param M: Large model (oracel)
    :param train_dataloader: training set dataloader
    :param device: device to run on (cpu/cuda)
    :return: depths, a list of sorted unique depths
    """
    M.eval()
    depths = None

    for i, (samples, labels) in enumerate(train_dataloader):
        samples = samples.to(device, dtype=torch.float)
        with torch.no_grad():
            M_outputs = M(samples)
            M_preds = F.softmax(M_outputs, dim=1)

        if depths is None:
            depths = M_preds
        else:
            depths = torch.cat([depths, M_preds], dim=0)


    depths[depths < 0.01] = 0
    depths[depths > 0.99] = 1
    depths = depths.flatten().sort()[0].unique()
    return depths


def is_consistent(model, M, train_dataloader, d, device, soft=False):
    """
    :param model: model to check for consistency
    :param M: Large model (oracel)
    :param train_dataloader: training set dataloader
    :param d: depth
    :param device: device to run on (cpu/cuda)
    :param soft: soft predictions
    :return: boolean, True if consistent else False
    """
    model.eval()
    M.eval()

    for i, (samples, labels) in enumerate(train_dataloader):
        samples = samples.to(device, dtype=torch.float)
        with torch.no_grad():
            M_outputs = M(samples)
            M_preds = F.softmax(M_outputs, dim=1)
            Y = preds2lables(M_preds, d, soft=soft).to(device)

            predictions = model(samples)
            _, predicted = torch.max(predictions.data, 1)
            for j in range(Y.shape[0]):
                if Y[j][predicted[j]] == 0:
                    return False

    return True


def sklearn_get_depths(M, train_dataloader, device='cpu'):
    X_train, y_train = train_dataloader

    # get M predictions
    preds = M.predict_proba(X_train)

    depths = preds.flatten()
    depths = np.unique(depths)
    depths.sort()

    return depths


def sklearn_is_consistent(model, M, train_dataloader, d, device, soft=False):
    X_train, y_train = train_dataloader

    # get M predictions
    preds = M.predict_proba(X_train)

    # calculate allowed labels
    Y, y = calc_allowed_labels(preds, d)

    # get model predictions
    pred = model.predict(X_train)
    for i in range(len(pred)):
        if not np.any(pred[i] in Y[i]):
            return False

    return True
