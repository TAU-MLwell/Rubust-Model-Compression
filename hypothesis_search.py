import numpy as np


class MCHypothesisSearch:
    def __init__(self, consistency, X_val=None, y_val=None, score=None):
        self.consistency = consistency
        self.X_val = X_val
        self.y_val = y_val
        self.score = score

    def calc_allowed_labels(self, p_ic, threshold):
        Y_targets = []
        y = np.zeros(p_ic.shape)
        for i in range(p_ic.shape[0]):
            allowed_targets = np.where(p_ic[i, :] >= threshold)[0]
            y[i, allowed_targets] = 1
            Y_targets.append(allowed_targets)

        return Y_targets, y

    def search(self, x, p_ic, thresholds):
        # get sorted thresholds
        start = 0
        end = len(thresholds) - 1
        f_out = None
        idx = 0
        while start <= end:
            m = (end + start) // 2
            # calculate allowed labels
            Y, y = self.calc_allowed_labels(p_ic, thresholds[m])
            f = self.consistency.find(x, y, Y)

            # not consistent
            if f is None:
                end = m - 1

            # consistent
            else:
                f_out = f
                idx = m
                start = m + 1

        return f_out, idx

    def depth_opt(self, x, p_ic, thresholds, f, idx, delta=1):
        if self.X_val is None:
            raise AssertionError('Validation set must be supplied')

        start = idx
        end = len(thresholds) - 1
        best = 0
        thresh = []
        scores = []
        for m in range(start, end, delta):
            h = self.consistency.get_clone()
            _, y = self.calc_allowed_labels(p_ic, thresholds[m])
            h.train(x, y)
            if self.score is None:
                score = h.score(self.X_val, self.y_val)
            else:
                score = self.score(h, self.X_val, self.y_val)

            if score > best:
                best = score
                idx = m
                f = h

        return f, idx
