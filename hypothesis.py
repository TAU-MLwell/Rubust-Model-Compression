import numpy as np


class Hypothesis:
    def __init__(self, name, f=None):
        self.name = name
        self.f = f

    def get_name(self):
        return self.name

    def train(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class SkLearnHypothesis(Hypothesis):
    def __index__(self, name, f):
        super.__init__(name, f)

    def train(self, x, y):
        self.f.fit(x, y)
        return

    def predict(self, x):
        return self.f.predict(x)

    def predict_proba(self, x):
        return self.f.predict_proba(x)

    def score(self, x, y):
        return self.f.score(x, y)


class MLTreeHypothesis(Hypothesis):
    def __index__(self, name, f):
        super.__init__(name, f)

    def train(self, x, y):
        self.f.fit(x, y)
        return

    def predict(self, x):
        predicted_prob = self.f.predict_proba(x)
        for i in range(len(predicted_prob)):
            if predicted_prob[i].shape[1] == 1:
                prob_i = np.zeros((predicted_prob[i].shape[0], 2))
                prob_i[:, 1] = predicted_prob[i][:, 0]
                predicted_prob[i] = prob_i

        prob = np.asarray(predicted_prob)
        pred = np.argmax(prob[:, :, prob.shape[2]-1], axis=0)
        return pred

    def predict_proba(self, x):
        return self.f.predict_proba(x, n_outputs=6)

    def score(self, x, y, device=None):
        pred = self.predict(x)
        score = sum(pred == y) / len(y)
        return score


class TreeHypothesis(SkLearnHypothesis):
    def __index__(self, name, f):
        super.__init__(name, f)
