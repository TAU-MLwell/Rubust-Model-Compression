from sklearn.tree import DecisionTreeClassifier
from hypothesis import TreeHypothesis
import sklearn
from hypothesis import SkLearnHypothesis, MLTreeHypothesis
import numpy as np


class Consistency:
    def find(self, x, y):
        raise NotImplementedError

    def get_clone(self, name='median'):
        raise NotImplementedError


class MCSkLearnConsistensy(Consistency):
    def __init__(self):
        self.f = None

    def get_clone(self, name='median'):
        f = sklearn.base.clone(self.f.f)
        f = MLTreeHypothesis(name, f)
        return f

    def find(self, x, y, Y):
        # train
        f = self.get_clone()
        f.train(x, y)

        # check consistency
        pred = f.predict(x)
        for i in range(len(pred)):
            if not np.any(pred[i] in Y[i]):
                return None

        return f


class MCConsistentTree(MCSkLearnConsistensy):
    def __init__(self, depth=None, class_weight=None):
        # create hypothesis of depth depth
        self.max_depth = depth
        self.class_weight = class_weight
        t = DecisionTreeClassifier(max_depth=depth, class_weight=class_weight)
        self.f = MLTreeHypothesis('median_tree', t)

    def get_clone(self, name='median'):
        t = DecisionTreeClassifier(max_depth=self.max_depth, class_weight=self.class_weight)
        f = MLTreeHypothesis('median_tree', t)
        return f