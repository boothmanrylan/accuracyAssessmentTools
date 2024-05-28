import numpy as np
import pandas as pd

from acc_assessment.utils import build_error_matrix, _expand_error_matrix
from acc_assessment.utils import users_accuracy_error, producers_accuracy_error
from acc_assessment.utils import AccuracyAssessment

class Congalton(AccuracyAssessment):
    """
    Calculates users/producers/overall accuracy from an error matrix without
    considering the proportion of mapped area of each class.

    Based on:
    "A Review of Assessing the Accuracy of Classifications of Remotely
    Sensed Data", Congalton, R. G., 1991. Remote Sensing of Environment,
    Vol. 37. pp 35-46 https://doi.org/10.106/0034-4257(91)90048-B
    """
    def __init__(self, data, map_class, ref_class):
        self.all_classes = np.unique(data[map_class])
        self.num_classes = len(self.all_classes)
        self._error_matrix = build_error_matrix(
            data[map_class], data[ref_class])
        self.N = data.shape[0]

    def overall_accuracy(self):
        return np.sum(np.diagonal(self._error_matrix)) / self.N, None

    def users_accuracy(self, k):
        correct = self._error_matrix.loc[k, k]
        incorrect = self._error_matrix.loc[k, :].sum()
        if incorrect == 0:
            users_accuracy_error(k)
        return correct / incorrect, None

    def commission_error_rate(self, k):
        return 1 - self.users_accuracy(k), None

    def producers_accuracy(self, k):
        correct = self._error_matrix.loc[k, k]
        incorrect = self._error_matrix.loc[:, k].sum()
        if incorrect == 0:
            producers_accuracy_error(k)
        return correct / incorrect, None

    def omission_error_rate(self, k):
        return 1 - self.producers_accuracy(k), None

    def error_matrix(self):
        return self._error_matrix

    def area(self, k):
        msg = "can't estimate area of a class from the error matrix alone"
        raise NotImplementedError(msg)

