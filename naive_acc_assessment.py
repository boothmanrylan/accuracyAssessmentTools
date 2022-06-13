import numpy as np
import pandas as pd

from utils import build_error_matrix

class NaiveAccAssessment():
    """
    Calculates users/producers/overall accuracy from an error matrix without
    considering the proportion of mapped area of each class.
    """
    def __init__(self, data, map_class, ref_class):
        self.all_classes = np.unique(data[map_class])
        self.num_classes = len(self.all_classes)
        self.error_matrix = build_error_matrix(
            data[map_class], data[ref_class])
        self.N = data.shape[0]

    def overall_accuracy(self):
        return np.sum(np.diagonal(self.error_matrix)) / self.N

    def users_accuracy(self, k):
        return self.error_matrix.loc[k, k] / self.error_matrix.loc[k, :].sum()

    def commission_error_rate(self, k):
        return 1 - self.users_accuracy(k)

    def producers_accuracy(self, k):
        return self.error_matrix.loc[k, k] / self.error_matrix.loc[:, k].sum()

    def omission_error_rate(self, k):
        return 1 - self.producers_accuracy(k)

    def error_matrix(self):
        return self.error_matrix

    def area(self, k):
        msg = "can't estimate area of a class from the error matrix alone"
        raise NotImplementedError(msg)
