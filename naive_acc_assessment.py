import numpy as np
import pandas as pd

class NaiveAccAssessment():
    def __init__(self, data, map_class, ref_class):
        self.all_classes = np.unique(data[map_class])
        self.num_classes = len(self.all_classes)
        matrix = np.zeros((self.num_classes, self.num_classes))
        for i in self.all_classes:
            for j in self.all_classes:
                mapped = data[map_class] == i
                ref = data[ref_class] == j
                matrix[i, j] = np.sum(mapped * ref)
        self.error_matrix = pd.DataFrame(
            matrix, index=self.all_classes, columns=self.all_classes)
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
