import numpy as np
import pandas as pd

from acc_assessment.utils import build_error_matrix
from acc_assessment.utils import users_accuracy_error, producers_accuracy_error
from acc_assessment.utils import AccuracyAssessment

class Olofsson(AccuracyAssessment):
    """
    Based on:
    Olofsson, P., et al., 2014 "Good practices for estimating area and
    assessing accuracy of land change", Remote Sensing of Environment. Vol
    148 pp. 42-57 10.1016/j.rse.2014/02/015
    """

    def __init__(self, data, mapped_population, map_col=None, ref_col=None):
        """
        data: pd.DataFrame, either the error matrix or the long form table of
            each pixel in the assessment
        mapped_population: dict[str -> float], dictionary mapping each classes
            to its total mapped area
        map_col: str or None, the name of the column containing the map values
            in data (only give this if data is in long form)
        ref_col: str or None, the name of the column containing the reference
            values in data (only give this if the data is in long form)
        """
        if map_col is not None and ref_col is not None:
            # data is in long format i.e. each row is one pixel map and ref values
            mapped_classes = data[map_col].values
            ref_classes = data[ref_col].values

            _all_classes = np.vstack([mapped_classes, ref_classes])
            self.all_classes = np.unique(_all_classes)
            self.all_map_classes = np.unique(mapped_classes)
            self.all_ref_classes = np.unique(ref_classes)

            matrix = build_error_matrix(mapped_classes, ref_classes)
            self._error_matrix_counts = matrix
        else:
            # data is already in error matrix format
            self.all_map_classes = data.index
            self.all_ref_classes = data.columns
            self.all_classes = np.unique(np.vstack([data.index, data.columns]))
            matrix = data
            self._error_matrix_counts = data

        self.N = np.sum(list(mapped_population.values()))
        self.mapped_proportions = {
            k: v / self.N for k, v in iter(mapped_population.items())
            if k in self.all_classes
        }

        # convert error matrix counts to proportions
        props_as_df = pd.DataFrame(
            self.mapped_proportions, index=self.mapped_proportions.keys()
        ).T
        _matrix = matrix.divide(matrix.sum(1), axis="index")
        self._error_matrix = _matrix.multiply(props_as_df).fillna(0.0)

    def overall_accuracy(self):
        """ equation 1 after substituting with equation 4 """
        acc = np.sum(np.diagonal(self._error_matrix))
        vs = {i: self.users_accuracy(i)[1] ** 2
              for i in self.all_map_classes}
        var = np.sum([
            (self.mapped_proportions[i] ** 2) * vs[i]
            for i in self.all_map_classes
        ])
        return acc, np.sqrt(var)

    def users_accuracy(self, i):
        """ equation 2 after substituting with equation 4
        proportion of the area mapped as class i that has reference class i
        """
        correct = self._error_matrix.loc[i, i]
        incorrect = np.sum(self._error_matrix.loc[i, self.all_map_classes])
        if incorrect == 0:
            users_accuracy_error(i)
        acc = correct / incorrect
        n_i = np.sum(self._error_matrix_counts.loc[i, self.all_map_classes])
        if n_i.astype(int) == 1:
            print('divide by zero warning occurs when n_i = 1; see paper')
        var = acc * (1 - acc) / (n_i - 1)
        return acc, np.sqrt(var)

    def commission_error_rate(self, i):
        acc, se = self.users_accuracy(i)
        return 1 - acc, se

    def producers_accuracy(self, j):
        """ equation 3 after substituting with equation 4
        proportion of the area of reference class j that is mapped as class j
        """
        correct = self._error_matrix.loc[j, j]
        incorrect = np.sum(self._error_matrix.loc[self.all_ref_classes, j])
        if incorrect == 0:
            producers_accuracy_error(j)
        acc = correct / incorrect

        N_hat_j = 0
        b = 0
        for i, N_i in iter(self.mapped_proportions.items()):
            if i not in self.all_map_classes:
                continue
            n_i = np.sum(self._error_matrix_counts.loc[i, self.all_map_classes])
            if n_i.astype(int) == 1:
                print('divide by zero warning occurs when n_i = 1; see paper')
            n_ij = self._error_matrix_counts.loc[i, j]
            N_hat_j += (N_i / n_i) * n_ij

            if i != j:
                b += (N_i ** 2) * (n_ij / n_i) * ((1 - (n_ij / n_i)) / (n_i - 1))

        N_j = self.mapped_proportions[j]
        users_acc, _ = self.users_accuracy(j)
        n_j = np.sum(self._error_matrix_counts.loc[j, self.all_map_classes])

        a = ((N_j ** 2) * ((1 - acc) ** 2) * users_acc * (1 - users_acc))
        a /= n_j - 1

        var = (1 / (N_hat_j ** 2)) * (a + ((acc ** 2) * b))
        return acc, np.sqrt(var)

    def omission_error_rate(self, j):
        acc, se = self.producers_accuracy(j)
        return 1 - acc, se

    def proportion_area(self, k):
        """ equation 9 and equation 10 """
        area = 0
        var = 0
        for i in self.all_map_classes:
            W_i  = self.mapped_proportions[i]
            n_i = np.sum(self._error_matrix_counts.loc[i, self.all_map_classes])
            if n_i.astype(int) == 1:
                print('divide by zero warning occurs when n_i = 1; see paper')
            p_ik = W_i * self._error_matrix_counts.loc[i, k] / n_i
            area += p_ik
            var += (W_i * p_ik - (p_ik ** 2)) / (n_i - 1)
        return area, np.sqrt(var)

    def area(self, k):
        p_k, se = self.proportion_area(k)
        return self.N * p_k, self.N * se

    def error_matrix(self, proportions=True):
        if proportions:
            return self._error_matrix
        else:
            return self._error_matrix_counts

