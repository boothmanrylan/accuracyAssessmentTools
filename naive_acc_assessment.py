import numpy as np
import pandas as pd

from utils import build_error_matrix, _expand_error_matrix

class NaiveAccAssessment():
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
        return correct / incorrect, None

    def commission_error_rate(self, k):
        return 1 - self.users_accuracy(k), None

    def producers_accuracy(self, k):
        correct = self._error_matrix.loc[k, k]
        incorrect = self._error_matrix.loc[:, k].sum()
        return correct / incorrect, None

    def omission_error_rate(self, k):
        return 1 - self.producers_accuracy(k), None

    def error_matrix(self):
        return self._error_matrix

    def area(self, k):
        msg = "can't estimate area of a class from the error matrix alone"
        raise NotImplementedError(msg)

if __name__ == "__main__":
    data = [[65, 4, 22, 24], [6, 81, 5, 8], [0, 11, 85, 19], [4, 7, 3, 90]]
    classes = ["D", "C", "BA", "SB"]
    df = pd.DataFrame(data, index=classes, columns=classes)
    input_df = _expand_error_matrix(df, "map", "ref")

    assessment = NaiveAccAssessment(input_df, "map", "ref")

    # ========================================================
    # THESE ARE THE VALUES FROM THE PAPER
    # ========================================================
    expected_overall = 0.74
    expected_prods = {"D": 0.87, "C": 0.79, "BA": 0.74, "SB": 0.64}
    expected_users = {"D": 0.57, "C": 0.81, "BA": 0.74, "SB": 0.87}

    overall = assessment.overall_accuracy()[0]
    print(f"Overall Acc: {overall:.2f} | EXPECTED: {expected_overall}")

    for k in classes:
        users = assessment.users_accuracy(k)[0]
        print(f"Users Acc {k}: {users:.2f} | EXPECTED: {expected_users[k]}")

    for k in classes:
        prods = assessment.producers_accuracy(k)[0]
        print(f"Prods Acc {k}: {prods:.2f} | EXPECTED: {expected_prods[k]}")
