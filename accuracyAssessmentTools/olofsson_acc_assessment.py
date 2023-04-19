import numpy as np
import pandas as pd

from accuracyAssessmentTools.utils import build_error_matrix
from accuracyAssessmentTools.utils import _expand_error_matrix
from accuracyAssessmentTools.utils import users_accuracy_error
from accuracyAssessmentTools.utils import producers_accuracy_error

class Olofsson2014AccAssessment():
    """
    Based on:
    Olofsson, P., et al., 2014 "Good practices for estimating area and   
    assessing accuracy of land change", Remote Sensing of Environment. Vol
    148 pp. 42-57 10.1016/j.rse.2014/02/015
    """

    def __init__(self, data, map_col, ref_col,  mapped_population):
        mapped_classes = data[map_col].values
        ref_classes = data[ref_col].values

        _all_classes = np.vstack([mapped_classes, ref_classes])
        self.all_classes = np.unique(_all_classes)
        self.all_map_classes = np.unique(mapped_classes)
        self.all_ref_classes = np.unique(ref_classes)

        self.N = np.sum(list(mapped_population.values()))
        self.mapped_proportions = {
            k: v / self.N for k, v in iter(mapped_population.items())
            if k in self.all_classes
        }

        matrix = build_error_matrix(mapped_classes, ref_classes)
        self._error_matrix_counts = matrix

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
                b += (N_i ** 2) * n_ij / n_i * (1 - (n_ij / n_i)) / (n_i - 1)

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

    def error_matrix(self):
        return self._error_matrix

def test():
    data = {"Deforestation": [66, 0, 1, 2],
            "Forest gain": [0, 55, 0, 1],
            "Stable forest": [5, 8, 153, 9],
            "Stable non-forest": [4, 12, 11, 313]}
    mapped_area = {"Deforestation": 200000,
                   "Forest gain": 150000,
                   "Stable forest": 3200000,
                   "Stable non-forest": 6450000}

    df = pd.DataFrame(data)
    df.index = df.columns
    input_df = _expand_error_matrix(df, "map", "ref")
    assessment = Olofsson2014AccAssessment(
        input_df, "map", "ref", mapped_area)

    # ===================================================
    # THESE ARE THE VALUES GIVEN IN THE PAPER
    # ===================================================
    expected_users_accuracies = {
        "Deforestation": "0.88 +/- 0.07",
        "Forest gain": "0.73 +/- 0.10",
        "Stable forest": "0.93 +/- 0.04",
        "Stable non-forest": "0.96 +/- 0.02"
    }

    expected_producers_accuracies = {
        "Deforestation": "0.75 +/- 0.21",
        "Forest gain": "0.85 +/- 0.23",
        "Stable forest": "0.93 +/- 0.03",
        "Stable non-forest": "0.96 +/- 0.01"
    }

    expected_class_pixel_counts = {
        "Deforestation": "235086 +/- 68418 pixels",
        "Forest gain": "not given",
        "Stable forest": "not given",
        "Stable non-forest": "not given"
    }

    expected_class_areas = {
        "Deforestation": "21158 +/- 6158 ha",
        "Forest gain": "11686 +/- 3756 ha",
        "Stable forest": "285770 +/- 15510 ha",
        "Stable non-forest": "581386 +/- 16282"
    }

    expected_error_matrix = pd.DataFrame({
        "Deforestation": [0.0176, 0, 0.0019, 0.0040],
        "Forest gain": [0, 0.0110, 0, 0.0020],
        "Stable forest": [0.0013, 0.0016, 0.2967, 0.0179],
        "Stable non-forest": [0.0011, 0.0024, 0.0213, 0.6212],
    });
    expected_error_matrix.index = expected_error_matrix.columns

    print("--------------USERS ACC----------------")
    for k in df.columns:
        users_acc, se = assessment.users_accuracy(k)
        expected = expected_users_accuracies[k]
        print(f"{k}:\t{users_acc:.2g} +/- {1.96 * se:.2g}", end="\t| ")
        print(f"EXPECTED: {expected}")

    print("\n-------------PRODUCERS ACC-------------")
    for k in df.columns:
        prods_acc, se = assessment.producers_accuracy(k)
        expected = expected_producers_accuracies[k]
        print(f"{k}:\t{prods_acc:.2g} +/- {1.96 * se:.2g}", end="\t| ")
        print(f"EXPECTED: {expected}")

    print("\n--------ESTIMATED AREA (pixels)--------")
    for k in df.columns:
        area, se = assessment.area(k)
        expected = expected_class_pixel_counts[k]
        print(f"{k}:\t{area:.2f} +/- {1.96 * se:.2f}", end="\t| ")
        print(f"EXPECTED: {expected}")

    print("\n--------ESTIMATED AREA (ha)--------")
    for k in df.columns:
        area, se = assessment.area(k)
        area /= 11.11
        se /= 11.11
        expected = expected_class_areas[k]
        print(f"{k}:\t{area:.2f} +/- {1.96 * se:.2f}", end="\t| ")
        print(f"EXPECTED: {expected}")

    overall_acc, se = assessment.overall_accuracy()
    print(f"\noverall accuracy: {overall_acc:.2g} +/- {1.96 * se:.2g}",
        end=" | ")
    print("EXPECTED: 0.95 +/- 0.02")

    print("--------------ERROR MATRIX----------------")
    print(assessment.error_matrix())
    print("\nEXPECTED ERROR MATRIX:")
    print(expected_error_matrix)

if __name__ == '__main__':
    test()
