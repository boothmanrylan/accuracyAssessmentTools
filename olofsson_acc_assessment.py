import numpy as np
import pandas as pd

def build_error_matrix(mapped_classes, ref_classes):
    """ build the error matrix of sample counts """
    all_classes = np.unique(ref_classes)
    n_classes = all_classes.shape[0]
    counts = np.zeros((n_classes, n_classes))
    for i, map_class in enumerate(all_classes):
        mapped_as_i = mapped_classes == map_class
        for j, ref_class in enumerate(all_classes):
            ref_is_j = ref_classes == ref_class
            counts[i, j] = np.sum(mapped_as_i * ref_is_j)
    return pd.DataFrame(counts, columns=all_classes, index=all_classes)


def _expand_error_matrix(mat, map_col, ref_col):
    """
    Converts an error matrix into a dataframe of points' ref classes and map
    classes. Used to verify that we get the same results as the paper, while
    keeping the class init parameters the same as for the Stehman version.
    """
    map_values = []
    ref_values = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ij = mat.iloc[i, j]
            for _ in range(ij):
                map_values.append(mat.index[i])
                ref_values.append(mat.index[j])
    return pd.DataFrame({map_col: map_values, ref_col: ref_values})


class Olofsson2014AccAssessment():
    """
    Based on:
    Olofsson, P., et al., 2014 "Good practices for estimating are and   
    assessing accuracy of land change", Remote Sensing of Environment. Vol
    148 pp. 42-57 10.1016/j.rse.2014/02/015
    """

    def __init__(self, data, map_col, ref_col,  mapped_population):
        self.N = np.sum(list(mapped_population.values()))
        self.mapped_proportions = {
            k: v / self.N for k, v in iter(mapped_population.items())
        }

        mapped_classes = data[map_col].values
        ref_classes = data[ref_col].values

        self.all_classes = np.unique(ref_classes)
        matrix = build_error_matrix(mapped_classes, ref_classes)
        self._error_matrix_counts = matrix

        # convert error matrix counts to proportions
        props_as_df = pd.DataFrame(
            self.mapped_proportions, index=self.mapped_proportions.keys()
        ).T
        _matrix = matrix.divide(matrix.sum(1), axis="index")
        self._error_matrix = _matrix.multiply(props_as_df)

    def overall_accuracy(self):
        """ equation 1 after substituting with equation 4 """
        acc = np.sum(np.diagonal(self._error_matrix))
        vs = {i: self.users_accuracy(i)[1]["var"] for i in self.all_classes}
        var = np.sum([
            (self.mapped_proportions[i] ** 2) * vs[i]
            for i in self.all_classes
        ])
        return acc, {"var": var, "std_err": np.sqrt(var)}

    def users_accuracy(self, i):
        """ equation 2 after substituting with equation 4
        the propotion of the area mapped as class i that has reference class i
        """
        correct = self._error_matrix.loc[i, i]
        incorrect = np.sum(self._error_matrix.loc[i, :])
        acc = correct / incorrect
        n_i = np.sum(self._error_matrix_counts.loc[i, :])
        var = acc * (1 - acc) / (n_i - 1)
        return acc, {"var": var, "std_err": np.sqrt(var)}

    def commission_error_rate(self, i):
        return 1 - self.users_accuracy(i)[0]

    def producers_accuracy(self, j):
        """ equation 3 after substituting with equation 4
        proportion of the area of reference class j that is mapped as class j
        """
        correct = self._error_matrix.loc[j, j]
        incorrect = np.sum(self._error_matrix.loc[:, j])
        acc = correct / incorrect

        N_hat_j = 0
        b = 0
        for i, N_i in iter(self.mapped_proportions.items()):
            n_i = np.sum(self._error_matrix_counts.loc[i, :])
            n_ij = self._error_matrix_counts.loc[i, j]
            N_hat_j += (N_i / n_i) * n_ij

            if i != j:
                b += (N_i ** 2) * n_ij / n_i * (1 - (n_ij / n_i)) / (n_i - 1)

        N_j = self.mapped_proportions[j]
        users_acc, _ = self.users_accuracy(j)
        n_j = np.sum(self._error_matrix_counts.loc[j, :])

        a = ((N_j ** 2) * ((1 - acc) ** 2) * users_acc * (1 - users_acc))
        a /= n_j - 1

        var = (1 / (N_hat_j ** 2)) * (a + ((acc ** 2) * b))
        return acc, {"var": var, "std_err": np.sqrt(var)}

    def omission_error_rate(self, j):
        return 1 - self.producers_accuracy(j)[0]

    def proportion_area(self, k):
        """ equation 9 and equation 10 """
        area = 0
        var = 0
        for i in self.all_classes:
            W_i  = self.mapped_proportions[i]
            n_i = np.sum(self._error_matrix_counts.loc[i, :])
            p_ik = W_i * self._error_matrix_counts.loc[i, k] / n_i
            area += p_ik
            var += (W_i * p_ik - (p_ik ** 2)) / (n_i - 1)
        return area, {"var": var, "std_err": np.sqrt(var)}

    def area(self, k):
        p_k, stats = self.proportion_area(k)
        area = self.N * p_k
        std_err = self.N * stats["std_err"]
        return self.N * p_k, {"var": std_err ** 2, "std_err": std_err}

    def error_matrix(self):
        return self._error_matrix

if __name__ == "__main__":
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

    print("--------------USERS ACC----------------")
    for k in df.columns:
        users_acc, stats = assessment.users_accuracy(k)
        se = stats["std_err"]
        print(f"{k}: {users_acc:.2g} +/- {1.96 * se:.2g}")

    print("\n-------------PRODUCERS ACC-------------")
    for k in df.columns:
        prods_acc, stats = assessment.producers_accuracy(k)
        se = stats["std_err"]
        print(f"{k}: {prods_acc:.2g} +/- {1.96 * se:.2g}")

    print("\n--------ESTIMATED AREA (pixels)--------")
    for k in df.columns:
        area, stats = assessment.area(k)
        se = stats["std_err"]
        print(f"{k}: {area} +/- {1.96 * se}")

    overall_acc, stats = assessment.overall_accuracy()
    se = stats["std_err"]
    print(f"\noverall accuracy: {overall_acc:.2g} +/- {1.96 * se:.2g}\n")

    print(assessment.error_matrix())


