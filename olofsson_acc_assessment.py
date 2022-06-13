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


class Olofsson2014AccAssessment():
    """
    Based on:
    Olofsson, P., et al., 2014 "Good practices for estimating are and assessing
    accuracy of land change", Remote Sensing of Environment. Vol 148 pp. 42-57
    10.1016/j.rse.2014/02/015
    """

    def __init__(self, data, mapped_population, calc_error_matrix=False,
                 map_col=None, ref_col=None):
        self.N = np.sum(list(mapped_population.values()))
        self.mapped_proportions = {
            k: v / self.N for k, v in iter(mapped_population.items())
        }

        props_as_df = pd.DataFrame(
            self.mapped_proportions, index=self.mapped_proportions.keys()
        ).T

        if calc_error_matrix:
            mapped_classes = data[map_col].values
            ref_classes = data[ref_col].values

            self.all_classes = np.unique(ref_classes)
            matrix = build_error_matrix(mapped_classes, ref_classes)
            self.error_matrix_counts = matrix

            # convert error matrix counts to proportions
            _matrix = matrix.divide(matrix.sum(1), axis="index")
            self.error_matrix = _matrix.multiply(props_as_df)
        else:
            self.all_classes = data.columns
            self.error_matrix_counts = data

            # convert error matrix counts to proportions
            _data = data.divide(data.sum(1), axis="index")
            self.error_matrix = _data.multiply(props_as_df)

    def overall_accuracy(self):
        """ equation 1 after substituting with equation 4 """
        acc = np.sum(np.diagonal(self.error_matrix))
        var = np.sum([
            (self.mapped_proportions[i] ** 2) * self.users_accuracy(i)["var"]
            for i in self.all_classes
        ])
        return {"acc": acc, "var": var, "std_err": np.sqrt(var)}

    def users_accuracy(self, i):
        """ equation 2 after substituting with equation 4
        the propotion of the area mapped as class i that has reference class i
        """
        acc = self.error_matrix.loc[i, i] / np.sum(self.error_matrix.loc[i, :])
        n_i = np.sum(self.error_matrix_counts.loc[i, :])
        var = acc * (1 - acc) / (n_i - 1)
        return {"acc": acc, "var": var, "std_err": np.sqrt(var)}

    def commission_error_rate(self, i):
        return 1 - self.users_accuracy(i)

    def producers_accuracy(self, j):
        """ equation 3 after substituting with equation 4
        proportion of the area of reference class j that is mapped as class j
        """
        acc = self.error_matrix.loc[j, j] / np.sum(self.error_matrix.loc[:, j])

        N_hat_j = 0
        b = 0
        for i, N_i in iter(self.mapped_proportions.items()):
            n_i = np.sum(self.error_matrix_counts.loc[i, :])
            n_ij = self.error_matrix_counts.loc[i, j]
            N_hat_j += (N_i / n_i) * n_ij

            if i != j:
                b += (N_i ** 2) * (n_ij / n_i) * (1 - (n_ij / n_i)) / (n_i - 1)

        N_j = self.mapped_proportions[j]
        users_acc = self.users_accuracy(j)["acc"]
        n_j = np.sum(self.error_matrix_counts.loc[j, :])

        a = ((N_j ** 2) * ((1 - acc) ** 2) * users_acc * (1 - users_acc))
        a /= n_j - 1

        var = (1 / (N_hat_j ** 2)) * (a + ((acc ** 2) * b))
        return {"acc": acc, "var": var, "std_err": np.sqrt(var)}

    def omission_error_rate(self, j):
        return 1 - self.producers_accuracy(j)

    def proportion_area(self, k):
        """ equation 9 and equation 10 """
        area = 0
        var = 0
        for i in self.all_classes:
            W_i  = self.mapped_proportions[i]
            n_i = np.sum(self.error_matrix_counts.loc[i, :])
            p_ik = W_i * self.error_matrix_counts.loc[i, k] / n_i
            area += p_ik
            var += (W_i * p_ik - (p_ik ** 2)) / (n_i - 1)
        return {"proportion_area": area, "var": var, "std_err": np.sqrt(var)}

    def area(self, k):
        p_k = self.proportion_area(k)
        area = self.N * p_k["proportion_area"]
        std_err = self.N * p_k["std_err"]
        return {"area": area, "var": None, "std_err": std_err}

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
    assessment = Olofsson2014AccAssessment(df, mapped_area)

    print("--------------USERS ACC----------------")
    for k in df.columns:
        users_acc = assessment.users_accuracy(k)
        acc = users_acc["acc"]
        se = users_acc["std_err"]
        print(f"{k}: {acc:.2g} +/- {1.96 * se:.2g}")

    print("\n-------------PRODUCERS ACC-------------")
    for k in df.columns:
        prods_acc = assessment.producers_accuracy(k)
        acc = prods_acc["acc"]
        se = prods_acc["std_err"]
        print(f"{k}: {acc:.2g} +/- {1.96 * se:.2g}")

    print("\n--------ESTIMATED AREA (pixels)--------")
    for k in df.columns:
        area = assessment.area(k)
        A = area["area"]
        se = area["std_err"]
        print(f"{k}: {A} +/- {1.96 * se}")

    overall_acc = assessment.overall_accuracy()
    acc = overall_acc["acc"]
    se = overall_acc["std_err"]
    print(f"\noverall accuracy: {acc:.2g} +/- {1.96 * se:.2g}\n")

    print(assessment.error_matrix)


