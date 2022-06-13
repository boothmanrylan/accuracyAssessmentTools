import pandas as pd
import numpy as np

class Stehman2014AccAssessment(): 
    """
    Based on:
    Stehman, S.V., 2014. "Estimating area and map accuracy for stratified
    random sampling when the strata are different from the map classes",
    International Journal of Remote Sensing. Vol. 35 (No. 13).
    10.1080/01431161.2014.930207
    """

    def __init__(self, data, strata_col, map_col, ref_col, strata_population):
        """
        data should be a dataframe with columns named strata_col, map_col and
        ref_col containing the strata class, map class, and reference class of
        each point respectively.

        strata_population should be a dictionary mapping strata classes to the
        total number of pixels in that stratum in the study area.
        """
        self.map_classes = data[map_col].values
        self.ref_classes = data[ref_col].values
        self.strata_classes = data[strata_col].values

        self.N = np.sum(list(strata_population.values()))

        self.strata_population = strata_population

    def _indicator_func(self, map_val=None, ref_val=None):
        """ Calculate the indicator function y_u """
        if map_val is not None and ref_val is None:
            return self.map_classes == map_val
        elif ref_val is not None and map_val is None:
            return self.ref_classes == ref_val
        elif ref_val is not None and map_val is not None:
            return (self.ref_classes == ref_val) * (self.map_classes == map_val)
        else:
            return self.map_classes == self.ref_classes

    def _Y_bar_hat(self, y_u):
        """ equation 3 """
        total = 0
        for h, N_star_h in iter(self.strata_population.items()):
            n_star_h = np.sum(self.strata_classes == h)
            y_bar_h = np.sum(y_u * (self.strata_classes == h) / n_star_h)
            total += N_star_h * y_bar_h / self.N
        return total

    def _sample_var_Y_bar_hat(self, y_u, h):
        """ equation 26 """
        selector = self.strata_classes == h
        n_star_h = np.sum(selector)
        y_bar_h = np.sum((y_u / n_star_h) * selector)
        return np.sum((((y_u - y_bar_h) ** 2) / (n_star_h - 1)) * selector)

    def _var_Y_bar_hat(self, y_u):
        """ equation 25 """
        total = 0
        for h, N_star_h in iter(self.strata_population.items()):
            n_star_h = np.sum(self.strata_classes == h)
            s2_yh = self._sample_var_Y_bar_hat(y_u, h)
            a = N_star_h ** 2
            b = (1 - (n_star_h / N_star_h)) # can be skipped b/c very small
            c = s2_yh / n_star_h
            total += (a * b * c)
        ans = (1 / (self.N ** 2)) * total
        return total if total >= 0 else 0 # if N is large will overflow

    def _unbiased_estimator(self, y_u):
        Y = self._Y_bar_hat(y_u)
        var = self._var_Y_bar_hat(y_u)
        return {"Y": Y, "var": var, "std_err": np.sqrt(var)}

    def overall_accuracy(self):
        """ get the unbiased overall accuracy estimate """
        y_u = self._indicator_func()
        return self._unbiased_estimator(y_u)

    def PkA_estimate(self, k):
        """ get the area estimate for reference class k """
        y_u = self._indicator_func(ref_val=k)
        return self._unbiased_estimator(y_u)

    def Pij_estimate(self, i, j):
        """ get the proportion of area for map class i and ref class j """
        y_u = self._indicator_func(map_val=i, ref_val=j)
        return self._unbiased_estimator(y_u)

    def _R_hat(self, y_u, x_u):
        """ equation 27 """
        numerator_total = 0
        denominator_total = 0
        for h, N_star_h in iter(self.strata_population.items()):
            y_bar_h = np.sum(y_u * (self.strata_classes == h))
            x_bar_h = np.sum(x_u * (self.strata_classes == h))
            numerator_total += N_star_h * y_bar_h
            denominator_total += N_star_h * x_bar_h
        return numerator_total / denominator_total

    def _var_R_hat(self, y_u, x_u):
        """ equation 28 """
        R = self._R_hat(y_u, x_u)
        X_hat = 0
        total = 0
        for h, N_star_h in iter(self.strata_population.items()):
            selector = self.strata_classes == h

            s2_yh = self._sample_var_Y_bar_hat(y_u, h)
            s2_xh = self._sample_var_Y_bar_hat(x_u, h)

            n_star_h = np.sum(selector)

            X_hat += N_star_h * np.sum(x_u * selector) / n_star_h

            y_bar_h = np.sum(y_u * selector) / n_star_h
            x_bar_h = np.sum(x_u * selector) / n_star_h

            # equation 29
            s_xyh = np.sum(
                selector * (y_u - y_bar_h) * (x_u - x_bar_h) / (n_star_h - 1)
            )

            a = N_star_h ** 2
            b = 1 - (n_star_h / N_star_h) # can be skipped b/c very small
            c = s2_yh + ((R ** 2) * s2_xh) - (2 * R * s_xyh)

            total += (a * b * c) / n_star_h
        return total / (X_hat ** 2)

    def _design_consistent_estimator(self, y_u, x_u):
        R = self._R_hat(y_u, x_u)
        var = self._var_R_hat(y_u, x_u)
        return {"R": R, "var": var, "std_err": np.sqrt(var)}

    def users_accuracy(self, k):
        """ users accuracy for class k """
        y_u = self._indicator_func() * self._indicator_func(map_val=k)
        x_u = self._indicator_func(map_val=k)
        return self._design_consistent_estimator(y_u, x_u)

    def producers_accuracy(self, k):
        """ producers accuracy for class k """
        y_u = self._indicator_func() * self._indicator_func(ref_val=k)
        x_u = self._indicator_func(ref_val=k)
        return self._design_consistent_estimator(y_u, x_u)

    def commission_error_rate(self, k):
        """ commission error rate for class k """
        y_u = (self._indicator_func() == 0) * self._indicator_func(map_val=k)
        x_u = self._indicator_func(map_val=k)
        return self._design_consistent_estimator(y_u, x_u)

    def omission_error_rate(self, k):
        """ omission error rate for class k """
        y_u = (self._indicator_func() == 0) * self._indicator_func(ref_val=k)
        x_u = self._indicator_func(ref_val=k)
        return self._design_consistent_estimator(y_u, x_u)

    def error_matrix(self):
        """ complete error matrix """
        all_map_classes = np.unique(self.map_classes)
        all_ref_classes = np.unique(self.ref_classes)
        matrix = np.zeros((all_map_classes.shape[0], all_ref_classes.shape[0]))
        for i, map_class in enumerate(all_map_classes):
            for j, ref_class in enumerate(all_ref_classes):
                matrix[i, j] = self.Pij_estimate(map_class, ref_class)["Y"]
        return matrix

if __name__ == "__main__":
    df = pd.read_csv("./stehman2014_table2.csv", skiprows=1)
    stratum_totals = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    accAssessment = Stehman2014AccAssessment(
        df, "Stratum", "Map class", "Reference class", stratum_totals
    )

    print("Area of class A:\n", accAssessment.PkA_estimate("A"))
    print("Area of class C:\n", accAssessment.PkA_estimate("C"))
    print("Overall Accuracy:\n", accAssessment.overall_accuracy())
    print("User acc class B:\n", accAssessment.users_accuracy("B"))
    print("Producers Acc class B:\n", accAssessment.producers_accuracy("B"))
    print("Cell 2, 3 of error matrix:\n", accAssessment.Pij_estimate("B", "C"))
    print("Error Matrix:\n", accAssessment.error_matrix())
