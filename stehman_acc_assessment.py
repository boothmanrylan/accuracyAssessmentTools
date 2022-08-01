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
            ref_matches = self.ref_classes == ref_val
            map_matches = self.map_classes == map_val
            return ref_matches * map_matches
        else:
            return self.map_classes == self.ref_classes

    def _Y_bar_hat(self, y_u, return_by_strata=False):
        """ equation 3 """
        total = 0
        by_strata = {}
        for h, N_star_h in iter(self.strata_population.items()):
            n_star_h = np.sum(self.strata_classes == h)
            y_bar_h = np.sum(y_u * (self.strata_classes == h)) / n_star_h
            curr = N_star_h * y_bar_h / self.N
            total += curr
            by_strata[h] = curr

        if return_by_strata:
            return by_strata
        return total

    def _sample_var_Y_bar_hat(self, y_u, h):
        """ equation 26 """
        selector = self.strata_classes == h
        n_star_h = np.sum(selector)
        y_bar_h = np.sum((y_u / n_star_h) * selector)
        return np.sum((((y_u - y_bar_h) ** 2) / (n_star_h - 1)) * selector)

    def _se_Y_bar_hat(self, y_u, return_by_strata=False):
        """ equation 25 """
        try:
            assert 1 / (self.N ** 2) > 0
        except AssertionError as E:
            msg = "integer overflow, self.N is likely too large"
            raise ValueError(msg) from E

        total = 0
        by_strata = {}
        for h, N_star_h in iter(self.strata_population.items()):
            n_star_h = np.sum(self.strata_classes == h)
            s2_yh = self._sample_var_Y_bar_hat(y_u, h)
            a = N_star_h ** 2
            b = 1 - (n_star_h / N_star_h) # can be skipped b/c ~1
            c = s2_yh / n_star_h
            curr = a * b * c / (self.N ** 2)
            total += curr
            by_strata[h] = curr

        if return_by_strata:
            return {k: np.sqrt(v) for k, v in iter(by_strata.items())}
        return np.sqrt(total)

    def _unbiased_estimator(self, y_u, return_by_strata=False):
        Y = self._Y_bar_hat(y_u, return_by_strata)
        se = self._se_Y_bar_hat(y_u, return_by_strata)
        return Y, se

    def overall_accuracy(self):
        """ get the unbiased overall accuracy estimate """
        y_u = self._indicator_func()
        return self._unbiased_estimator(y_u)

    def PkA_estimate(self, k, return_area=False, return_by_strata=False):
        """ get the proportion of area estimate for reference class k """
        y_u = self._indicator_func(ref_val=k)
        pka, se = self._unbiased_estimator(y_u, return_by_strata)

        if return_by_strata and return_area:
            pkas = {k: self.N * v for k, v in iter(pka.items())}
            ses = {k: self.N * v for k, v in iter(se.items())}
            return pkas, ses
        elif return_area:
            return self.N * pka, self.N * se
        else:
            return pka, se

    def Pij_estimate(self, i, j, return_by_strata=False):
        """ get the proportion of area for map class i and ref class j """
        y_u = self._indicator_func(map_val=i, ref_val=j)
        return self._unbiased_estimator(y_u, return_by_strata)

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
            b = 1 - (n_star_h / N_star_h) # can be skipped b/c ~1
            c = s2_yh + ((R ** 2) * s2_xh) - (2 * R * s_xyh)

            total += (a * b * c) / n_star_h
        return total / (X_hat ** 2)

    def _design_consistent_estimator(self, y_u, x_u):
        R = self._R_hat(y_u, x_u)
        var = self._var_R_hat(y_u, x_u)
        return R, np.sqrt(var)

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
        """ returns error matrix of class proportions """
        all_map_classes = np.unique(self.map_classes)
        all_ref_classes = np.unique(self.ref_classes)
        matrix = np.zeros(
            (all_map_classes.shape[0], all_ref_classes.shape[0]))
        for i, map_class in enumerate(all_map_classes):
            for j, ref_class in enumerate(all_ref_classes):
                matrix[i, j] = self.Pij_estimate(map_class, ref_class)[0]
        return pd.DataFrame(matrix, columns=all_map_classes,
                index=all_map_classes)

    def area(self, i):
        """ estimate the total area of class k """
        pij, se = self.Pij_estimate(i, i)
        return self.N * pij, self.N * se
        # pka, se = self.PkA_estimate(i)
        # return self.N * pka, self.N * se
        total_proportion = 0
        total_var = 0
        for j in np.unique(self.ref_classes):
            pij, se = self.Pij_estimate(i, j)
            total_proportion += pij
            total_var += (se ** 2)
        return self.N * total_proportion, self.N * np.sqrt(total_var)

    def area_by_strata(self, i, mapped=False, reference=False, correct=False):
        """ estimate the area of class i within each of the strata """
        try:
            assert(sum([int(mapped), int(reference), int(correct)]) == 1)
        except AssertionError as E:
            msg = "exactly 1 of mapped, reference, and correct must be true"
            raise ValueError(msg) from E

        if mapped:
            raise NotImplementedError

        if reference:
            return self.PkA_estimate(i, True, True)

        if correct:
            pijs, ses = self.Pij(i, i, return_by_strata=True)
            return {k: (self.N * p, self.N * s) for k, v in iter(ses.items())}


if __name__ == "__main__":
    df = pd.read_csv("./stehman2014_table2.csv", skiprows=1)
    stratum_totals = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    assessment = Stehman2014AccAssessment(
        df, "Stratum", "Map class", "Reference class", stratum_totals
    )

    # ===============================================================
    # THESE ARE THE VALUES GIVEN IN THE PAPER
    # ==============================================================
    prop_class_A = 0.35
    prop_class_C = 0.20
    overall_accuracy = 0.63
    users_class_B = 0.574
    producers_class_B = 0.794
    cell_2_3 = 0.08
    se_prop_class_A = 0.082
    se_prop_class_C = 0.064
    se_overall_accuracy = 0.085
    se_users_class_B = 0.125
    se_producers_class_B = 0.114

    test, se = assessment.PkA_estimate("A")
    print(f"Area of class A:\t{test:.3f}, SE: {se:.3f}", end=" | ")
    print(f"EXPECTED: {prop_class_A}, {se_prop_class_A}")

    test, se = assessment.PkA_estimate("C")
    print(f"Area of class C:\t{test:.3f}, SE: {se:.3f}", end=" | ")
    print(f"EXPECTED: {prop_class_C}, {se_prop_class_C}")

    test, se = assessment.overall_accuracy()
    print(f"Overall Accuracy:\t{test:.3f}, SE: {se:.3f}", end=" | ")
    print(f"EXPECTED: {overall_accuracy}, {se_overall_accuracy}")

    test, se = assessment.users_accuracy("B")
    print(f"User acc class B:\t{test:.3f}, SE: {se:.3f}", end=" | ")
    print(f"EXPECTED: {users_class_B}, {se_users_class_B}")

    test, se = assessment.producers_accuracy("B")
    print(f"Producers Acc class B:\t{test:.3f}, SE: {se:.3f}", end=" | ")
    print(f"EXPECTED: {producers_class_B}, {se_producers_class_B}")

    test, se = assessment.Pij_estimate("B", "C")
    print(f"Cell 2, 3 of error matrix: {test:.3f}, SE: {se:.3f}", end=" | ")
    print(f"EXPECTED: {cell_2_3}, not given")

    print("Error Matrix:\n", assessment.error_matrix())
