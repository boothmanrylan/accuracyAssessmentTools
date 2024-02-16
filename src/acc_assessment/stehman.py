import pandas as pd
import numpy as np

from acc_assessment.utils import users_accuracy_error, producers_accuracy_error

class Stehman(): 
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

        self.strata_population = {
            k: v for k, v in iter(strata_population.items())
            if k in np.unique(self.strata_classes)
        }

        self.N = np.sum(list(self.strata_population.values()))

    def _calculate_n_star_h(self, selector):
        return np.sum(selector)

    def _indicator_func(self, map_val=None, ref_val=None):
        """ Calculate the indicator function y_u """
        if map_val is not None and ref_val is None:
            output = self.map_classes == map_val
        elif ref_val is not None and map_val is None:
            output = self.ref_classes == ref_val
        elif ref_val is not None and map_val is not None:
            ref_matches = self.ref_classes == ref_val
            map_matches = self.map_classes == map_val
            output = ref_matches * map_matches
        else:
            output = self.map_classes == self.ref_classes
        return output.astype(float)

    def _Y_bar_hat(self, y_u, return_by_strata=False):
        """ equation 3 """
        total = 0
        by_strata = {}
        for h, N_star_h in iter(self.strata_population.items()):
            selector = (self.strata_classes == h).astype(float)
            n_star_h = self._calculate_n_star_h(selector)
            y_bar_h = np.sum(y_u * selector) / n_star_h
            curr = N_star_h * y_bar_h / self.N
            total += curr
            by_strata[h] = curr

        if return_by_strata:
            return by_strata
        return total

    def _sample_var_Y_bar_hat(self, y_u, h):
        """ equation 26 """
        selector = self.strata_classes == h
        n_star_h = self._calculate_n_star_h(selector)
        y_bar_h = np.sum((y_u / n_star_h) * selector)
        if n_star_h - 1 == 0:
            # there will be a divide by zero runtime warning
            # assume that the variance is zero since there is only one point
            print(f'Stratum {h} has only one member, assuming var is 0')
            return 0
        else:
            output = np.sum((((y_u - y_bar_h) ** 2) / (n_star_h - 1)) * selector)
            return output

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
            n_star_h = self._calculate_n_star_h(self.strata_classes == h)
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

            n_star_h = self._calculate_n_star_h(selector)

            X_hat += N_star_h * np.sum(x_u * selector) / n_star_h

            y_bar_h = np.sum(y_u * selector) / n_star_h
            x_bar_h = np.sum(x_u * selector) / n_star_h

            if n_star_h - 1 == 0:
                # there will be a divide by zero warning 
                # assume the sample covariance is 0 since there is only one
                # point
                s_xyh = 0
                print(f'Stratum {h} has only one member, assuming var is 0')
            else: 
                # equation 29
                a = selector * (y_u - y_bar_h) * (x_u - x_bar_h)
                s_xyh = np.sum(a / (n_star_h - 1))

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
        if np.sum(x_u) == 0:
            users_accuracy_error(k)
        return self._design_consistent_estimator(y_u, x_u)

    def producers_accuracy(self, k):
        """ producers accuracy for class k """
        y_u = self._indicator_func() * self._indicator_func(ref_val=k)
        x_u = self._indicator_func(ref_val=k)
        if np.sum(x_u) == 0:
            producers_accuracy_error(k)
        return self._design_consistent_estimator(y_u, x_u)

    def commission_error_rate(self, k):
        """ commission error rate for class k """
        y_u = (self._indicator_func() == 0) * self._indicator_func(map_val=k)
        x_u = self._indicator_func(map_val=k)
        if np.sum(x_u) == 0:
            users_accuracy_error(k)
        return self._design_consistent_estimator(y_u, x_u)

    def omission_error_rate(self, k):
        """ omission error rate for class k """
        y_u = (self._indicator_func() == 0) * self._indicator_func(ref_val=k)
        x_u = self._indicator_func(ref_val=k)
        if np.sum(x_u) == 0:
            producers_accuracy_error(k)
        return self._design_consistent_estimator(y_u, x_u)

    def error_matrix(self):
        """ returns error matrix of class proportions """
        all_map_classes = np.unique(self.map_classes)
        all_ref_classes = np.unique(self.ref_classes)
        all_classes = np.unique(np.hstack([all_map_classes, all_ref_classes]))
        matrix = np.zeros((all_classes.shape[0], all_classes.shape[0]))
        for i, map_class in enumerate(all_classes):
            for j, ref_class in enumerate(all_classes):
                matrix[i, j] = self.Pij_estimate(map_class, ref_class)[0]
        return pd.DataFrame(matrix, columns=all_classes, index=all_classes)

    def area(self, i, mapped=False, reference=True, correct=False):
        """ estimate the area of class i
        
        If mapped is true, returns the area that was mapped as class i.
        If reference is true, returns the estimate of the true area of class i.
        If correct is true, returns the estimate of the area that was correctly
        mapped as class i.
        """
        try:
            assert(sum([int(mapped), int(reference), int(correct)]) == 1)
        except AssertionError as E:
            msg = "exactly 1 of mapped, reference, and correct must be true"
            raise ValueError(msg) from E
        
        if mapped:
            raise NotImplementedError

        if reference:
            return self.PkA_estimate(i, return_area=True)

        if correct:
            pij, se = self.Pij_estimate(i, i)
            return self.N * pij, self.N * se

    def area_by_strata(self, i, mapped=False, reference=True, correct=False):
        """ estimate the area of class i within each of the strata

        If mapped is true, returns the area that was mapped as class i.
        If reference is true, returns the estimate of the true area of class i.
        If correct is true, returns the estimate of the area that was correctly
        mapped as class i.
        """
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


