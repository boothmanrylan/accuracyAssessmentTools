import numpy as np
import pandas as pd

from acc_assessment.stehman import Stehman
from acc_assessment.utils import shannon_diversity


class Cardille(Stehman):
    """
    Performs a Stehman accuracy assessment with points contribution to accuracy
    weighted based on model and reference confidence in their class value.
    """

    def __init__(
        self,
        map_data,
        ref_data,
        strata_col,
        id_col,
        strata_population,
        score_function=shannon_diversity,
        combine_function=np.multiply,
    ):
        """
        map_data: pd.DataFrame, containing the map class probabilities for
            each point along with a point id and the stratum the point was
            sampled from.
        ref_data: pd.DataFrame, containing the reference class probabilities
            for each point along with a point id and the stratum the point was
            sampled from.
        strata_col: str, name of column in map_data and ref_data that contains
            the label of the strata each point was sampled from.
        id_col: str, name of the column in map_data and ref_data that contains
            the unique id of each point.
        strata_population: dict, mapping each of the strata classes to the
            total number of pixels in each stratum across the entire study area.
        score_function: function, applied to class probabilities in map_data
            and ref_data to compute a "trustworthiness" of the point class.
        combine_function: function, applied to the result of calling
            score_function on  map_data and ref_data to get the weight of a
            point.
        """
        # ensure that map_data and ref_data have the same shape
        assert np.all(map_data.shape == ref_data.shape)

        # ensure the rows in both map_data and ref_data are in the same order
        assert (map_data[id_col] == ref_data[id_col]).all()

        # ensure the cols in both map_data and ref_data are in the same order
        assert (map_data.columns == ref_data.columns).all()

        self.strata_classes = map_data[strata_col].values

        class_names = [x for x in ref_data.columns if x not in [strata_col, id_col]]
        map_data = map_data[class_names]
        ref_data = ref_data[class_names]

        map_scores = map_data.apply(score_function, axis=1, raw=True)
        ref_scores = ref_data.apply(score_function, axis=1, raw=True)
        self.point_weights = combine_function(map_scores, ref_scores)

        self.map_classes = map_data.apply(np.argmax, axis=1, raw=True).values
        self.ref_classes = ref_data.apply(np.argmax, axis=1, raw=True).values

        self.strata_population = {
            k: v for k, v in iter(strata_population.items())
            if k in np.unique(self.strata_classes)
        }

        self.N = np.sum(list(self.strata_population.values()))

    def _unbiased_estimator(self, y_u, return_by_strata=False):
        y_u *= self.point_weights
        Y = self._Y_bar_hat(y_u, return_by_strata)
        se = self._se_Y_bar_hat(y_u, return_by_strata)
        return Y, se

    def _calculate_n_star_h(self, selector):
        return np.sum(selector.astype(float) * self.point_weights)
