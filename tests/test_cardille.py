import pytest
import pandas as pd
import numpy as np
from acc_assessment.cardille import Cardille
from acc_assessment.stehman import Stehman
from acc_assessment.utils import _expand_probabilities
from utils import check_within_tolerance

def _scale_class_probs(df, max_prob=0.7):
    num_classes = df.shape[1]
    min_prob = (1 - max_prob) / (num_classes - 1)
    return df.replace(1, max_prob).replace(0, min_prob)

@pytest.fixture
def stehman_table():
    return pd.read_csv("./tests/stehman2014_table2.csv", skiprows=1)

@pytest.fixture
def strata_totals():
    return {1: 40000, 2: 30000, 3: 20000, 4: 10000}

@pytest.fixture
def scaled():
    return False

@pytest.fixture
def cardille_assessment(stehman_table, strata_totals, scaled):
    map_table, ref_table = _expand_probabilities(
        stehman_table,
        "Map class",
        "Reference class",
        "Stratum",
    )

    if scaled:
        non_class_names = ["Stratum", "id"]
        non_class_columns = map_table[non_class_names]
        class_columns = [x for x in map_table.columns if x not in non_class_names]

        map_classes = _scale_class_probs(map_table[class_columns])
        ref_classes = _scale_class_probs(ref_table[class_columns])

        map_table = pd.concat([map_classes, non_class_columns], axis=1)
        ref_table = pd.concat([ref_classes, non_class_columns], axis=1)

    return Cardille(
        map_table,
        ref_table,
        id_col="id",
        strata_col="Stratum",
        strata_population=strata_totals,
    )

@pytest.fixture
def stehman_assessment(stehman_table, strata_totals):
    return Stehman(
        stehman_table,
        "Stratum",
        "Map class",
        "Reference class",
        strata_totals,
    )


@pytest.mark.parametrize("scaled", [True, False])
class TestCardilleSameAsStehman():
    """
    When we take class predictions and convert them to probabilities by
    one-hot-encoding them, the Cardille assessment of those probabilities
    should give us the same values as the Stehman assessment of the pure
    classes.
    """
    def test_overall_acc(self, stehman_assessment, cardille_assessment):
        cardille_overall, _ = cardille_assessment.overall_accuracy()
        stehman_overall, _ = stehman_assessment.overall_accuracy()
        check_within_tolerance(cardille_overall, stehman_overall)

    def test_error_matrix(self, stehman_assessment, cardille_assessment):
        cardille_error_matrix = cardille_assessment.error_matrix()
        stehman_error_matrix = stehman_assessment.error_matrix()
        assert np.allclose(cardille_error_matrix, stehman_error_matrix)

    def test_users_acc(self, stehman_assessment, cardille_assessment):
        classes = ["A", "B", "C", "D"]
        cardille_users = [
            cardille_assessment.users_accuracy(x)[0] for x in classes
        ]
        stehman_users = [
            stehman_assessment.users_accuracy(x)[0] for x in classes
        ]
        assert np.allclose(cardille_users, stehman_users)

    def test_producers_acc(self, stehman_assessment, cardille_assessment):
        classes = ["A", "B", "C", "D"]
        cardille_prods = [
            cardille_assessment.producers_accuracy(x)[0] for x in classes
        ]
        stehman_prods = [
            stehman_assessment.producers_accuracy(x)[0] for x in classes
        ]
        assert np.allclose(cardille_prods, stehman_prods)

# assessment = Cardille(
#     pd.read_csv("~/school/accuracyAssessmentTools/tests/map_data_table.csv"),
#     pd.read_csv("~/school/accuracyAssessmentTools/tests/ref_data_table.csv"),
#     id_col="id",
#     strata_col="strata",
#     strata_population={"a": 500000, "f": 500000, "w": 5000},
# )
# print(assessment.point_weights)
# print("ag producers", assessment.producers_accuracy(0))
# print("ag users", assessment.users_accuracy(0))
# print("f producers", assessment.producers_accuracy(1))
# print("f users", assessment.users_accuracy(1))
# print("w producers", assessment.producers_accuracy(2))
# print("w users", assessment.users_accuracy(2))
# print(assessment.error_matrix())
# print(np.sum(np.sum(assessment.error_matrix())))
