import pytest
import pandas as pd
import numpy as np
from acc_assessment.cardille import Cardille
from acc_assessment.stehman import Stehman
from acc_assessment.utils import _expand_probabilities
from utils import check_within_tolerance

TOLERANCE = 1e-4

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

@pytest.fixture
def example_cardille_assessment():
    data = pd.read_csv("./tests/cardille_table.csv", skiprows=1)
    ref_data = data.filter(like='.').copy()
    model_data = data.drop(ref_data.columns, axis=1)
    ref_data.columns = [x.split('.')[0] for x in ref_data.columns]

    # the spreadsheet does not account for class sizes therefore assume simple
    # random sampling from the entire map
    stratum = ["A"] * data.shape[0]
    strata_pop = {"A": 100}

    uid = np.arange(model_data.shape[0])
    model_data['stratum'] = stratum
    model_data['id'] = uid
    ref_data['stratum'] = stratum
    ref_data['id'] = uid

    return Cardille(
        model_data,
        ref_data,
        id_col='id',
        strata_col='stratum',
        strata_population=strata_pop
    )

def test_spreadsheet_users_acc_0(example_cardille_assessment):
    val, _ = example_cardille_assessment.users_accuracy("Class 1")
    check_within_tolerance(val, 0.9321, tolerance=TOLERANCE)

def test_spreadsheet_users_acc_1(example_cardille_assessment):
    val, _ = example_cardille_assessment.users_accuracy("Class 2")
    check_within_tolerance(val, 0.8954, tolerance=TOLERANCE)

def test_spreadsheet_users_acc_2(example_cardille_assessment):
    val, _ = example_cardille_assessment.users_accuracy("Class 3")
    check_within_tolerance(val, 0.9819, tolerance=TOLERANCE)

def test_spreadsheet_producers_acc_0(example_cardille_assessment):
    val, _ = example_cardille_assessment.producers_accuracy("Class 1")
    check_within_tolerance(val, 0.9533, tolerance=TOLERANCE)

def test_spreadsheet_producers_acc_1(example_cardille_assessment):
    val, _ = example_cardille_assessment.producers_accuracy("Class 2")
    check_within_tolerance(val, 0.9193, tolerance=TOLERANCE)

def test_spreadsheet_producers_acc_2(example_cardille_assessment):
    val, _ = example_cardille_assessment.producers_accuracy("Class 3")
    check_within_tolerance(val, 0.9011, tolerance=TOLERANCE)
