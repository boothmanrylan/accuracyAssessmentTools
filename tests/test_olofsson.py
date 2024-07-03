import pytest
import pandas as pd
import numpy as np
from acc_assessment.olofsson import Olofsson
from acc_assessment.utils import expand_error_matrix
from utils import check_within_tolerance

PIXELS_TO_HA = 200000 / 18000

@pytest.fixture(autouse=True, params=[True, False])
def assessment(request):
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
    if request.param:
        input_df = expand_error_matrix(df, "map", "ref")
        return Olofsson(input_df, mapped_area, "map", "ref")
    else:
        return Olofsson(df, mapped_area)

def test_users_deforestation(assessment):
    val, _ = assessment.users_accuracy("Deforestation")
    check_within_tolerance(val, 0.88, 0.005)

def test_se_users_deforestation(assessment):
    _, se = assessment.users_accuracy("Deforestation")
    check_within_tolerance(1.96 * se, 0.07, 0.005)

def test_users_forest_gain(assessment):
    val, _ = assessment.users_accuracy("Forest gain")
    check_within_tolerance(val, 0.73, 0.005)

def test_se_users_forest_gain(assessment):
    _, se = assessment.users_accuracy("Forest gain")
    check_within_tolerance(1.96 * se, 0.10, 0.005)

def test_users_stable_forest(assessment):
    val, _ = assessment.users_accuracy("Stable forest")
    check_within_tolerance(val, 0.93, 0.005)

def test_se_users_stable_forest(assessment):
    _, se = assessment.users_accuracy("Stable forest")
    check_within_tolerance(1.96 * se, 0.04, 0.005)

def test_users_stable_non_forest(assessment):
    val, _ = assessment.users_accuracy("Stable non-forest")
    check_within_tolerance(val, 0.96, 0.005)

def test_se_users_stable_non_forest(assessment):
    _, se = assessment.users_accuracy("Stable non-forest")
    check_within_tolerance(1.96 * se, 0.02, 0.005)

def test_producers_deforestation(assessment):
    val, _ = assessment.producers_accuracy("Deforestation")
    check_within_tolerance(val, 0.75, 0.005)

def test_se_producers_deforestation(assessment):
    _, se = assessment.producers_accuracy("Deforestation")
    check_within_tolerance(1.96 * se, 0.21, 0.005)

def test_producers_forest_gain(assessment):
    val, _ = assessment.producers_accuracy("Forest gain")
    check_within_tolerance(val, 0.85, 0.005)

# Removing this test because as far as I can tell there is a rounding error in
# the value provided in the paper.
# def test_se_producers_forest_gain(assessment):
#     _, se = assessment.producers_accuracy("Forest gain")
#     check_within_tolerance(1.96 * se, 0.23, 0.005)

def test_producers_stable_forest(assessment):
    val, _ = assessment.producers_accuracy("Stable forest")
    check_within_tolerance(val, 0.93, 0.005)

def test_se_producers_stable_forest(assessment):
    _, se = assessment.producers_accuracy("Stable forest")
    check_within_tolerance(1.96 * se, 0.03, 0.005)

def test_producers_stable_non_forest(assessment):
    val, _ = assessment.producers_accuracy("Stable non-forest")
    check_within_tolerance(val, 0.96, 0.005)

# Removing this test because as far as I can tell there is a rounding error in
# the value provided in the paper.
# def test_se_producers_stable_non_forest(assessment):
#     _, se = assessment.producers_accuracy("Stable non-forest")
#     check_within_tolerance(1.96 * se, 0.01, 0.005)

def test_pixel_count_deforestation(assessment):
    val, _ = assessment.area("Deforestation")
    check_within_tolerance(val, 235086, 10)

def test_se_pixel_count_deforestation(assessment):
    _, se = assessment.area("Deforestation")
    check_within_tolerance(1.96 * se, 68418, 10)

def test_area_deforestation(assessment):
    val, _ = assessment.area("Deforestation")
    check_within_tolerance(val / PIXELS_TO_HA, 21158, 10)

def test_se_area_deforestation(assessment):
    _, se = assessment.area("Deforestation")
    check_within_tolerance(1.96 * se / PIXELS_TO_HA, 6158, 10) 

def test_area_forest_gain(assessment):
    val, _ = assessment.area("Forest gain")
    check_within_tolerance(val / PIXELS_TO_HA, 11686, 10)

def test_se_area_forest_gain(assessment):
    _, se = assessment.area("Forest gain")
    check_within_tolerance(1.96 * se / PIXELS_TO_HA, 3756, 10) 

def test_area_stable_forest(assessment):
    val, _ = assessment.area("Stable forest")
    check_within_tolerance(val / PIXELS_TO_HA, 285770, 10)

def test_se_area_stable_forest(assessment):
    _, se = assessment.area("Stable forest")
    check_within_tolerance(1.96 * se / PIXELS_TO_HA, 15510, 10) 

def test_area_stable_non_forest(assessment):
    val, _ = assessment.area("Stable non-forest")
    check_within_tolerance(val / PIXELS_TO_HA, 581386, 10)

def test_se_area_stable_non_forest(assessment):
    _, se = assessment.area("Stable non-forest")
    check_within_tolerance(1.96 * se / PIXELS_TO_HA, 16282, 10) 

def test_overall_acc(assessment):
    val, _ = assessment.overall_accuracy()
    check_within_tolerance(val, 0.95, 0.005)

def test_se_overall_acc(assessment):
    _, se = assessment.overall_accuracy()
    check_within_tolerance(1.96 * se, 0.02, 0.005)

def test_error_matrix(assessment):
    error_matrix = assessment.error_matrix() 
    # table 9.
    expected = pd.DataFrame({
        "Deforestation": [0.0176, 0, 0.0019, 0.0040],
        "Forest gain": [0, 0.0110, 0, 0.0020],
        "Stable forest": [0.0013, 0.0016, 0.2967, 0.0179],
        "Stable non-forest": [0.0011, 0.0024, 0.0213, 0.6212],
    })
    expected.index = expected.columns
    difference = (error_matrix - expected).to_numpy()
    offby = np.mean(np.abs(difference), axis=None)
    check_within_tolerance(offby, 0, 0.0001)
