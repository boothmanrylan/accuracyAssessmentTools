import pytest
import pandas as pd
import numpy as np

from acc_assessment.stehman import Stehman
from acc_assessment.olofsson import Olofsson
from acc_assessment.utils import _expand_error_matrix


TOLERANCE = 1e-4
CLASSES = ["Deforestation", "Forest gain", "Stable forest", "Stable non-forest"]


@pytest.fixture()
def assessments():
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

    stehman_assessment = Stehman(
        input_df,
        map_col="map",
        ref_col="ref",
        strata_col="map",  # make the map == the strata
        strata_population=mapped_area,
    )

    olofsson_assessment = Olofsson(
        input_df,
        map_col="map",
        ref_col="ref",
        mapped_population=mapped_area,
    )

    return {"stehman": stehman_assessment, "olofsson": olofsson_assessment}


def test_overall(assessments):
    a, _ = assessments["stehman"].overall_accuracy()
    b, _ = assessments["olofsson"].overall_accuracy()
    assert np.allclose(a, b)


def test_overall_se(assessments):
    _, a = assessments["stehman"].overall_accuracy()
    _, b = assessments["olofsson"].overall_accuracy()
    print(a, b)
    assert np.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("class_label", CLASSES)
def test_users(assessments, class_label):
    a, _ = assessments["stehman"].users_accuracy(class_label)
    b, _ = assessments["olofsson"].users_accuracy(class_label)
    assert np.allclose(a, b)


@pytest.mark.parametrize("class_label", CLASSES)
def test_users_se(assessments, class_label):
    _, a = assessments["stehman"].users_accuracy(class_label)
    _, b = assessments["olofsson"].users_accuracy(class_label)
    print(a, b)
    assert np.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("class_label", CLASSES)
def test_producers(assessments, class_label):
    a, _ = assessments["stehman"].producers_accuracy(class_label)
    b, _ = assessments["olofsson"].producers_accuracy(class_label)
    assert np.allclose(a, b)


@pytest.mark.parametrize("class_label", CLASSES)
def test_producers_se(assessments, class_label):
    _, a = assessments["stehman"].producers_accuracy(class_label)
    _, b = assessments["olofsson"].producers_accuracy(class_label)
    print(a, b)
    assert np.allclose(a, b, atol=1e-4)
