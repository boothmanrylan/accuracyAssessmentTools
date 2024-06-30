import pytest
import pandas as pd
import numpy as np

from acc_assessment.stehman import Stehman
from acc_assessment.olofsson import Olofsson


TOLERANCE = 1e-4
CLASSES = ["A", "B", "C", "D"]


@pytest.fixture()
def assessments():
    data = pd.read_csv("./tests/stehman2014_table2.csv", skiprows=1)

    population = dict(zip(CLASSES, [40000, 30000, 20000, 10000]))

    stehman_assessment = Stehman(
        data,
        map_col="Map class",
        ref_col="Reference class",
        strata_col="Map class",  # make the map == the strata
        strata_population=population,
    )

    olofsson_assessment = Olofsson(
        data,
        map_col="Map class",
        ref_col="Reference class",
        mapped_population=population,
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
