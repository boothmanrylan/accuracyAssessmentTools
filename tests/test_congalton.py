import pytest
import pandas as pd
from acc_assessment.congalton import Congalton
from acc_assessment.utils import expand_error_matrix
from utils import check_within_tolerance

@pytest.fixture(autouse=True)
def assessment():
    data = [[65, 4, 22, 24], [6, 81, 5, 8], [0, 11, 85, 19], [4, 7, 3, 90]]
    classes = ["D", "C", "BA", "SB"]
    df = pd.DataFrame(data, index=classes, columns=classes)
    input_df = pd.DataFrame(expand_error_matrix(df, "map", "ref"),)
    return Congalton(input_df, "map", "ref")

def test_overall_acc(assessment):
    val, _ = assessment.overall_accuracy()
    check_within_tolerance(val, 0.74, 0.005)

def test_users_D(assessment):
    val, _ = assessment.users_accuracy("D")
    check_within_tolerance(val, 0.57, 0.005)

def test_users_C(assessment):
    val, _ = assessment.users_accuracy("C")
    check_within_tolerance(val, 0.81, 0.005)

def test_users_BA(assessment):
    val, _ = assessment.users_accuracy("BA")
    check_within_tolerance(val, 0.74, 0.005)

def test_users_SB(assessment):
    val, _ = assessment.users_accuracy("SB")
    check_within_tolerance(val, 0.87, 0.005)

def test_producers_D(assessment):
    val, _ = assessment.producers_accuracy("D")
    check_within_tolerance(val, 0.87, 0.005)

def test_producers_C(assessment):
    val, _ = assessment.producers_accuracy("C")
    check_within_tolerance(val, 0.79, 0.005)

def test_producers_BA(assessment):
    val, _ = assessment.producers_accuracy("BA")
    check_within_tolerance(val, 0.74, 0.005)

def test_producers_SB(assessment):
    val, _ = assessment.producers_accuracy("SB")
    check_within_tolerance(val, 0.64, 0.005)
