import pytest
import pandas as pd
from acc_assessment.stehman import Stehman
from utils import check_within_tolerance

@pytest.fixture(autouse=True)
def assessment():
    return Stehman(
        pd.read_csv("./tests/stehman2014_table2.csv", skiprows=1),
        "Stratum",
        "Map class", 
        "Reference class",
        {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    )

def test_proportion_of_A(assessment):
    val, _ = assessment.PkA_estimate("A")
    check_within_tolerance(val, 0.35)

def test_se_proportion_of_A(assessment):
    _, se = assessment.PkA_estimate("A")
    check_within_tolerance(se, 0.082)

def test_proportion_of_C(assessment):
    val, _ = assessment.PkA_estimate("C")
    check_within_tolerance(val, 0.20)

def test_se_proportion_of_C(assessment):
    _, se = assessment.PkA_estimate("C")
    check_within_tolerance(se, 0.064)

def test_overall_acc(assessment):
    val, _ = assessment.overall_accuracy()
    check_within_tolerance(val, 0.63)

def test_se_overall_acc(assessment):
    _, se = assessment.overall_accuracy()
    check_within_tolerance(se, 0.085)

def test_users_acc_of_B(assessment):
    val, _ = assessment.users_accuracy("B")
    check_within_tolerance(val, 0.574)

def test_se_users_acc_of_B(assessment):
    _, se = assessment.users_accuracy("B")
    check_within_tolerance(se, 0.125)

def test_producers_acc_of_B(assessment):
    val, _ = assessment.producers_accuracy("B")
    check_within_tolerance(val, 0.794)

def test_se_producers_acc_of_B(assessment):
    """
    NOTE: the value we are comparing against here is NOT the value that is
    given in the paper. The paper calculates the standard error for the
    producer's accuracy of class B as 0.114. However, this is not the correct
    value. The paper sets s2_xh (for h == 4) as 0 when calculating the
    producer's accuracy for class B when it should be 0.1. This is backed up in
    the paper in Table 3 where s2_xh (for h == 4) is correctly calculated as
    0.1. If we force s2_xh (for h == 4) to be 0 then we calculate the same
    value as the paper gives.
    """
    _, se = assessment.producers_accuracy("B")
    check_within_tolerance(se, 0.116)

def test_cell_2_3(assessment):
    val, _ = assessment.Pij_estimate("B", "C")
    check_within_tolerance(val, 0.08)
