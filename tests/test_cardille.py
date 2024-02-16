import pandas as pd
import numpy as np
from acc_assessment.cardille import Cardille

assessment = Cardille(
    pd.read_csv("~/school/accuracyAssessmentTools/tests/map_data_table.csv"),
    pd.read_csv("~/school/accuracyAssessmentTools/tests/ref_data_table.csv"),
    id_col="id",
    strata_col="strata",
    strata_population={"a": 500, "f": 500, "w": 50},
)
print(assessment.point_weights)
print("ag producers", assessment.producers_accuracy(0))
print("ag users", assessment.users_accuracy(0))
print("f producers", assessment.producers_accuracy(1))
print("f users", assessment.users_accuracy(1))
print("w producers", assessment.producers_accuracy(2))
print("w users", assessment.users_accuracy(2))
print(assessment.error_matrix())
print(np.sum(np.sum(assessment.error_matrix())))
