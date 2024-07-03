import pytest
import numpy as np
import pandas as pd

from acc_assessment.utils import expand_error_matrix


def test_expand_error_matrix_from_dict():
    error_matrices = {
        "A": pd.DataFrame(
            [[2, 1],
             [1, 2]],
            index=["F", "NF"],
            columns=["F", "NF"]
        ),
        "B": pd.DataFrame(
            [[4, 0],
             [3, 10]],
            index=["F", "NF"],
            columns=["F", "NF"],
        )
    }
    longform = expand_error_matrix(
        error_matrices,
        map_col="map",
        ref_col="ref",
        strata_col="strata",
    )

    assert longform.shape[0] == 23
    assert longform.shape[1] == 3
    assert longform.loc[longform["strata"] == "A"].shape[0] == 6
    assert longform.loc[longform["strata"] == "B"].shape[0] == 17
    assert longform.loc[longform["map"] == "F"].shape[0] == 7
    assert longform.loc[longform["map"] == "NF"].shape[0] == 16
    assert longform.loc[longform["ref"] == "F"].shape[0] == 10
    assert longform.loc[longform["ref"] == "NF"].shape[0] == 13
