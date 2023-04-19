import pytest 
import numpy as np

def check_within_tolerance(prediction, true, tolerance=0.001):
    __tracebackhide__ = True
    if not np.abs(prediction - true) < tolerance:
        pytest.fail(
            f"Predicted value ({prediction}) and "
            f"true value ({true}) are not within {tolerance}"
        )
