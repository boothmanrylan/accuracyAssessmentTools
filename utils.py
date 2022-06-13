import numpy as np
import pandas as pd

def build_error_matrix(map_classes, ref_classes):
    all_classes = np.unique(map_classes)
    num_classes = len(all_classes)
    matrix = pd.DataFrame(np.zeros((num_classes, num_classes)),
        index=all_classes, columns=all_classes)
    for i in all_classes:
        for j in all_classes:
            mapped = (map_classes == i).astype(int)
            ref = (ref_classes == j).astype(int)
            matrix.loc[i, j] = np.sum(mapped * ref)
    return matrix

def _expand_error_matrix(mat, map_col, ref_col):
    """
    Converts an error matrix into a dataframe of points' ref classes and map
    classes. Used to verify that we get the same results as the paper, while
    keeping the class init parameters the same as for the Stehman version.
    """
    map_values = []
    ref_values = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ij = mat.iloc[i, j]
            for _ in range(ij):
                map_values.append(mat.index[i])
                ref_values.append(mat.index[j])
    return pd.DataFrame({map_col: map_values, ref_col: ref_values})

