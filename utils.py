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
