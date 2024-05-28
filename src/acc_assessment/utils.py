import numpy as np
import pandas as pd

def build_error_matrix(map_classes, ref_classes):
    all_classes = np.unique(np.vstack([map_classes, ref_classes]))
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

def _expand_probabilities(data, map_col, ref_col, strata_col, id_col=None):
    """
    Converts a dataframe with a column containing the map class and a column
    containing the reference class into two dataframes one containing the
    one-hot encoded map classes and one containing the one-hot encoded ref
    classes

    Args:
        data: pd.DataFrame
        map_col: str, name of column in data that contains the map classes
        ref_col: str, name of column in data that contains the ref classes
        strata_col: str, name of column in data that contains the stratum a
            poin was sampled from.
        id_col: str, name of column in data that contains the unique id of the
            point, if not given row numbers are used as ids in a new column
            with the name "id".
    
    Returns:
        2-tuple of pd.DataFrames (map dataframe, reference dataframe)
    """
    # TODO: need a fallback if one of these is missing some class values,
    # ideally without adding another dependency besides pandas
    # For now just check that both the ref and the map contain the same unique
    # class values
    map_classes = pd.get_dummies(data[map_col])
    ref_classes = pd.get_dummies(data[ref_col])
    unique_class_vals = data[map_col].unique().sort()
    unique_ref_vals = data[ref_col].unique().sort()
    msg = "pd.get_dummies will fail b/c a class in ref is not represented in map"
    assert np.all(unique_class_vals == unique_ref_vals), msg

    strata_classes = data[strata_col]
    if id_col is not None:
        ids = data[id_col]
    else:
        id_col = "id"
        ids = data.index

    map_df = pd.concat(
        [
            map_classes,
            pd.DataFrame({strata_col: strata_classes, id_col: ids})
        ],
        axis=1,
    )

    ref_df = pd.concat(
        [
            ref_classes,
            pd.DataFrame({strata_col: strata_classes, id_col: ids})
        ],
        axis=1,
    )

    return map_df, ref_df

def pretty(matrix, class_names, total=False, accuracy=False):
    try:
        assert not (total and accuracy)
    except AssertionError as E:
        msg = "Can't calculate totals and accuracies simultaneously"
        raise ValueError(msg) from E

    if total:
       matrix = matrix.append(matrix.sum(0), ignore_index=True)
       matrix = matrix.T.append(matrix.sum(1), ignore_index=True).T
       matrix.index = pd.MultiIndex.from_product([
           ["Map"], class_names + ["Total"]
       ])
       matrix.columns = pd.MultiIndex.from_product([
            ["Reference"], class_names + ["Total"]
       ])
       return matrix

    if accuracy:
        true = np.diag(matrix)
        overall = pd.Series([np.sum(true) / matrix.sum().sum()])
        users = true / matrix.sum(0)
        producers = true / matrix.sum(1)
        producers_w_overall = producers.append(overall, ignore_index=True)

        matrix = matrix.append(users, ignore_index=True)
        matrix = matrix.T.append(producers_w_overall, ignore_index=True).T

        matrix.index = pd.MultiIndex.from_product([
            ["Map"], class_names + ["Producer's Acc."]
        ])
        matrix.columns = pd.MultiIndex.from_product([
            ["Reference"], class_names + ["User's Acc."]
        ])
        return matrix

    matrix.index = pd.MultiIndex.from_product([["Map"], class_names])
    matrix.columns = pd.MultiIndex.from_product([["Reference"], class_names])
    return matrix

def users_accuracy_error(k):
    msg = 'Cannot calculate user\'s accuracy/commission error rate '
    msg += f'for class {k} as it never appears as a mapped value'
    print(msg)

def producers_accuracy_error(k):
    msg = 'Cannot calculate producer\'s accuracy/ommission error rate '
    msg += f'for class {k} as it never appears as a reference value'
    print(msg)

def shannon_evenness(X):
    """ compute the shannon evenness of X

    Shannon evenness = (-sum xlnx for x in X) / (ln size of x)

    Args:
        X: np.ndarray

    Returns:
        number
    """
    # y = xln(x) approaches 0 when x approaches 0
    # can therefore replace ln(x) with 0 when x == 0
    X = X.astype(np.float32)
    lnX = np.log(X, out=np.zeros_like(X), where=(X != 0))
    return -1 * np.divide(np.sum(X * lnX), np.log(X.shape[0]))

def shannon_diversity(X):
    """ compute the shannon diversity of X

    Shannon diversity = 1 - shannon evenness

    Args:
        X: np.ndarray

    Returns:
        number
    """
    return 1 - shannon_evenness(X)

class AccuracyAssessment():
    """Base class for all assessment types.
    """
    def __repr__(self):
        def val_se(value, standard_error):
            output = f"{value:.4f}"
            if standard_error is not None:
                output += f"  +/- {standard_error:.4f}"
            return output

        seperator = "\n" + ("=" * 40) + "\n"

        output_string = seperator + "\tOVERALL ACCURACY" + seperator
        output_string += f"{val_se(*self.overall_accuracy())}\n"

        output_string += seperator + "\tUSER'S ACCURACIES" + seperator
        for c in self.all_classes:
            output_string += f"{c}:\t{val_se(*self.users_accuracy(c))}\n"

        output_string += seperator + "\tPRODUCER'S ACCURACIES" + seperator
        for c in self.all_classes:
            output_string += f"{c}:\t{val_se(*self.producers_accuracy(c))}\n"

        output_string += seperator + "\tERROR MATRIX" + seperator
        output_string += repr(self.error_matrix())
        return output_string

    def __str__(self):
        return repr(self)
