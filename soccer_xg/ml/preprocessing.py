import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pipeline import AsString, ColumnsSelector, OrdinalEncoder


def simple_proc_for_tree_algoritms(numeric_features, categoric_features):
    """
    Create a simple preprocessing pipeline for tree based algorithms
    """

    catpipe = make_pipeline(
        ColumnsSelector(categoric_features),
        OrdinalEncoder(min_support=5)
        # ColumnApplier(FillNaN('nan')),
        # ColumnApplier(TolerantLabelEncoder())
    )
    numpipe = make_pipeline(
        ColumnsSelector(numeric_features),
        SimpleImputer(strategy='mean'),
        StandardScaler(),
    )
    if numeric_features and categoric_features:
        return make_union(catpipe, numpipe)
    elif numeric_features:
        return numpipe
    elif categoric_features:
        return catpipe
    raise Exception('Both variable lists are empty')


def simple_proc_for_linear_algoritms(numeric_features, categoric_features):
    """
    Create a simple preprocessing pipeline for linear algorithms
    """

    catpipe = make_pipeline(
        ColumnsSelector(categoric_features),
        AsString(),
        ce.OneHotEncoder()
        # ColumnApplier(FillNaN('nan')),
        # ColumnApplier(TolerantLabelEncoder())
    )
    numpipe = make_pipeline(
        ColumnsSelector(numeric_features),
        SimpleImputer(strategy='mean'),
        StandardScaler(),
    )
    if numeric_features and categoric_features:
        return make_union(catpipe, numpipe)
    elif numeric_features:
        return numpipe
    elif categoric_features:
        return catpipe
    raise Exception('Both variable lists are empty')
