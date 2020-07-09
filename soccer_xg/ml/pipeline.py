import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        assert isinstance(columns, list)
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class TolerantLE(LabelEncoder):
    def transform(self, y):
        return np.searchsorted(self.classes_, y)


class UniqueCountColumnSelector(BaseEstimator, TransformerMixin):
    """
    To select those columns whose unique-count values are between
    lowerbound (inclusive) and upperbound (exclusive)
    """

    def __init__(self, lowerbound, upperbound):
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def fit(self, X, y=None):
        counts = X.apply(lambda vect: vect.unique().shape[0])
        self.columns = counts.index[
            counts.between(self.lowerbound, self.upperbound + 1)
        ]
        return self

    def transform(self, X):
        return X[self.columns]


class ColumnApplier(BaseEstimator, TransformerMixin):
    """
    Some sklearn transformers can apply only on ONE column at a time
    Wrap them with ColumnApplier to apply on all the dataset
    """

    def __init__(self, underlying):
        self.underlying = underlying

    def fit(self, X, y=None):
        m = {}
        X = pd.DataFrame(X)  # TODO: :( reimplement in pure numpy?
        for c in X.columns:
            k = clone(self.underlying)
            k.fit(X[c])
            m[c] = k
        self._column_stages = m
        return self

    def transform(self, X):
        ret = {}
        X = pd.DataFrame(X)
        for c, k in self._column_stages.items():
            ret[c] = k.transform(X[c])
        return pd.DataFrame(ret)[X.columns]  # keep the same order


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode the categorical value by natural number based on alphabetical order
    N/A are encoded to -2
    rare values to -1
    Very similar to TolerentLabelEncoder
    TODO: improve the implementation
    """

    def __init__(self, min_support):
        self.min_support = min_support
        self.vc = {}

    def _mapping(self, vc):
        mapping = {}
        for i, v in enumerate(vc[vc >= self.min_support].index):
            mapping[v] = i
        for v in vc.index[vc < self.min_support]:
            mapping[v] = -1
        mapping['nan'] = -2
        return mapping

    def _transform_column(self, x):
        x = x.astype(str)
        vc = self.vc[x.name]

        mapping = self._mapping(vc)

        output = pd.DataFrame()
        output[x.name] = x.map(
            lambda a: mapping[a] if a in mapping.keys() else -3
        )
        output.index = x.index
        return output.astype(int)

    def fit(self, x, y=None):
        x = x.astype(str)
        self.vc = dict((c, x[c].value_counts()) for c in x.columns)
        return self

    def transform(self, df):
        if len(df[df.index.duplicated()]):
            print(df[df.index.duplicated()].index)
            raise ValueError('Input contains duplicate index')
        dfs = [self._transform_column(df[c]) for c in df.columns]
        out = pd.DataFrame(index=df.index)
        for df in dfs:
            out = out.join(df)
        return out.values


class CountFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode the value by their frequency observed in the training set
    """

    def __init__(self, min_card=5, count_na=False):
        self.min_card = min_card
        self.count_na = count_na
        self.vc = None

    def fit(self, x, y=None):
        x = pd.Series(x)
        vc = x.value_counts()
        self.others_count = vc[vc < self.min_card].sum()
        self.vc = vc[vc >= self.min_card].to_dict()
        self.num_na = x.isnull().sum()
        return self

    def transform(self, x):
        vc = self.vc
        output = x.map(lambda a: vc.get(a, self.others_count))
        if self.count_na:
            output = output.fillna(self.num_na)
        return output.values


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Boxcox transformation for numerical columns
    To make them more Gaussian-like
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.shift = 0.0001

    def fit(self, x, y=None):
        x = x.values.reshape(-1, 1)
        x = self.scaler.fit_transform(x) + self.shift
        self.boxcox_lmbda = stats.boxcox(x)[1]
        return self

    def transform(self, x):
        x = x.values.reshape(-1, 1)
        scaled = np.maximum(self.shift, self.scaler.transform(x) + self.shift)
        ret = stats.boxcox(scaled, self.boxcox_lmbda)
        return ret[:, 0]


class Logify(BaseEstimator, TransformerMixin):
    """
    Log transformation
    """

    def __init__(self):
        self.shift = 2

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.log10(x - x.min() + self.shift)


class YToLog(BaseEstimator, TransformerMixin):
    """
    Transforming Y to log before fitting
    and transforming back the prediction to real values before return
    """

    def __init__(self, delegate, shift=0):
        self.delegate = delegate
        self.shift = shift

    def fit(self, X, y):
        logy = np.log(y + self.shift)
        self.delegate.fit(X, logy)
        return self

    def predict(self, X):
        pred = self.delegate.predict(X)
        return np.exp(pred) - self.shift


class FillNaN(BaseEstimator, TransformerMixin):
    def __init__(self, replace):
        self.replace = replace

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.fillna(self.replace)


class AsString(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.astype(str)


class StackedModel(BaseEstimator, TransformerMixin):
    def __init__(self, delegate, cv=5, method='predict_proba'):
        self.delegate = delegate
        self.cv = cv
        self.method = method

    def fit(self, X, y):
        raise Exception

    def fit_transform(self, X, y):
        a = cross_val_predict(
            self.delegate, X, y, cv=self.cv, method=self.method
        )
        self.delegate.fit(X, y)
        if len(a.shape) == 1:
            a = a.reshape(-1, 1)
        return a

    def transform(self, X):
        if self.method == 'predict_proba':
            return self.delegate.predict_proba(X)
        else:
            return self.delegate.predict(X).reshape(-1, 1)


class To1D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values.reshape(-1)


class TolerantLabelEncoder(TransformerMixin):
    """ LabelEncoder is not tolerant to unseen values
    """

    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, x, y=None):
        assert len(x.shape) == 1
        vc = x.value_counts()
        vc = vc[vc > self.min_count]
        self.values = {
            value: (1 + index) for index, value in enumerate(vc.index)
        }
        return self

    def transform(self, x):
        values = self.values
        return x.map(lambda a: values.get(a, 0))

    def inverse_transform(self, y):
        if not hasattr(self, 'inversed_mapping'):
            self.inversed_mapping = {v: k for k, v in self.values.items()}
            self.inversed_mapping[0] = None
        mapping = self.inversed_mapping
        return pd.Series(y).map(lambda a: mapping[a])


class SVD_Embedding(TransformerMixin):
    def __init__(
        self, rowname, colname, valuename=None, svd_kwargs={'n_components': 10}
    ):
        self.rowname = rowname
        self.colname = colname
        self.valuename = valuename
        self.rowle = TolerantLabelEncoder()
        self.colle = TolerantLabelEncoder()
        self.svd = TruncatedSVD(**svd_kwargs)

    def fit(self, X, y=None):
        row = self.rowle.fit_transform(X[self.rowname])
        col = self.colle.fit_transform(X[self.colname])
        if not self.valuename:
            data = np.ones(X.shape[0])
        else:
            data = X[self.valuename].groupby([row, col]).mean()
            row = data.index.get_level_values(0).values
            col = data.index.get_level_values(1).values
            data = data.values
        matrix = sp.coo_matrix((data, (row, col)))
        self.embedding = self.svd.fit_transform(matrix)
        return self

    def transform(self, X):
        row = self.rowle.transform(X[self.rowname])
        return self.embedding[row, :]
