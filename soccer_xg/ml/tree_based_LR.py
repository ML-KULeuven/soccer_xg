import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder


# class for the tree-based/logistic regression pipeline
# see: https://gdmarmerola.github.io/probability-calibration/
class TreeBasedLR(BaseEstimator, ClassifierMixin):

    # initialization
    def __init__(self, forest, lr):

        # configuring the models
        self.forest = forest
        self.lr = lr

    # method for fitting the model
    def fit(self, X, y, sample_weight=None, fit_params={}):

        self.classes_ = np.unique(y)

        # first, we fit our tree-based model on the dataset
        self.forest.fit(X, y, **fit_params)

        # then, we apply the model to the data in order to get the leave indexes
        # if self.forest_model == 'cat':
        # leaves = self.forest.calc_leaf_indexes(X)
        # else:
        leaves = self.forest.named_steps['clf'].apply(
            self.forest.named_steps['preprocessing'].transform(X)
        )

        # then, we one-hot encode the leave indexes so we can use them in the logistic regression
        self.encoder = OneHotEncoder(sparse=True)
        leaves_encoded = self.encoder.fit_transform(leaves)

        # and fit it to the encoded leaves
        self.lr.fit(leaves_encoded, y)

    # method for predicting probabilities
    def predict_proba(self, X):

        # then, we apply the model to the data in order to get the leave indexes
        # if self.forest_model == 'cat':
        # leaves = self.forest.calc_leaf_indexes(X)
        # else:
        leaves = self.forest.named_steps['clf'].apply(
            self.forest.named_steps['preprocessing'].transform(X)
        )

        # then, we one-hot encode the leave indexes so we can use them in the logistic regression
        leaves_encoded = self.encoder.transform(leaves)

        # and fit it to the encoded leaves
        y_hat = self.lr.predict_proba(leaves_encoded)

        # retuning probabilities
        return y_hat

    # get_params, needed for sklearn estimators
    def get_params(self, deep=True):
        return {
            'forest': self.forest,
            'lr': self.lr,
        }
