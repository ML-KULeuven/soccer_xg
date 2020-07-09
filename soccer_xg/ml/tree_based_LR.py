import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


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


def xgboost_gridsearch_classifier(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    param_space = {
        'clf__max_depth': randint(2, 11),
        'clf__min_child_weight': randint(1, 11),
        'clf__subsample': uniform(0.5, 0.5),
        'clf__colsample_bytree': uniform(0.5, 0.5),
        'clf__colsample_bylevel': uniform(0.5, 0.5),
        'clf__gamma': uniform(0, 1),
        'clf__reg_alpha': uniform(0, 1),
        'clf__reg_lambda': uniform(0, 10),
        'clf__base_score': uniform(0.1, 0.9),
        'clf__scale_pos_weight': uniform(0.1, 9.9),
        'solver': 'sag',
        'C': 0.001756,
        'fit_intercept': False,
    }

    # configuring the model
    model = TreeBasedLR(tree_params, lr_params, forest_model='et')

    model = xgbsk.XGBClassifier(learning_rate=learning_rate)

    pipe = Pipeline(
        [
            (
                'preprocessing',
                simple_proc_for_tree_algoritms(
                    numeric_features, categoric_features
                ),
            ),
            ('clf', model),
        ]
    )

    if use_dask:
        from dask_ml.model_selection import RandomizedSearchCV

        return RandomizedSearchCV(
            pipe, param_space, n_iter=n_iter, scoring=scoring, cv=5
        )
    else:
        from sklearn.model_selection import RandomizedSearchCV

        return RandomizedSearchCV(
            pipe, param_space, n_iter=n_iter, scoring=scoring, cv=5
        )
