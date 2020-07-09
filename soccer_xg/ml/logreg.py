import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from soccer_xg.ml.preprocessing import simple_proc_for_linear_algoritms


def logreg_gridsearch_classifier(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    """
    Simple classification pipeline using hyperband to optimize logreg hyper-parameters
    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _logreg_gridsearch_model(
        'classification',
        numeric_features,
        categoric_features,
        learning_rate,
        use_dask,
        n_iter,
        scoring,
    )


def logreg_gridsearch_regressor(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    """
    Simple regression pipeline using hyperband to optimize logreg hyper-parameters
    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _logreg_gridsearch_model(
        'regression',
        numeric_features,
        categoric_features,
        learning_rate,
        use_dask,
        n_iter,
        scoring,
    )


def _logreg_gridsearch_model(
    task,
    numeric_features,
    categoric_features,
    learning_rate,
    use_dask,
    n_iter,
    scoring,
):
    if learning_rate is None:
        param_space = {
            'clf__C': np.logspace(-5, 5, 100),
            'clf__class_weight': ['balanced', None],
        }
        model = LogisticRegression(max_iter=10000, fit_intercept=False)
    else:
        param_space = {
            'clf__penalty': ['l1', 'l2'],
            'clf__alpha': np.logspace(-5, 5, 100),
            'clf__class_weight': ['balanced', None],
        }
        learning_rate_schedule = (
            'constant' if isinstance(learning_rate, float) else learning_rate
        )
        eta0 = learning_rate if isinstance(learning_rate, float) else 0
        model = SGDClassifier(
            learning_rate=learning_rate_schedule,
            eta0=eta0,
            loss='log',
            max_iter=10000,
            fit_intercept=False,
        )

    pipe = Pipeline(
        [
            (
                'preprocessing',
                simple_proc_for_linear_algoritms(
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
