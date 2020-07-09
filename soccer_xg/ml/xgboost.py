from scipy.stats.distributions import randint, uniform
from sklearn.pipeline import Pipeline
from xgboost import sklearn as xgbsk

from .preprocessing import simple_proc_for_tree_algoritms


def xgboost_gridsearch_classifier(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    """
    Simple classification pipeline using hyperband to optimize xgboost hyper-parameters
    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _xgboost_gridsearch_model(
        'classification',
        numeric_features,
        categoric_features,
        learning_rate,
        use_dask,
        n_iter,
        scoring,
    )


def xgboost_gridsearch_regressor(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    """
    Simple regression pipeline using hyperband to optimize xgboost hyper-parameters
    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _xgboost_gridsearch_model(
        'regression',
        numeric_features,
        categoric_features,
        learning_rate,
        use_dask,
        n_iter,
        scoring,
    )


def _xgboost_gridsearch_model(
    task,
    numeric_features,
    categoric_features,
    learning_rate,
    use_dask,
    n_iter,
    scoring,
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
    }

    model = (
        xgbsk.XGBClassifier(learning_rate=learning_rate)
        if task == 'classification'
        else xgbsk.XGBRegressor(learning_rate=learning_rate)
    )

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
