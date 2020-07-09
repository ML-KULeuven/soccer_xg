from scipy.stats.distributions import randint, uniform
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import simple_proc_for_linear_algoritms


def mlp_gridsearch_classifier(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    """
    Simple classification pipeline using hyperband to optimize mlp hyper-parameters
    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _mlp_gridsearch_model(
        'classification',
        numeric_features,
        categoric_features,
        learning_rate,
        use_dask,
        n_iter,
        scoring,
    )


def mlp_gridsearch_regressor(
    numeric_features,
    categoric_features,
    learning_rate=0.08,
    use_dask=False,
    n_iter=100,
    scoring='roc_auc',
):
    """
    Simple regression pipeline using hyperband to optimize mlp hyper-parameters
    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _mlp_gridsearch_model(
        'regression',
        numeric_features,
        categoric_features,
        learning_rate,
        use_dask,
        n_iter,
        scoring,
    )


def _mlp_gridsearch_model(
    task,
    numeric_features,
    categoric_features,
    learning_rate,
    use_dask,
    n_iter,
    scoring,
):
    param_space = {
        'clf__hidden_layer_sizes': [
            (24,),
            (12, 12),
            (6, 6, 6, 6),
            (4, 4, 4, 4, 4, 4),
            (12, 6, 3, 3),
        ],
        'clf__activation': ['relu', 'logistic', 'tanh'],
        'clf__batch_size': [16, 32, 64, 128, 256, 512],
        'clf__alpha': uniform(0.0001, 0.9),
        'clf__learning_rate': ['constant', 'adaptive'],
    }

    model = (
        MLPClassifier(learning_rate_init=learning_rate)
        if task == 'classification'
        else MLPRegressor(learning_rate_init=learning_rate)
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
