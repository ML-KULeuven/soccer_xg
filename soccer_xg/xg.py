"""Tools for creating and analyzing xG models."""
import os

import joblib
import pandas as pd
import socceraction.spadl.config as spadlcfg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import NotFittedError
from soccer_xg import features as fs
from soccer_xg import metrics, utils
from soccer_xg.api import DataApi
from soccer_xg.ml.preprocessing import simple_proc_for_linear_algoritms
from tqdm import tqdm


class XGModel(object):
    """A wrapper around a pipeline for computing xG values.

    Parameters
    ----------
    copy_data : boolean (default=``True``)
        Whether or not to copy data when fitting and applying the model. Running the model
        in-place (``copy_data=False``) will be faster and have a smaller memory footprint,
        but if not done carefully can lead to data integrity issues.

    Attributes
    ----------
    model : A Scikit-learn pipeline (or equivalent)
        The actual model used to compute xG. Upon initialization it will be set to
        a default model, but can be overridden by the user.
    column_descriptions : dictionary
        A dictionary whose keys are the names of the columns used in the model, and the values are
        string descriptions of what the columns mean. Set at initialization to be the default model,
        if you create your own model you'll need to update this attribute manually.
    training_seasons : A list of tuples, or ``None`` (default=``None``)
        If the model was trained using data from the DataApi, a list of (competition_id, season_id) tuples
        used to train the model. If the DataApi was **not** used, an empty list. If no model
        has been trained yet, ``None``.
    validation_seasons : same as ``training_seasons``, but for validation data.
    sample_probabilities : A numpy array of floats or ``None`` (default=``None``)
        After the model has been validated, contains the sampled predicted probabilities used to
        compute the validation statistic.
    predicted_goal_percents : A numpy array of floats or ``None`` (default=``None``)
        After the model has been validated, contains the actual probabilities in the test
        set at each probability in ``sample_probabilities``.
    num_shots_used : A numpy array of floats or ``None`` (default=``None``)
        After the model has been validated, contains the number of shots used to compute each
        element of ``predicted_goal_percents``.
    model_directory : string
        The directory where all models will be saved to or loaded from.
    """

    model_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'models'
    )
    _default_model_filename = 'default_model.xg'

    def __init__(self, copy_data=True):
        self.copy_data = copy_data
        self.column_descriptions = None

        self.model = self.create_default_pipeline()
        self._fitted = False
        self._training_seasons = None
        self._validation_seasons = None

        self._sample_probabilities = None
        self._predicted_goal_percents = None
        self._num_shots_used = None

    @property
    def training_seasons(self):
        return self._training_seasons

    @property
    def validation_seasons(self):
        return self._validation_seasons

    @property
    def sample_probabilities(self):
        return self._sample_probabilities

    @property
    def predicted_goal_percents(self):
        return self._predicted_goal_percents

    @property
    def num_shots_used(self):
        return self._num_shots_used

    def train(
        self,
        source_data,
        training_seasons=(('ENG', '1617'), ('ENG', '1718')),
        target_colname='goal',
    ):
        """Train the model.

        Once a modeling pipeline is set up (either the default or something
        custom-generated), historical data needs to be fed into it in order to
        "fit" the model so that it can then be used to predict future results.
        This method implements a simple wrapper around the core Scikit-learn functionality
        which does this.

        The default is to use data from a DataApi object, however that can be changed
        to a simple Pandas DataFrame with precomputed features and labels if desired.

        There is no particular output from this function, rather the parameters governing
        the fit of the model are saved inside the model object itself. If you want to get an
        estimate of the quality of the fit, use the ``validate_model`` method after running
        this method.

        Parameters
        ----------
        source_data : ``DataApi`` or a Pandas DataFrame
            The data to be used to train the model. If an instance of
            ``DataApi`` is given, will query the api database for the training data.
        training_seasons : list of tuples (default=``[('ENG', '1617'), ('ENG', '1718')]``)
            What seasons to use to train the model if getting data from a DataApi instance.
            If ``source_data`` is not a ``DataApi``, this argument will be ignored.
            **NOTE:** it is critical not to use all possible data in order to train the
            model - some will need to be reserved for a final validation (see the
            ``validate_model`` method). A good dataset to reserve
            for validation is the most recent one or two seasons.
        target_colname : string or integer (default=``"goal"``)
            The name of the target variable column. This is only relevant if
            ``source_data`` is not a ``DataApi``.

        Returns
        -------
        ``None``
        """
        if isinstance(self.model, list):
            for model in self.model:
                model.train(source_data, training_seasons, target_colname)
        else:
            self._training_seasons = []
            if isinstance(source_data, DataApi):
                game_ids = source_data.games[
                    source_data.games.season_id.astype(str).isin(
                        [s[1] for s in training_seasons]
                    )
                    & source_data.games.competition_id.astype(str).isin(
                        [s[0] for s in training_seasons]
                    )
                ].index
                feature_cols = get_features(source_data, game_ids)
                target_col = get_labels(source_data, game_ids)
                self._training_seasons = training_seasons
            else:
                target_col = source_data[target_colname]
                feature_cols = source_data.drop(target_colname, axis=1)
            self.model.fit(feature_cols, target_col)
        self._fitted = True

    def validate(
        self,
        source_data,
        validation_seasons=(('ENG', '1819')),
        target_colname='goal',
        plot=True,
    ):
        """Validate the model.

        Once a modeling pipeline is trained, a different dataset must be fed into the trained model
        to validate the quality of the fit.
        This method implements a simple wrapper around the core Scikit-learn functionality
        which does this.

        The default is to use data from a DataApi object, however that can be changed
        to a simple Pandas DataFrame with precomputed features and labels if desired.

        The output of this method is a dictionary with relevant error metrics (see ``soccer_xg.metrics``).

        Parameters
        ----------
        source_data : ``DataApi`` or a Pandas DataFrame
            The data to be used to validate the model. If an instance of
            ``DataApi`` is given, will query the api database for the training data.
        validation_seasons : list of tuples (default=``[('ENG', '1819')]``)
            What seasons to use to validated the model if getting data from a DataApi instance.
            If ``source_data`` is not a ``DataApi``, this argument will be ignored.
            **NOTE:** it is critical not to use the same data to validate the model as was used
            in the fit. Generally a good data set to use for validation is one from a time
            period more recent than was used to train the model.
        target_colname : string or integer (default=``"goal"``)
            The name of the target variable column. This is only relevant if
            ``source_data`` is not a ``DataApi``.
        plot: bool (default=true)
            Whether to plot the AUROC and probability calibration curves.

        Returns
        -------
        dict
            Error metrics on the validation data.

        Raises
        ------
        NotFittedError
            If the model hasn't been fit.

        """
        if not self._fitted:
            raise NotFittedError('Must fit model before validating.')

        if isinstance(source_data, DataApi):
            game_ids = source_data.games[
                source_data.games.season_id.astype(str).isin(
                    [s[1] for s in validation_seasons]
                )
                & source_data.games.competition_id.astype(str).isin(
                    [s[0] for s in validation_seasons]
                )
            ].index
            target_col = get_labels(source_data, game_ids)
            self._validation_seasons = validation_seasons
        else:
            game_ids = None
            target_col = source_data[target_colname]
            self._validation_seasons = []

        df_predictions = self.estimate(source_data, game_ids)
        predicted_probabilities = df_predictions['xG']
        target_col = target_col.loc[df_predictions.index]

        (
            self._sample_probabilities,
            self._predicted_goal_percents,
            self._num_shots_used,
        ) = metrics.bayesian_calibration_curve(
            target_col.values, predicted_probabilities
        )

        # Compute the maximal deviation from a perfect prediction as well as the area under the
        # curve of the residual between |predicted - perfect|:
        max_deviation = metrics.max_deviation(
            self.sample_probabilities, self.predicted_goal_percents
        )
        residual_area = metrics.residual_area(
            self.sample_probabilities, self.predicted_goal_percents
        )
        roc = roc_auc_score(target_col, predicted_probabilities)
        brier = brier_score_loss(target_col, predicted_probabilities)
        ece = metrics.expected_calibration_error(
            target_col, predicted_probabilities, 10, 'uniform'
        )
        ace = metrics.expected_calibration_error(
            target_col, predicted_probabilities, 10, 'quantile'
        )

        if plot:
            import matplotlib.pyplot as plt
            from soccer_xg.visualisation import (
                plot_roc_curve,
                plot_calibration_curve,
            )

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_roc_curve(target_col, predicted_probabilities, axis=ax[0])
            plot_calibration_curve(
                target_col,
                predicted_probabilities,
                min_samples=100,
                axis=ax[1],
            )

        return {
            'max_dev': max_deviation,
            'residual_area': residual_area,
            'roc': roc,
            'brier': brier,
            'ece': ece,
            'ace': ace,
            'fig': fig if plot else None,
        }

    def estimate(self, source_data, game_ids=None):
        """Estimate the xG values for all shots in a set of games.

        The default is to use data from a DataApi object, however that can be changed
        to a simple Pandas DataFrame with precomputed features and labels if desired.

        Parameters
        ----------
        source_data : ``DataApi`` or a Pandas DataFrame
            The data to be used to validate the model. If an instance of
            ``DataApi`` is given, will query the api database for the training data.
        game_ids : list of ints (default=None)
            Only xG values for the games in this list are returned. By default,
            xG values are computed for all games in the source data.
            If ``source_data`` is not a ``DataApi``, this argument will be ignored.

        Returns
        -------
        A Pandas DataFrame
            A dataframe with a column 'xG', containing the predictted xG value
            of each shot in the given data, indexed by (game_id, action_id) of
            the corresponding shot.

        Raises
        ------
        NotFittedError
            If the model hasn't been fit.
        """
        if not self._fitted:
            raise NotFittedError('Must fit model before predicting WP.')

        if isinstance(self.model, list):
            xg = []
            for model in self.model:
                xg.append(model.estimate(source_data, game_ids))
            return pd.concat(xg).sort_index()
        else:
            if isinstance(source_data, DataApi):
                if game_ids is None:
                    game_ids = (
                        source_data.games.index
                        if game_ids is None
                        else game_ids
                    )
                source_data = get_features(source_data, game_ids)

            xg = pd.DataFrame(index=source_data.index)
            xg['xG'] = self.model.predict_proba(source_data)[:, 1]
            return xg

    def create_default_pipeline(self):
        """Create the default xG estimation pipeline.

        Returns
        -------
        Scikit-learn pipeline
            The default pipeline, suitable for computing xG
            but by no means the best possible model.
        """
        models = [OpenplayXGModel(), FreekickXGModel(), PenaltyXGModel()]
        self.column_descriptions = {
            m.__class__.__name__: m.column_descriptions for m in models
        }
        return models

    def save_model(self, filename=None):
        """Save the XGModel instance to disk.

        All models are saved to the same place, with the installed
        soccer_xg library (given by ``XGModel.model_directory``).

        Parameters
        ----------
        filename : string (default=None):
            The filename to use for the saved model. If this parameter
            is not specified, save to the default filename. Note that if a model
            already lists with this filename, it will be overwritten. Note also that
            this is a filename only, **not** a full path. If a full path is specified
            it is likely (albeit not guaranteed) to cause errors.

        Returns
        -------
        ``None``
        """
        if filename is None:
            filename = self._default_model_filename
        joblib.dump(self, os.path.join(self.model_directory, filename))

    @classmethod
    def load_model(cls, filename=None):
        """Load a saved XGModel.

        Parameters
        ----------
        filename : string (default=None):
            The filename to use for the saved model. If this parameter
            is not specified, load the default model. Note that
            this is a filename only, **not** a full path.

        Returns
        -------
        ``soccer_xg.XGModel`` instance.
        """
        if filename is None:
            filename = cls._default_model_filename

        return joblib.load(os.path.join(cls.model_directory, filename))


class OpenplayXGModel(XGModel):
    _default_model_filename = 'default_openplay_model.xg'

    def train(
        self,
        source_data,
        training_seasons=(('ENG', '1617'), ('ENG', '1718')),
        target_colname='goal',
    ):
        self._training_seasons = []
        if isinstance(source_data, DataApi):
            game_ids = source_data.games[
                source_data.games.season_id.astype(str).isin(
                    [s[1] for s in training_seasons]
                )
                & source_data.games.competition_id.astype(str).isin(
                    [s[0] for s in training_seasons]
                )
            ].index
            feature_cols = get_features(
                source_data, game_ids, shotfilter=OpenplayXGModel.filter_shots
            )
            target_col = get_labels(
                source_data, game_ids, shotfilter=OpenplayXGModel.filter_shots
            )
            self._training_seasons = training_seasons
        else:
            target_col = source_data[target_colname]
            feature_cols = source_data.drop(target_colname, axis=1)
        self.model.fit(feature_cols, target_col)
        self._fitted = True

    def estimate(self, source_data, game_ids=None):

        if isinstance(source_data, DataApi):
            game_ids = (
                source_data.games.index if game_ids is None else game_ids
            )
            source_data = get_features(
                source_data, game_ids, shotfilter=OpenplayXGModel.filter_shots
            )

        xg = pd.DataFrame(index=source_data.index)
        xg['xG'] = self.model.predict_proba(source_data)[:, 1]
        return xg

    def create_default_pipeline(self):
        bodypart_colname = 'bodypart_id_a0'
        dist_to_goal_colname = 'start_dist_to_goal_a0'
        angle_to_goal_colname = 'start_angle_to_goal_a0'

        self.column_descriptions = {
            bodypart_colname: 'Bodypart used for the shot (head, foot or other)',
            dist_to_goal_colname: 'Distance to goal',
            angle_to_goal_colname: 'Angle to goal',
        }

        preprocess_pipeline = simple_proc_for_linear_algoritms(
            [dist_to_goal_colname, angle_to_goal_colname], [bodypart_colname]
        )
        base_model = LogisticRegression(
            max_iter=10000, solver='lbfgs', fit_intercept=False
        )
        pipe = make_pipeline(preprocess_pipeline, base_model)
        return pipe

    @staticmethod
    def filter_shots(df_actions):
        shot_idx = (
            df_actions.type_name == 'shot'
        ) & df_actions.result_name.isin(['fail', 'success'])
        return shot_idx


class PenaltyXGModel(XGModel):
    _default_model_filename = 'default_penalty_model.xg'

    def __init__(self, copy_data=True):
        super().__init__(copy_data)
        self._fitted = True

    def train(
        self,
        source_data,
        training_seasons=(('ENG', '1617'), ('ENG', '1718')),
        target_colname='goal',
    ):
        pass

    def estimate(self, source_data, game_ids=None):

        if isinstance(source_data, DataApi):
            game_ids = (
                source_data.games.index if game_ids is None else game_ids
            )
            source_data = get_features(
                source_data,
                game_ids,
                xfns=[],
                shotfilter=PenaltyXGModel.filter_shots,
            )

        xg = pd.DataFrame(index=source_data.index)
        xg['xG'] = 0.792453

        return xg

    def create_default_pipeline(self):
        return None

    @staticmethod
    def filter_shots(df_actions):
        shot_idx = df_actions.type_name == 'shot_penalty'
        return shot_idx


class FreekickXGModel(XGModel):

    _default_model_filename = 'default_freekick_model.xg'

    def train(
        self,
        source_data,
        training_seasons=(('ENG', '1617'), ('ENG', '1718')),
        target_colname='goal',
    ):
        self._training_seasons = []
        if isinstance(source_data, DataApi):
            game_ids = source_data.games[
                source_data.games.season_id.astype(str).isin(
                    [s[1] for s in training_seasons]
                )
                & source_data.games.competition_id.astype(str).isin(
                    [s[0] for s in training_seasons]
                )
            ].index
            feature_cols = get_features(
                source_data, game_ids, shotfilter=FreekickXGModel.filter_shots
            )
            target_col = get_labels(
                source_data, game_ids, shotfilter=FreekickXGModel.filter_shots
            )
            self._training_seasons = training_seasons
        else:
            target_col = source_data[target_colname]
            feature_cols = source_data.drop(target_colname, axis=1)
        self.model.fit(feature_cols, target_col)
        self._fitted = True

    def estimate(self, source_data, game_ids=None):

        if isinstance(source_data, DataApi):
            game_ids = (
                source_data.games.index if game_ids is None else game_ids
            )
            source_data = get_features(
                source_data, game_ids, shotfilter=FreekickXGModel.filter_shots
            )

        xg = pd.DataFrame(index=source_data.index)
        xg['xG'] = self.model.predict_proba(source_data)[:, 1]
        return xg

    def create_default_pipeline(self):
        dist_to_goal_colname = 'start_dist_to_goal_a0'
        angle_to_goal_colname = 'start_angle_to_goal_a0'

        self.column_descriptions = {
            dist_to_goal_colname: 'Distance to goal',
            angle_to_goal_colname: 'Angle to goal',
        }

        preprocess_pipeline = simple_proc_for_linear_algoritms(
            [dist_to_goal_colname, angle_to_goal_colname], []
        )
        base_model = LogisticRegression(
            max_iter=10000, solver='lbfgs', fit_intercept=True
        )
        pipe = make_pipeline(preprocess_pipeline, base_model)
        return pipe

    @staticmethod
    def filter_shots(df_actions):
        shot_idx = df_actions.type_name == 'shot_freekick'
        return shot_idx


def get_features(
    api,
    game_ids=None,
    xfns=fs.all_features,
    shotfilter=None,
    nb_prev_actions=3,
):
    game_ids = api.games.index if game_ids is None else game_ids
    X = {}
    for game_id in tqdm(game_ids, desc=f'Generating features'):
        # try:
        game = api.games.loc[game_id]
        game_actions = utils.enhance_actions(api.get_actions(game_id))
        X[game_id] = _compute_features_game(
            game, game_actions, xfns, shotfilter, nb_prev_actions
        )
        X[game_id].index.name = 'action_id'
        X[game_id]['game_id'] = game_id
        # except Exception as e:
        # print(f"Failed for game with id={game_id}: {e}")
    X = pd.concat(X.values()).reset_index().set_index(['game_id', 'action_id'])
    # remove post-shot features (these will all have a single unique value)
    f = X.columns[X.nunique() > 1]
    return X[f]


def _compute_features_game(
    game, actions, xfns=fs.all_features, shotfilter=None, nb_prev_actions=3
):
    if shotfilter is None:
        # filter shots and ignore own goals
        shot_idx = actions.type_name.isin(
            ['shot', 'shot_penalty', 'shot_freekick']
        ) & actions.result_name.isin(['fail', 'success'])
    else:
        shot_idx = shotfilter(actions)
    if shot_idx.sum() < 1:
        return pd.DataFrame()
    if len(xfns) < 1:
        return pd.DataFrame(index=actions.index.values[shot_idx])
    # convert actions to gamestates
    gamestates = [
        states.loc[shot_idx].copy()
        for states in fs.gamestates(actions, nb_prev_actions)
    ]
    gamestates = fs.play_left_to_right(gamestates, game.home_team_id)
    # remove post-shot attributes
    gamestates[0].loc[shot_idx, 'end_x'] = float('NaN')
    gamestates[0].loc[shot_idx, 'end_y'] = float('NaN')
    gamestates[0].loc[shot_idx, 'result_id'] = float('NaN')
    # compute features
    X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
    # fix data types
    for c in [c for c in X.columns.values if c.startswith('type_id')]:
        X[c] = pd.Categorical(
            X[c].replace(spadlcfg.actiontypes_df().type_name.to_dict()),
            categories=spadlcfg.actiontypes,
            ordered=False,
        )
    for c in [c for c in X.columns.values if c.startswith('result_id')]:
        X[c] = pd.Categorical(
            X[c].replace(spadlcfg.results_df().result_name.to_dict()),
            categories=spadlcfg.results,
            ordered=False,
        )
    for c in [c for c in X.columns.values if c.startswith('bodypart_id')]:
        X[c] = pd.Categorical(
            X[c].replace(spadlcfg.bodyparts_df().bodypart_name.to_dict()),
            categories=spadlcfg.bodyparts,
            ordered=False,
        )
    return X


def get_labels(api, game_ids=None, shotfilter=None):
    game_ids = api.games.index if game_ids is None else game_ids
    y = {}
    for game_id in tqdm(game_ids, desc=f'Generating labels'):
        try:
            game = api.games.loc[game_id]
            game_actions = utils.enhance_actions(api.get_actions(game_id))
            y[game_id] = _compute_labels_game(game, game_actions, shotfilter)
            y[game_id].index.name = 'action_id'
            y[game_id]['game_id'] = game_id
        except Exception as e:
            print(e)
    return (
        pd.concat(y.values())
        .reset_index()
        .set_index(['game_id', 'action_id'])['goal']
    )


def _compute_labels_game(game, actions, shotfilter=None):
    # compute labels
    y = actions['result_name'] == 'success'
    if shotfilter is None:
        # filter shots and ignore own goals
        shot_idx = actions.type_name.isin(
            ['shot', 'shot_penalty', 'shot_freekick']
        ) & actions.result_name.isin(['fail', 'success'])
    else:
        shot_idx = shotfilter(actions)
    return y.loc[shot_idx].to_frame('goal')
