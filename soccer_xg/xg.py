"""Tools for creating and analyzing xG models."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Literal

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import NotFittedError
from tqdm.auto import tqdm

from soccer_xg import attributes as fs
from soccer_xg import metrics
from soccer_xg.data.base import Dataset
from soccer_xg.ml.preprocessing import simple_proc_for_linear_algoritms
from soccer_xg.ml.pipeline import InteractionFeature


class XGModel:
    """A wrapper around a pipeline for computing xG values.

    Parameters
    ----------
    dataset_transformer : DatasetTransformer
        A dataset transformer to convert a dataset to features.
    pipeline : Pipeline
        Scikit-Learn pipeline to use for the model.
    column_descriptions : dictionary
        A dictionary whose keys are the names of the columns used in the
        model, and the values are string descriptions of what the columns
        mean.

    Attributes
    ----------
    training_seasons : A list of tuples, or ``None`` (default=``None``)
        If the model was trained using data from a Dataset, a list of
        (competition_id, season_id) tuples used to train the model. If no
        Dataset was used, an empty list. If no model has been trained yet,
        ``None``.
    validation_seasons :  A list of tuples, or ``None`` (default=``None``)
        Same as ``training_seasons``, but for validation data.
    sample_probabilities : A numpy array of floats or ``None`` (default=``None``)
        After the model has been validated, contains the sampled predicted
        probabilities used to compute the validation statistic.
    predicted_goal_percents : A numpy array of floats or ``None`` (default=``None``)
        After the model has been validated, contains the actual probabilities
        in the test set at each probability in ``sample_probabilities``.
    num_shots_used : A numpy array of floats or ``None`` (default=``None``)
        After the model has been validated, contains the number of shots used
        to compute each element of ``predicted_goal_percents``.
    model_directory : string
        The directory where all models will be saved to or loaded from.
    """

    model_directory = Path(__file__).resolve().parent / "models"
    _default_model_filename = "default_model.xg"

    def __init__(
        self,
        dataset_transformer: DatasetTransformer,
        pipeline: Pipeline,
        column_descriptions: dict[str, str] | None = None,
    ):
        self.dataset_transformer = dataset_transformer
        self.pipeline = pipeline
        self._fitted = False
        self._training_seasons = None
        self._validation_seasons = None

        self._sample_probabilities = None
        self._predicted_goal_percents = None
        self._num_shots_used = None

    @property
    def training_seasons(self) -> list[tuple[str, str]] | None:
        return self._training_seasons

    @property
    def validation_seasons(self) -> list[tuple[str, str]] | None:
        return self._validation_seasons

    @property
    def sample_probabilities(self) -> list[float] | None:
        return self._sample_probabilities

    @property
    def predicted_goal_percents(self) -> list[float] | None:
        return self._predicted_goal_percents

    @property
    def num_shots_used(self) -> int | None:
        return self._num_shots_used

    @classmethod
    def filter_shots(cls, df_actions) -> pd.Series:
        """Return a boolean mask indicating which shots to handle.

        This method is used to filter out shots that should not be used
        for training, validation, or prediction. By default, it filters
        out own-goals only.

        Parameters
        ----------
        df_actions : pd.DataFrame
            A dataframe containing the SPADL actions.

        Returns
        -------
        pd.Series
            A boolean mask indicating which shots to handle.
        """
        shot_mask = df_actions.type_name.isin(
            ["shot", "shot_penalty", "shot_freekick"]
        ) & df_actions.result_name.isin(["fail", "success"])
        return shot_mask

    def train(
        self,
        source_data: Dataset | pd.DataFrame,
        target_colname: str = "goal",
        training_seasons: list[tuple[str, str]] | None = None,
    ):
        """Train the model.

        Once a modeling pipeline is set up (either the default or something
        custom-generated), historical data needs to be fed into it in order to
        "fit" the model so that it can then be used to predict future results.
        This method implements a simple wrapper around the core Scikit-learn
        functionality which does this.

        The default is to use data from a Dataset object, however that can be
        changed to a simple Pandas DataFrame with precomputed features and
        labels if desired.

        There is no particular output from this function, rather the
        parameters governing the fit of the model are saved inside the model
        object itself. If you want to get an estimate of the quality of the
        fit, use the ``validate_model`` method after running this method.

        Parameters
        ----------
        source_data : ``Dataset`` or a Pandas DataFrame
            The data to be used to train the model. If an instance of
            ``Dataset`` is given, will query the database for the training data.
        target_colname : string or integer (default=``"goal"``)
            The name of the target variable column. This is only relevant if
            ``source_data`` is not a ``Dataset``.
        training_seasons : list of tuples (default=``[('ENG', '1617'), ('ENG', '1718')]``)
            What seasons to use to train the model if getting data from a Dataset instance.
            If ``source_data`` is not a ``Dataset``, this argument will be ignored.
            **NOTE:** it is critical not to use all possible data in order to train the
            model - some will need to be reserved for a final validation (see the
            ``validate_model`` method). A good dataset to reserve
            for validation is the most recent one or two seasons.

        Returns
        -------
        ``None``
        """
        self._training_seasons = []
        if isinstance(source_data, Dataset):
            game_ids = pd.concat(
                source_data.games(competition_id=s[0], season_id=s[1]) for s in training_seasons
            ).index.tolist()
            feature_cols, target_col = self.dataset_transformer.transform(
                dataset=source_data,
                game_ids=game_ids,
            )
            self._training_seasons = training_seasons
        else:
            target_col = source_data[target_colname]
            feature_cols = source_data.drop(target_colname, axis=1)
        self.pipeline.fit(feature_cols, target_col.squeeze())
        self._fitted = True

    def validate(
        self,
        source_data: Dataset | pd.DataFrame,
        target_colname: str = "goal",
        validation_seasons: list[tuple[str, str]] | None = None,
        n_bins=10,
        bin_strategy="quantile",
        plot: bool = True,
    ):
        """Validate the model.

        Once a modeling pipeline is trained, a different dataset must be fed
        into the trained model to validate the quality of the fit. This method
        implements a simple wrapper around the core Scikit-learn functionality
        which does this.

        The default is to use data from a Dataset object, however that can be
        changed to a simple Pandas DataFrame with precomputed features and
        labels if desired.

        The output of this method is a dictionary with relevant error metrics
        (see ``soccer_xg.metrics``).

        Parameters
        ----------
        source_data : ``Dataset`` or a Pandas DataFrame
            The data to be used to validate the model. If an instance of
            ``Dataset`` is given, will query the api database for the training
            data.
        validation_seasons : list of tuples (default=``[('ENG', '1819')]``)
            What seasons to use to validated the model if getting data from
            a Dataset instance. If ``source_data`` is not a ``Dataset``, this
            argument will be ignored. **NOTE:** it is critical not to use the
            same data to validate the model as was used in the fit. Generally
            a good data set to use for validation is one from a time period
            more recent than was used to train the model.
        target_colname : string or integer (default=``"goal"``)
            The name of the target variable column. This is only relevant if
            ``source_data`` is not a ``Dataset``.
        n_bins : int, default=10
            Number of bins to discretize the [0, 1] interval. A bigger number
            requires more data.
        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.
            uniform
                The bins have identical widths.
            quantile
                The bins have the same number of samples and depend on `y_prob`.
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
            raise NotFittedError("Must fit model before validating.")

        if isinstance(source_data, Dataset):
            game_ids = pd.concat(
                source_data.games(competition_id=s[0], season_id=s[1]) for s in validation_seasons
            ).index.tolist()
            feature_cols, target_col = self.dataset_transformer.transform(
                dataset=source_data, game_ids=game_ids
            )
            self._validation_seasons = validation_seasons
        else:
            game_ids = None
            feature_cols = source_data.drop(target_colname, axis=1)
            target_col = source_data[target_colname]
            self._validation_seasons = []

        predicted_probabilities = self.estimate(feature_cols)["xG"]
        target_col = target_col.squeeze()

        (
            self._predicted_goal_percents,
            self._sample_probabilities,
            _,
            self._num_shots_used,
        ) = metrics.calibration_curve(target_col, predicted_probabilities, n_bins, bin_strategy)

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
            target_col, predicted_probabilities, n_bins, "uniform"
        )
        ace = metrics.expected_calibration_error(
            target_col, predicted_probabilities, n_bins, "quantile"
        )

        if plot:
            import matplotlib.pyplot as plt

            from soccer_xg.visualisation import (
                plot_reliability_diagram,
                plot_roc_curve,
            )

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_roc_curve(target_col, predicted_probabilities, fig=fig, ax=ax[0])
            plot_reliability_diagram(
                target_col,
                predicted_probabilities,
                fig=fig,
                ax=ax[1],
                bayesian=False,
                bins=n_bins,
                bin_strategy=bin_strategy,
                show_counts=False,
                fmt='s-',
                min_samples=100,
                show_histogram=False,
                overlay_histogram=False,
                invert_histogram=False,
                ci=0.95,
                shaded_ci=True,
                show_gaps=False,
                show_bars=False,
            )
            ax[0].set_title("ROC curve")
            ax[1].set_title("Reliability diagram")
            plt.tight_layout()
            plt.show()

        return {
            "max_dev": max_deviation,
            "residual_area": residual_area,
            "roc": roc,
            "brier": brier,
            "ece": ece,
            "ace": ace,
        }

    def estimate(self, source_data: Dataset | pd.DataFrame, game_ids: list[int] | None = None):
        """Estimate the xG values for all shots in a set of games.

        The default is to use data from a Dataset object, however that can be changed
        to a simple Pandas DataFrame with precomputed features and labels if desired.

        Parameters
        ----------
        source_data : ``Dataset`` or a Pandas DataFrame
            The data to be used to validate the model. If an instance of
            ``Dataset`` is given, will query the api database for the training data.
        game_ids : list of ints (default=None)
            Only xG values for the games in this list are returned. By default,
            xG values are computed for all games in the source data.
            If ``source_data`` is not a ``Dataset``, this argument will be ignored.

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
            raise NotFittedError("Must fit model before predicting WP.")

        if isinstance(source_data, Dataset):
            game_ids = source_data.games().index.tolist() if game_ids is None else game_ids
            source_data, _ = self.dataset_transformer.transform(
                dataset=source_data, game_ids=game_ids
            )

        xg = pd.DataFrame(index=source_data.index)
        xg["xG"] = self.pipeline.predict_proba(source_data)[:, 1]
        return xg

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
        joblib.dump(self, self.model_directory / filename)

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

        return joblib.load(cls.model_directory / filename)


class PenaltyXGModel(XGModel):
    _default_model_filename = "default_penalty_model.xg"

    def __init__(self):
        self._fitted = True
        self.dataset_transformer = DatasetTransformer(
            xfns=[], shotfilter=lambda x: x.type_name == "shot_penalty"
        )
        self.pipeline = Pipeline([])

    def train(
        self,
        source_data: Dataset | pd.DataFrame,
        target_colname: str = "goal",
        training_seasons: list[tuple[str, str]] | None = None,
    ):
        pass

    def estimate(self, source_data: Dataset | pd.DataFrame, game_ids: list[int] | None = None):
        if isinstance(source_data, Dataset):
            game_ids = source_data.games().index.tolist() if game_ids is None else game_ids
            source_data, _ = self.dataset_transformer.transform(
                dataset=source_data, game_ids=game_ids
            )

        xg = pd.DataFrame(index=source_data.index)
        xg["xG"] = 0.792453

        return xg


class FreekickXGModel(XGModel):
    _default_model_filename = "default_freekick_model.xg"

    def __init__(self):
        self.dataset_transformer = DatasetTransformer(
            xfns=[fs.shot_dist, fs.shot_visible_angle],
            shotfilter=lambda x: x.type_name == "shot_freekick",
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        dist_colname = "dist_shot"
        angle_colname = "visible_angle_shot"
        dist_x_angle_colname = "dist_x_visible_angle_shot"

        self.column_descriptions = {
            dist_colname: "Distance to goal",
            angle_colname: "Angle to goal",
            dist_x_angle_colname: "Distance * angle to goal",
        }

        feature_pipeline = InteractionFeature([dist_colname, angle_colname], dist_x_angle_colname)
        preprocess_pipeline = simple_proc_for_linear_algoritms(
            numeric_features=[dist_colname, angle_colname, dist_x_angle_colname],
            categoric_features=[],
        )
        base_model = LogisticRegression(max_iter=10000, solver="lbfgs", fit_intercept=True)
        return make_pipeline(feature_pipeline, preprocess_pipeline, base_model)


class BasicOpenplayXGModel(XGModel):
    _default_model_filename = "default_openplay_model.xg"

    def __init__(self):
        self.dataset_transformer = DatasetTransformer(
            xfns=[fs.shot_dist, fs.shot_visible_angle, fs.shot_bodypart],
            shotfilter=lambda x: x.type_name == 'shot' and x.result_name in ["fail", "success"],
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        bodypart_colname = "bodypart_name_shot"
        dist_colname = "dist_shot"
        angle_colname = "visible_angle_shot"
        dist_x_angle_colname = "dist_x_visible_angle_shot"

        self.column_descriptions = {
            bodypart_colname: "Bodypart used for the shot (head, foot or other)",
            dist_colname: "Distance to goal",
            angle_colname: "Angle to goal",
            dist_x_angle_colname: "Distance * angle to goal",
        }

        feature_pipeline = InteractionFeature([dist_colname, angle_colname], dist_x_angle_colname)
        preprocess_pipeline = simple_proc_for_linear_algoritms(
            numeric_features=[dist_colname, angle_colname, dist_x_angle_colname],
            categoric_features=[bodypart_colname],
        )
        base_model = LogisticRegression(max_iter=10000, solver="lbfgs", fit_intercept=False)
        return make_pipeline(feature_pipeline, preprocess_pipeline, base_model)


class AdvancedOpenplayXGModel(XGModel):
    _default_model_filename = "default_openplay_model.xg"

    def __init__(self):
        self.dataset_transformer = DatasetTransformer(
            xfns=[fs.shot_dist, fs.shot_visible_angle, fs.shot_bodypart],
            shotfilter=lambda x: x.type_name == 'shot' and x.result_name in ["fail", "success"],
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        bodypart_colname = "bodypart_name_shot"
        dist_colname = "dist_shot"
        angle_colname = "visible_angle_shot"
        dist_x_angle_colname = "dist_x_visible_angle_shot"

        self.column_descriptions = {
            bodypart_colname: "Bodypart used for the shot (head, foot or other)",
            dist_colname: "Distance to goal",
            angle_colname: "Angle to goal",
            dist_x_angle_colname: "Distance * angle to goal",
        }

        preprocess_pipeline = simple_proc_for_linear_algoritms(
            numeric_features=[dist_colname, angle_colname, dist_x_angle_colname],
            categoric_features=[bodypart_colname],
        )
        base_model = LogisticRegression(max_iter=10000, solver="lbfgs", fit_intercept=False)
        return make_pipeline(preprocess_pipeline, base_model)


class StatsBombOpenplayXGModel(XGModel):
    _default_model_filename = "default_openplay_model.xg"

    def __init__(self):
        self.dataset_transformer = DatasetTransformer(
            xfns=[fs.shot_dist, fs.shot_visible_angle, fs.shot_bodypart],
            shotfilter=lambda x: x.type_name == 'shot' and x.result_name in ["fail", "success"],
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        bodypart_colname = "bodypart_name_shot"
        dist_colname = "dist_shot"
        angle_colname = "visible_angle_shot"
        dist_x_angle_colname = "dist_x_visible_angle_shot"

        self.column_descriptions = {
            bodypart_colname: "Bodypart used for the shot (head, foot or other)",
            dist_colname: "Distance to goal",
            angle_colname: "Angle to goal",
            dist_x_angle_colname: "Distance * angle to goal",
        }

        feature_pipeline = make_pipeline(
            InteractionFeature([dist_colname, angle_colname], dist_x_angle_colname)
        )
        preprocess_pipeline = simple_proc_for_linear_algoritms(
            numeric_features=[dist_colname, angle_colname, dist_x_angle_colname],
            categoric_features=[bodypart_colname],
        )
        base_model = LogisticRegression(max_iter=10000, solver="lbfgs", fit_intercept=False)
        return make_pipeline(feature_pipeline, preprocess_pipeline, base_model)


class XGModelEnsemble:
    def __init__(self, models=None):
        super().__init__()
        if models is None:
            models = [BasicOpenplayXGModel(), FreekickXGModel(), PenaltyXGModel()]
        self.model = models
        self.column_descriptions = {m.__class__.__name__: m.column_descriptions for m in models}

    def train(self, source_data, training_seasons, target_colname="goal"):
        for model in self.model:
            model.train(source_data, training_seasons, target_colname)

    def validate(self, source_data, validation_seasons, target_colname="goal", plot=True):
        results = {}
        for model in self.model:
            results.update(model.validate(source_data, validation_seasons, target_colname, plot))
        return results

    def estimate(self, source_data, game_ids=None):
        xg = []
        for model in self.model:
            xg.append(model.estimate(source_data, game_ids))
        return pd.concat(xg).sort_index()

    def save_model(self, filename=None):
        if filename is None:
            filename = self._default_model_filename
        for i, model in enumerate(self.model):
            model.save_model(filename=f"{filename}_{i}")

    @classmethod
    def load_model(cls, filename=None):
        if filename is None:
            filename = cls._default_model_filename

        models = []
        i = 0
        while True:
            try:
                models.append(joblib.load(cls.model_directory / f"{filename}_{i}"))
                i += 1
            except FileNotFoundError:
                break
        return cls(models)


def is_shot(action) -> bool:
    """Return a boolean mask indicating which shots to handle.

    This method is used to filter out shots that should not be used
    for training, validation, or prediction. By default, it filters
    out own-goals only.

    Parameters
    ----------
    df_actions : pd.Series
        A dataframe containing the SPADL actions.

    Returns
    -------
    pd.Series
        A boolean mask indicating which shots to handle.
    """
    return (action.type_name in ["shot", "shot_penalty", "shot_freekick"]) and (
        action.result_name in ["fail", "success"]
    )


class DatasetTransformer:
    """Transforms a dataset to xG features and labels.

    Parameters
    ----------
    xfns : list(callable)
        List of feature generators to apply. Defaults to ``default_features``.
    yfns : list(callable)
        List of label generators to apply. Defaults to ``default_labels``.
    shotfilter: callable(pd.Series) -> bool
        A function that takes a SPADL action and returns True if the
        action should be used for feature extraction. If None, all shots will
        be used (excluding own-goals).
    nb_prev_actions: int
        The number of previous actions to consider when calculating labels
    """

    def __init__(
        self,
        xfns: list[fs.AttributeGenerator | str] = fs.default_features,
        yfns: list[fs.AttributeGenerator | str] = fs.default_labels,
        shotfilter: Callable[[pd.Series], bool] = is_shot,
        nb_prev_actions: int = 3,
    ):
        self.xfns = xfns
        self.yfns = yfns
        self.shotfilter = shotfilter
        self.nb_prev_actions = nb_prev_actions

    def transform(
        self,
        dataset: Dataset,
        game_ids: list[int] | None = None,
        on_fail: Literal["raise", "warn"] = "raise",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare a dataset for training and validation.

        Parameters
        ----------
        dataset : Dataset
            The dataset to use.
        game_ids : list of ints (default=None)
            Only use data from the games in this list. By default, all games
            in the dataset are used.
        on_fail: 'raise' or 'warn'
            What to do if a feature or label function fails on a specific game.

        Returns
        -------
        X : pd.DataFrame
            A dataframe containing the features.
        y : pd.DataFrame
            A dataframe containing the labels.
        """
        games = dataset.games()
        game_ids = games.index.tolist() if game_ids is None else game_ids
        X, y = {}, {}
        for game_id in tqdm(game_ids, desc="Preparing dataset"):
            try:
                game = games.loc[game_id]
                game_actions = dataset.actions(game_id)
                game_events = dataset.events(game_id)
                _X, _y = fs.compute_attributes(
                    game,
                    game_actions,
                    events=game_events,
                    xfns=self.xfns,
                    yfns=self.yfns,
                    shotfilter=self.shotfilter,
                    nb_prev_actions=self.nb_prev_actions,
                )
                _X["game_id"] = game_id
                _y["game_id"] = game_id
                if len(_X) and len(_y):
                    X[game_id] = _X
                    y[game_id] = _y
            except Exception as e:
                if on_fail == "warn":
                    warnings.warn(f"Failed for game with id={game_id}: {e}", stacklevel=2)
                else:
                    raise RuntimeError(f"Failed for game with id={game_id}.") from e
        X = pd.concat(X.values()).reset_index().set_index(["game_id", "action_id"])
        # remove post-shot features (these will all have a single unique value)
        # f = X.columns[X.nunique() > 1]
        f = X.columns
        y = pd.concat(y.values()).reset_index().set_index(["game_id", "action_id"])
        return X[f], y
