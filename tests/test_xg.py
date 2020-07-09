import collections
import os

from soccer_xg import xg


class TestDefaults(object):
    """Tests for defaults."""

    def test_column_descriptions_set(self, model):
        assert isinstance(model.column_descriptions, collections.Mapping)


class TestModelTrain(object):
    """Tests for the train_model method."""

    def test_api_input(self, model, api):
        model.train(source_data=api, training_seasons=[('WC', '2018')])

    def test_dataframe_input(self, model, api):
        features = xg.get_features(api)
        labels = xg.get_labels(api)
        df = features.assign(goal=labels)
        model.train(source_data=df)


class TestModelValidate(object):
    """Tests for the validate_model method."""

    def test_api_input(self, model, api):
        model.train(source_data=api, training_seasons=[('WC', '2018')])
        model.validate(
            source_data=api, validation_seasons=[('WC', '2018')], plot=False
        )

    def test_dataframe_input(self, model, api):
        features = xg.get_features(api)
        labels = xg.get_labels(api)
        df = features.assign(goal=labels)
        model.train(source_data=df)
        model.validate(source_data=df, plot=False)


class TestModelIO(object):
    """Tests functions that deal with model saving and loading"""

    def teardown_method(self, method):
        try:
            os.remove(self.expected_path)
        except OSError:
            pass

    def test_model_save_default(self, model):
        model_name = 'test_hazard.xgmodel'
        model._default_model_filename = model_name

        self.expected_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'soccer_xg',
            'models',
            model_name,
        )
        assert os.path.isfile(self.expected_path) is False

        model.save_model()
        assert os.path.isfile(self.expected_path) is True

    def test_model_save_specified(self, model):
        model = xg.XGModel()
        model_name = 'test_lukaku.xgmodel'

        self.expected_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'soccer_xg',
            'models',
            model_name,
        )
        assert os.path.isfile(self.expected_path) is False

        model.save_model(filename=model_name)
        assert os.path.isfile(self.expected_path) is True

    def test_model_load_default(self, model):
        model_name = 'test_witsel.xgmodel'
        model._default_model_filename = model_name

        self.expected_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'soccer_xg',
            'models',
            model_name,
        )
        assert os.path.isfile(self.expected_path) is False

        model.save_model()

        xGModel_class = xg.XGModel
        xGModel_class._default_model_filename = model_name

        loaded_model = xGModel_class.load_model()

        assert isinstance(loaded_model, xg.XGModel)

    def test_model_load_specified(self):
        model = xg.XGModel()
        model_name = 'test_kompany.xgmodel'

        self.expected_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'soccer_xg',
            'models',
            model_name,
        )
        assert os.path.isfile(self.expected_path) is False

        model.save_model(filename=model_name)

        loaded_model = xg.XGModel.load_model(filename=model_name)
        assert isinstance(loaded_model, xg.XGModel)


def test_get_features(api):
    features = xg.get_features(api, game_ids=[7584])
    assert len(features) == 40  # one row for each shot


def test_get_labels(api):
    labels = xg.get_labels(api, game_ids=[7584])
    assert len(labels) == 40  # one row for each shot
    assert labels.sum() == 5
