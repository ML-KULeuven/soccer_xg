import pytest
import soccer_xg.xg as xg
from soccer_xg.api import DataApi


@pytest.fixture(scope='session')
def api():
    return DataApi('tests/data/spadl-statsbomb-WC-2018.h5')


@pytest.fixture()
def model():
    return xg.XGModel()
