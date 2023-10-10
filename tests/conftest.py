import pytest

from typing import Generator
import soccer_xg.xg as xg
from soccer_xg.data import HDFDataset


@pytest.fixture(scope='session')
def dataset() -> Generator[HDFDataset, None, None]:
    with HDFDataset('tests/data/spadl-statsbomb-WC-2018.h5') as db:
        yield db


@pytest.fixture()
def model():
    return xg.XGModel()
