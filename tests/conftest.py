import pytest
import pandas as pd

from typing import Generator
import soccer_xg.xg as xg
from soccer_xg.data import HDFDataset
from soccer_xg.utils import enhance_actions


@pytest.fixture(scope='session')
def dataset() -> Generator[HDFDataset, None, None]:
    with HDFDataset('tests/data/spadl-statsbomb-WC-2018.h5') as db:
        yield db


@pytest.fixture()
def shot() -> pd.DataFrame:
    return enhance_actions(pd.DataFrame(
        [
            {
                "game_id": 8658,
                "original_event_id": "a8692197-bb35-453d-9191-fe7daa25f8df",
                "period_id": 2,
                "time_seconds": 1176.0,
                "team_id": 771,
                "player_id": 3009.0,
                "start_x": 105 - 11,
                "start_y": 34,
                "end_x": 105.0,
                "end_y": 37.01265822784811,
                "type_id": 11,
                "result_id": 1,
                "bodypart_id": 0,
            }
        ]
    ))


@pytest.fixture()
def model() -> xg.XGModel:
    return xg.XGModel()
