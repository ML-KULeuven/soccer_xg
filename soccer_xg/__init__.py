"""
A Python package for training and analyzing expected goals (xG) models in soccer.

:copyright: (c) 2023 by DTAI KU Leuven.
:license: Apache v2, see LICENSE for more details.
"""
from soccer_xg.data import Dataset, HDFDataset, SQLDataset
from soccer_xg.xg import (
    DatasetTransformer,
    XGModel,
    PenaltyXGModel,
    FreekickXGModel,
    BasicOpenplayXGModel,
    AdvancedOpenplayXGModel,
    StatsBombOpenplayXGModel,
)

__version__ = '0.0.1'
__all__ = [
    'Dataset',
    'HDFDataset',
    'SQLDataset',
    'DatasetTransformer',
    'XGModel',
    'PenaltyXGModel',
    'FreekickXGModel',
    'BasicOpenplayXGModel',
    'AdvancedOpenplayXGModel',
    'StatsBombOpenplayXGModel',
]
