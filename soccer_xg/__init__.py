"""A Python package for training and analyzing expected goals (xG) models in soccer."""

__version__ = '0.0.1'

__all__ = ['DataApi', 'XGModel']

from soccer_xg.api import DataApi
from soccer_xg.xg import XGModel
