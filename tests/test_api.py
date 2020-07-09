import pytest

from tests.fixtures import api


def test_get_actions(api):
    df_actions = api.get_actions(7584)
    assert len(df_actions)


def test_get_home_away_team_id(api):
    home_id, away_id = api.get_home_away_team_id(7584)
    assert home_id == 782
    assert away_id == 778


def test_get_team_name(api):
    name = api.get_team_name(782)
    assert name == 'Belgium'


def test_get_player_name(api):
    name = api.get_player_name(3089)
    assert name == 'Kevin De Bruyne'
