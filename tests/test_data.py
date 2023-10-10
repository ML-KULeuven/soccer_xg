"""Tests for the soccer_xg.data module."""
from typing import Generator

import pytest
from pytest import fixture
from sqlalchemy import create_engine
from soccer_xg.data import Dataset, HDFDataset, SQLDataset
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl.statsbomb import convert_to_actions


def test_create_sqlite_dataset() -> None:
    """It should add the requested data to the database."""
    engine = create_engine("sqlite://")
    data_loader = StatsBombLoader(getter="remote")
    with SQLDataset(engine) as db:
        db.import_data(
            data_loader, convert_to_actions, competition_id=55, season_id=43, game_id=3795107
        )
        df_competitions = db.read_query("SELECT * FROM competitions")
        assert len(df_competitions) == 1
        df_games = db.read_query("SELECT * FROM competitions")
        assert len(df_games) == 1
        df_teams = db.read_query("SELECT * FROM teams")
        assert len(df_teams) == 2
        df_players = db.read_query("SELECT * FROM players")
        assert len(df_players) == 30
        df_actions = db.read_query("SELECT * FROM actions")
        assert len(df_actions) > 0


def test_create_hdf_dataset(tmp_path) -> None:
    """It should add the requested data to the database."""
    db_path = tmp_path / "db.h5"
    data_loader = StatsBombLoader(getter="remote")
    with HDFDataset(db_path, mode="w") as db:
        db.import_data(
            data_loader, convert_to_actions, competition_id=55, season_id=43, game_id=3795107
        )
        df_competitions = db["competitions"]
        assert len(df_competitions) == 1
        df_games = db["games"]
        assert len(df_games) == 1
        df_teams = db["teams"]
        assert len(df_teams) == 2
        df_players = db["players"]
        assert len(df_players) == 30
        df_player_games = db["player_games"]
        assert len(df_player_games) == 30
        df_actions = db["actions/game_3795107"]
        assert len(df_actions) > 0


@fixture(scope="session")
def sqlitedb() -> Generator[SQLDataset, None, None]:
    """Create a SQL dataset with BEL v ITA at EURO2020."""
    engine = create_engine("sqlite://")
    data_loader = StatsBombLoader(getter="remote")
    with SQLDataset(engine) as db:
        db.import_data(
            data_loader, convert_to_actions, competition_id=55, season_id=43, game_id=3795107
        )
        yield db


@fixture(scope="session")
def hdfdb() -> Generator[HDFDataset, None, None]:
    """Create a HDF dataset with BEL v ITA at EURO2020."""
    data_loader = StatsBombLoader(getter="remote")
    with HDFDataset(":memory:", mode="w", driver="H5FD_CORE", driver_core_backing_store=0) as db:
        db.import_data(
            data_loader, convert_to_actions, competition_id=55, season_id=43, game_id=3795107
        )
        yield db


database_interfaces = [pytest.lazy_fixture("sqlitedb"), pytest.lazy_fixture("hdfdb")]


@pytest.mark.parametrize("db", database_interfaces)
def test_create_dataset_duplicates(db: Dataset) -> None:
    """The database should not contain duplicate data."""
    data_loader = StatsBombLoader(getter="remote")
    db.import_data(
        data_loader, convert_to_actions, competition_id=55, season_id=43, game_id=3795107
    )
    db.import_data(
        data_loader, convert_to_actions, competition_id=55, season_id=43, game_id=3795107
    )
    df_games = db.games()
    assert len(df_games) == 1


@pytest.mark.parametrize("db", database_interfaces)
def test_games(db: Dataset) -> None:
    """It should return the games for a given competition and season."""
    df_games = db.games()
    assert len(df_games) == 1
    df_games = db.games(competition_id=0)
    assert len(df_games) == 0
    df_games = db.games(season_id=0)
    assert len(df_games) == 0


@pytest.mark.parametrize("db", database_interfaces)
def test_teams(db: Dataset) -> None:
    """It should return the teams in the dataset."""
    df_teams = db.teams()
    assert len(df_teams) == 2


@pytest.mark.parametrize("db", database_interfaces)
def test_players(db: Dataset) -> None:
    """It should return the players for a given game."""
    df_players = db.players()
    assert len(df_players) == 30
    df_players = db.players(game_id=3795107)
    assert len(df_players) == 30
    df_players = db.players(game_id=0)
    assert len(df_players) == 0


@pytest.mark.parametrize("db", database_interfaces)
def test_actions(db: Dataset) -> None:
    """It should return the actions for a given game."""
    df_actions = db.actions(game_id=3795107)
    assert len(df_actions) > 0
    with pytest.raises(IndexError, match="No game found with ID=0"):
        db.actions(game_id=0)


@pytest.mark.parametrize("db", database_interfaces)
def test_get_home_away_team_id(db: Dataset) -> None:
    """It should return the ID of the home and away team in a given game."""
    home_team_id, away_team_id = db.get_home_away_team_id(game_id=3795107)
    assert home_team_id == 782
    assert away_team_id == 914
    with pytest.raises(IndexError, match="No game found with ID=0"):
        db.get_home_away_team_id(game_id=0)


@pytest.mark.parametrize("db", database_interfaces)
def test_get_player_name(db: Dataset) -> None:
    """It should return the name of the player with a given ID."""
    player_name = db.get_player_name(player_id=3089)
    assert player_name == "Kevin De Bruyne"
    with pytest.raises(IndexError, match="No player found with ID=0"):
        db.get_player_name(player_id=0)


@pytest.mark.parametrize("db", database_interfaces)
def test_search_player(db: Dataset) -> None:
    """It should return a list with matching players."""
    players = db.search_player("Kevin")
    assert len(players) == 1
    assert players.at[3089, "player_name"] == "Kevin De Bruyne"


@pytest.mark.parametrize("db", database_interfaces)
def test_get_team_name(db: Dataset) -> None:
    """It should return the name of the team with a given ID."""
    team_name = db.get_team_name(team_id=782)
    assert team_name == "Belgium"
    with pytest.raises(IndexError, match="No team found with ID=0"):
        db.get_team_name(team_id=0)


@pytest.mark.parametrize("db", database_interfaces)
def test_search_team(db: Dataset) -> None:
    """It should return a list with matching teams."""
    teams = db.search_team("Belg")
    assert len(teams) == 1
    assert teams.at[782, "team_name"] == "Belgium"
