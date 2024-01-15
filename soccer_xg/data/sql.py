"""SQLite database interface."""
from typing import Optional, Tuple

import pandas as pd
import sqlalchemy

from .base import Dataset


class SQLDataset(Dataset, pd.io.sql.SQLDatabase):
    """Wrapper for a SQL database holding the raw data.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.
    """

    def _import_competitions(self, competitions: pd.DataFrame) -> None:
        self.to_sql(competitions, "competitions", if_exists="append", index=False)
        self._drop_duplicates("competitions", ["competition_id"])

    def _import_games(self, games: pd.DataFrame) -> None:
        self.to_sql(games, "games", if_exists="append", index=False)
        self._drop_duplicates("games", ["game_id"])

    def _import_teams(self, teams: pd.DataFrame) -> None:
        self.to_sql(teams, "teams", if_exists="append", index=False)
        self._drop_duplicates("teams", ["team_id"])

    def _import_players(self, players: pd.DataFrame) -> None:
        self.to_sql(players, "player_games", if_exists="append", index=False)
        self._drop_duplicates("player_games", ["player_id", "game_id"])

    def _import_events(self, events: pd.DataFrame) -> None:
        self.to_sql(
            events,
            "events",
            if_exists="append",
            index=False,
            dtype={
                "extra": sqlalchemy.types.JSON,
                "related_events": sqlalchemy.types.JSON,
                "location": sqlalchemy.types.JSON,
            },
        )
        self._drop_duplicates("events", ["game_id", "event_id"])

    def _import_actions(self, actions: pd.DataFrame) -> None:
        self.to_sql(actions, "actions", if_exists="append", index=False)
        self._drop_duplicates("actions", ["game_id", "action_id"])

    def _drop_duplicates(self, table: str, keys: list) -> None:
        """Drop duplicate rows from a table.

        Parameters
        ----------
        table : str
            The name of the table.
        keys : list
            The columns to use to determine duplicates.
        """
        query = f"""
            DELETE FROM {table}
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM {table}
                GROUP BY {", ".join(keys)}
            )
        """
        self.execute(query)

    def games(
        self, competition_id: Optional[int] = None, season_id: Optional[int] = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM games"
        filters = []
        if competition_id is not None:
            filters.append(f"competition_id = {competition_id}")
        if season_id is not None:
            filters.append(f"season_id = {season_id}")
        if len(filters):
            query += " WHERE " + " AND ".join(filters)
        return self.read_query(query, index_col="game_id")

    def teams(self) -> pd.DataFrame:
        return self.read_query("SELECT * FROM teams", index_col="team_id")

    def players(self, game_id: Optional[int] = None) -> pd.DataFrame:
        if game_id is not None:
            result = self.read_query(
                f"SELECT * FROM player_games WHERE game_id = {game_id}", index_col="player_id"
            )
            if result.empty:
                raise IndexError(f"No game found with ID={game_id}")
            return result
        else:
            return self.read_query(
                "SELECT DISTINCT player_id, player_name, nickname FROM player_games",
                index_col="player_id",
            )

    def events(self, game_id: int) -> pd.DataFrame:
        query = f"SELECT * FROM events WHERE game_id = {game_id}"
        df_events = self.read_query(query, index_col=["event_id"])
        if df_events.empty:
            raise IndexError(f"No game found with ID={game_id}")
        return df_events

    def actions(self, game_id: int) -> pd.DataFrame:
        query = f"SELECT * FROM actions WHERE game_id = {game_id}"
        df_actions = self.read_query(query, index_col=["action_id"])
        if df_actions.empty:
            raise IndexError(f"No game found with ID={game_id}")
        return df_actions

    def get_home_away_team_id(self, game_id: int) -> Tuple[int, int]:
        query = f"""
            SELECT home_team_id, away_team_id
            FROM games
            WHERE game_id = {game_id}
        """
        try:
            home_team_id, away_team_id = self.read_query(query).loc[0]
            return home_team_id, away_team_id
        except KeyError:
            raise IndexError(f"No game found with ID={game_id}")
