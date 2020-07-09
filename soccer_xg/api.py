"""An API wrapper for the SPADL format."""
import logging
import os
import warnings

import pandas as pd

deduplic = dict(
    games=('game_id', 'game_id'),
    teams=(['team_id'], 'team_id'),
    players=(['player_id'], 'player_id'),
    player_games=(
        ['game_id', 'team_id', 'player_id'],
        ['game_id', 'team_id', 'player_id'],
    ),
    files=('file_url', 'file_url'),
)


class DataApi:
    """An objectect that provides easy access to a SPADL event stream dataset.

    Automatically defines an attribute which lazy loads the contents of
    each table in the HDF files and defines a couple of methods to easily execute
    common queries on the SPADL data.

    Parameters
    ----------
    db_path : A list of strings or a single string
        Path(s) to HDF files containing the data.

    Attributes
    ----------
    ``table_name`` : ``pd.DataFrame``
        A single pandas dataframe that contains all records from all ``table_name``
        tables in each HDF file.
    """

    def __init__(self, db_path):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Loading datasets')
        if type(db_path) is list:
            self.db_path = set(db_path)
        elif type(db_path) is not set:
            self.db_path = set([db_path])

        for p in self.db_path:
            if not os.path.exists(p):
                raise ValueError(
                    'A database `{}` does not exist.'.format(str(p))
                )

    def __getattr__(self, name):
        self.logger.info(f'Loading `{name}` data')
        DB = []
        for p in self.db_path:
            with pd.HDFStore(p, 'r') as store:
                for key in [
                    k for k in store.keys() if (k[1:].rsplit('/')[0] == name)
                ]:
                    db = store[key]
                    db['db_path'] = p
                    DB.append(db)
        if len(DB) == 0:
            raise ValueError('A table `{}` does not exist.'.format(str(name)))
        else:
            DB = pd.concat(DB, sort=False)
            if name in deduplic:
                sortcols, idcols = deduplic[name]
                DB.sort_values(by=sortcols, ascending=False, inplace=True)
                DB.drop_duplicates(subset=idcols, inplace=True)
                DB.set_index(idcols, inplace=True)
            setattr(self, name, DB)
            return DB

    def get_events(self, game_id, only_home=False, only_away=False):
        """Return all events performed in a given game.

        Parameters
        ----------
        game_id : int
            The ID of a game.
        only_home : bool
            Include only events from the home team.
        only_away : bool
            Include only events from the away team.

        Returns
        -------
        pd.DataFrame
            A dataframe with a row for each event, indexed by period_id and
            a timestamp (ms) in which the event happened.

        Raises
        ------
        ValueError
            If both `only_home` and `only_away` are True.
        IndexError
            If no game exists with the provided ID.
        """
        if only_home and only_away:
            raise ValueError('only_home and only_away cannot be both True.')

        try:
            db = self.games.at[game_id, 'db_path']
            with pd.HDFStore(db, 'r') as store:
                df_game_events = store.get(f'events/game_{game_id}')
                home_team_id, away_team_id = self.get_home_away_team_id(
                    game_id
                )
                if only_home:
                    team_filter = df_game_events.team_id == home_team_id
                elif only_away:
                    team_filter = df_game_events.team_id == away_team_id
                else:
                    team_filter = [True] * len(df_game_events)
                return df_game_events.loc[(team_filter), :].set_index(
                    ['period_id', 'period_milliseconds']
                )
        except KeyError:
            raise IndexError(
                'No events found for a game with the provided ID.'
            )

    def get_actions(
        self, game_id, only_home=False, only_away=False, features=False
    ):
        """Return all actions performed in a given game.

        Parameters
        ----------
        game_id : int
            The ID of a game.
        only_home : bool
            Include only actions from the home team.
        only_away : bool
            Include only actions from the away team.

        Returns
        -------
        pd.DataFrame
            A dataframe with a row for each action, indexed by period_id and
            a timestamp (ms) in which the action was executed.

        Raises
        ------
        ValueError
            If both `only_home` and `only_away` are True.
        IndexError
            If no game exists with the provided ID.
        """
        if only_home and only_away:
            raise ValueError('only_home and only_away cannot be both True.')

        try:
            db = self.games.at[game_id, 'db_path']
            with pd.HDFStore(db, 'r') as store:
                df_game_actions = store.get(f'actions/game_{game_id}')
                if features:
                    try:
                        df_game_features = store.get(
                            f'features/game_{game_id}'
                        )
                        df_game_actions = pd.concat(
                            [df_game_actions, df_game_features], axis=1
                        )
                    except KeyError:
                        warnings.warn('Could not find precomputed features')

            home_team_id, away_team_id = self.get_home_away_team_id(game_id)
            if only_home:
                team_filter = df_game_actions.team_id == home_team_id
            elif only_away:
                team_filter = df_game_actions.team_id == away_team_id
            else:
                team_filter = [True] * len(df_game_actions)
            return df_game_actions.loc[(team_filter), :].set_index(
                ['action_id']
            )
        except KeyError:
            raise IndexError(
                'No actions found for a game with the provided ID.'
            )

    # Games ##################################################################

    def get_home_away_team_id(self, game_id):
        """Return the id of the home and away team in a given game.

        Parameters
        ----------
        game_id : int
            The ID of a game.

        Returns
        -------
        (int, int)
            The ID of the home and away team.

        Raises
        ------
        IndexError
            If no game exists with the provided ID.
        """
        try:
            return self.games.loc[
                game_id, ['home_team_id', 'away_team_id']
            ].values
        except KeyError:
            raise IndexError('No game found with the provided ID.')

    # Players ################################################################

    def get_player_name(self, player_id):
        """Return the name of a player with a given ID.

        Parameters
        ----------
        player_id : int
            The ID of a player.

        Returns
        -------
            The name of the player.

        Raises
        ------
        IndexError
            If no player exists with the provided ID.
        """
        try:
            return self.players.at[player_id, 'player_name']
        except KeyError:
            raise IndexError('No player found with the provided ID.')

    def search_player(self, query, limit=10):
        """Search for a player by name.

        Parameters
        ----------
        query : str
            The name of a player.
        limit : int
            Max number of results that are returned.

        Returns
        -------
        pd.DataFrame
            The first `limit` players that game the given query.

        """
        return self.players[
            self.players.player_name.str.contains(query, case=False)
        ].head(limit)

    # Teams ##################################################################

    def get_team_name(self, team_id):
        """Return the name of a team with a given ID.

        Parameters
        ----------
        team_id : int
            The ID of a team.

        Returns
        -------
            The name of the team.

        Raises
        ------
        IndexError
            If no team exists with the provided ID.
        """
        try:
            return self.teams.at[team_id, 'team_name']
        except KeyError:
            raise IndexError('No team found with the provided ID.')

    def search_team(self, query, limit=10):
        """Search for a team by name.

        Parameters
        ----------
        query : str
            The name of a team.
        limit : int
            Max number of results that are returned.

        Returns
        -------
        pd.DataFrame
            The first `limit` teams that game the given query.

        """
        return self.teams[
            self.teams.team_name.str.contains(query, case=False)
        ].head(limit)
