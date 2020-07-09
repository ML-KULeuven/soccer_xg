import math
import os
import sys

import pandas as pd
import socceraction.spadl as spadl
import socceraction.spadl.statsbomb as statsbomb
from tqdm import tqdm

seasons = {
    3: '2018',
}
leagues = {
    'FIFA World Cup': 'WC',
    }

free_open_data_remote = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/'
spadl_datafolder = 'tests/data'

SBL = statsbomb.StatsBombLoader(root=free_open_data_remote, getter='remote')

# View all available competitions
df_competitions = SBL.competitions()
df_selected_competitions = df_competitions[df_competitions.competition_name.isin(
    leagues.keys()
)]

for competition in df_selected_competitions.itertuples():
    # Get matches from all selected competition
    matches = SBL.matches(competition.competition_id, competition.season_id)

    matches_verbose = tqdm(list(matches.itertuples()), desc='Loading match data')
    teams, players, player_games = [], [], []

    competition_id = leagues[competition.competition_name]
    season_id = seasons[competition.season_id]
    spadl_h5 = os.path.join(spadl_datafolder, f'spadl-statsbomb-{competition_id}-{season_id}.h5')
    with pd.HDFStore(spadl_h5) as spadlstore:

        spadlstore.put('actiontypes', spadl.actiontypes_df(), format='table')
        spadlstore.put('results', spadl.results_df(), format='table')
        spadlstore.put('bodyparts', spadl.bodyparts_df(), format='table')

        for match in matches_verbose:
            # load data
            teams.append(SBL.teams(match.match_id))
            players.append(SBL.players(match.match_id))
            events = SBL.events(match.match_id)

            # convert data
            player_games.append(statsbomb.extract_player_games(events))
            spadlstore.put(f'actions/game_{match.match_id}', statsbomb.convert_to_actions(events,match.home_team_id), format='table')

        games = matches.rename(columns={'match_id': 'game_id', 'match_date': 'game_date'})
        games.season_id = season_id
        games.competition_id = competition_id
        spadlstore.put('games', games)
        spadlstore.put('teams', pd.concat(teams).drop_duplicates('team_id').reset_index(drop=True))
        spadlstore.put('players', pd.concat(players).drop_duplicates('player_id').reset_index(drop=True))
        spadlstore.put('player_games', pd.concat(player_games).reset_index(drop=True))
