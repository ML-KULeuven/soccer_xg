import math

import pandas as pd
import socceraction.spadl.config as spadlcfg
from fuzzywuzzy import fuzz


def play_left_to_right(actions, home_team_id):
    away_idx = actions.team_id != home_team_id
    for col in ['start_x', 'end_x']:
        actions.loc[away_idx, col] = (
            spadlcfg.field_length - actions.loc[away_idx][col].values
        )
    for col in ['start_y', 'end_y']:
        actions.loc[away_idx, col] = (
            spadlcfg.field_width - actions.loc[away_idx][col].values
        )
    return actions


def enhance_actions(actions):
    # data
    actiontypes = pd.DataFrame(
        list(enumerate(spadlcfg.actiontypes)), columns=['type_id', 'type_name']
    )

    bodyparts = pd.DataFrame(
        list(enumerate(spadlcfg.bodyparts)),
        columns=['bodypart_id', 'bodypart_name'],
    )

    results = pd.DataFrame(
        list(enumerate(spadlcfg.results)), columns=['result_id', 'result_name']
    )

    return (
        actions.merge(actiontypes, how='left')
        .merge(results, how='left')
        .merge(bodyparts, how='left')
        # .sort_values(["period_id", "time_seconds", "timestamp"])
    )


def match_name(name, list_names, min_score=0):
    # -1 score incase we don't get any matches
    max_score = -1
    # Returning empty name for no match as well
    max_name = ''
    # Iternating over all names in the other
    for name2 in list_names:
        # Finding fuzzy match score
        score = fuzz.ratio(name, name2)
        # Checking if we are above our threshold and have a better score
        if (score > min_score) & (score > max_score):
            max_name = name2
            max_score = score
    return (max_name, max_score)


def map_names(
    df1,
    df1_match_colname,
    df1_output_colname,
    df2,
    df2_match_colname,
    df2_output_colname,
    threshold=75,
):
    # List for dicts for easy dataframe creation
    dict_list = []
    for _, (id, name) in df1[
        [df1_output_colname, df1_match_colname]
    ].iterrows():
        # Use our method to find best match, we can set a threshold here
        match = match_name(name, df2[df2_match_colname], threshold)
        # New dict for storing data
        dict_ = {}
        dict_.update({'df1_name': name})
        dict_.update({'df1_id': id})
        if match[1] > threshold:
            dict_.update({'df2_name': match[0]})
            dict_.update(
                {
                    'df2_id': df2.loc[
                        df2[df2_match_colname] == match[0], df2_output_colname
                    ].iloc[0]
                }
            )
        else:
            dict_.update({'df2_name': 'unknown'})
            dict_.update({'df2_id': 0})
        dict_list.append(dict_)
    merge_table = pd.DataFrame(dict_list)
    return merge_table


def get_matching_game(api, game_id, provider, other_provider, teams_mapping):
    season_id = str(api[provider].games.loc[game_id, 'season_id'])
    competition_id = api[provider].games.loc[game_id, 'competition_id']
    # Get matching game
    home_team, away_team = api[provider].get_home_away_team_id(game_id)
    other_home_team = teams_mapping.set_index(f'{provider}_id').loc[
        home_team, f'{other_provider}_id'
    ]
    other_away_team = teams_mapping.set_index(f'{provider}_id').loc[
        away_team, f'{other_provider}_id'
    ]
    other_games = api[other_provider].games
    other_game_id = (
        other_games[
            (other_games.home_team_id == other_home_team)
            & (other_games.away_team_id == other_away_team)
            & (other_games.competition_id == competition_id)
            & (other_games.season_id.astype(str) == season_id)
        ]
        .iloc[0]
        .name
    )
    return other_game_id


def get_matching_shot(
    api,
    shot,
    provider_shot,
    other_shots,
    provider_other_shots,
    teams_mapping,
    players_mapping=None,
):
    # Get matching game
    game_id = shot.game_id
    season_id = str(api[provider_shot].games.loc[game_id, 'season_id'])
    competition_id = api[provider_shot].games.loc[game_id, 'competition_id']
    home_team, away_team = api[provider_shot].get_home_away_team_id(game_id)
    other_home_team = teams_mapping.set_index(f'{provider_shot}_id').loc[
        home_team, f'{provider_other_shots}_id'
    ]
    other_away_team = teams_mapping.set_index(f'{provider_shot}_id').loc[
        away_team, f'{provider_other_shots}_id'
    ]
    other_games = api[provider_other_shots].games
    other_game_id = (
        other_games[
            (other_games.home_team_id == other_home_team)
            & (other_games.away_team_id == other_away_team)
            & (other_games.competition_id == competition_id)
            & (other_games.season_id.astype(str) == season_id)
        ]
        .iloc[0]
        .name
    )
    other_shots_in_game = other_shots[other_shots.game_id == other_game_id]
    # Get matching shot-taker
    if players_mapping is not None:
        player_id = shot.player_id
        other_player_id = players_mapping.set_index(f'{provider_shot}_id').loc[
            int(player_id), f'{provider_other_shots}_id'
        ]
        other_shots_by_player = other_shots_in_game[
            other_shots_in_game.player_id == other_player_id
        ]
    else:
        other_shots_by_player = other_shots_in_game
    # Get shots in same period
    period_id = shot.period_id
    other_shots_by_player_in_period = other_shots_by_player[
        other_shots_by_player.period_id == period_id
    ]
    # Get shots that happened around the same time
    ts = shot.time_seconds
    best_match = other_shots_by_player_in_period.iloc[
        (other_shots_by_player_in_period['time_seconds'] - ts)
        .abs()
        .argsort()[:1]
    ].iloc[0]
    if abs(ts - best_match.time_seconds) < 3:
        return best_match
    return None


def sample_temporal(api, size_val=0.0, size_test=0.2):
    game_ids = api.games.sort_values(by='game_date').index.values
    nb_games = len(game_ids)
    games_train = game_ids[
        0 : math.floor((1 - size_val - size_test) * nb_games)
    ]
    games_val = game_ids[
        math.ceil((1 - size_val - size_test) * nb_games) : math.floor(
            (1 - size_test) * nb_games
        )
    ]
    games_test = game_ids[math.ceil((1 - size_test) * nb_games) + 1 : -1]
    return games_train, games_val, games_test
