"""Create a test dataset from StatsBomb data."""
import warnings
from pathlib import Path

import pandas as pd
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader, extract_player_games
from socceraction.spadl.statsbomb import convert_to_actions
from statsbombpy.api_client import NoAuthWarning
from tqdm import tqdm

warnings.simplefilter('ignore', NoAuthWarning)

datasets = [
    # 2018 World Cup
    {
        "season_id": 3,
        "season_name": "2018",
        "competition_id": 43,
        "competition_name": "WC",
    },
]

spadl_datafolder = Path(__file__).resolve().parent

SBL = StatsBombLoader(getter="remote")

for dataset in datasets:
    # Get matches from all selected competition
    matches = SBL.games(dataset["competition_id"], dataset["season_id"])

    matches_verbose = tqdm(list(matches.itertuples()), desc="Loading match data")
    teams, players, player_games = [], [], []

    spadl_h5 = (
        spadl_datafolder
        / f"spadl-statsbomb-{dataset['competition_name']}-{dataset['season_name']}.h5"
    )
    with pd.HDFStore(spadl_h5) as spadlstore:
        spadlstore.put("actiontypes", spadl.actiontypes_df(), format="table")
        spadlstore.put("results", spadl.results_df(), format="table")
        spadlstore.put("bodyparts", spadl.bodyparts_df(), format="table")

        for match in matches_verbose:
            # load data
            teams.append(SBL.teams(match.game_id))
            players.append(SBL.players(match.game_id))
            events = SBL.events(match.game_id)

            # convert data
            player_games.append(extract_player_games(events))
            spadlstore.put(
                f"events/game_{match.game_id}",
                events,
            )
            spadlstore.put(
                f"actions/game_{match.game_id}",
                convert_to_actions(events, match.home_team_id),
                format="table",
            )

        matches["season_id"] = dataset["season_name"]
        matches["competition_id"] = dataset["competition_name"]
        spadlstore.put("games", matches)
        spadlstore.put(
            "teams",
            pd.concat(teams).drop_duplicates("team_id").reset_index(drop=True),
        )
        spadlstore.put(
            "players",
            pd.concat(players).drop_duplicates("player_id").reset_index(drop=True),
        )
        spadlstore.put("player_games", pd.concat(player_games).reset_index(drop=True))
