"""A collection of attribute (i.e., feature and label) generators.

There are three types of generators:

gamestates
   Generators which calculate a set of attributes based on the shot and
   the N previous actions (i.e., shot context). The input is a list of
   gamestates. Internally each game state is represented as a list of SPADL
   action dataframes :math:`[a_0, a_1, ...]` where each row in the :math:`a_i`
   dataframe contains the previous action of the action in the same row in the
   :math:`a_{i-1}` dataframe. :math:`a_0` is the shot action.

actions
   Generators which calculate a set of attributes based on the shot and
   all preceding actions. The input is a :class:`pandas.DataFrame` of actions
   in SPADL format and a boolean mask to select the shots for which attributes
   should be computed.

events
   Generators which calculate a set of attributes based on the original
   event data. These generators are provider-specific. The input is
   a :class:`pandas.DataFrame` of events and a series with event IDs to select
   the shots for which attributes should be computed.

The types are specified using the ``ftype`` decorator. Only functions, which
have a parameter called "ftype" are seen by soccer-xg as a generator. Others
will not be calculated.

As the "gamestates" and "actions" generators compute attributes from
SPADL actions, they work for all data providers that are supported by the
SoccerAction library.
"""
import math
import warnings

import numpy as np
import pandas as pd

from socceraction import spadl
import socceraction.spadl.config as spadlcfg
import socceraction.vaep.features as fs
from socceraction.vaep.features import simple

_spadl_cfg = {
    "length": 105,
    "width": 68,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.3,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "goal_width": 7.32,
    "penalty_spot_distance": 11,
    "goal_length": 2,
    "origin_x": 0,
    "origin_y": 0,
    "circle_radius": 9.15,
}


def ftype(value):
    """
    This method returns a decorator that sets the property key of the function to value
    """

    def decorate_func(func):
        setattr(func, "ftype", value)
        return func

    return decorate_func


# ############################################################################
# SoccerAction-style gamestate attributes
# ############################################################################

actiontype = ftype("gamestates")(fs.actiontype)
actiontype_onehot = ftype("gamestates")(fs.actiontype_onehot)
result = ftype("gamestates")(fs.result)
result_onehot = ftype("gamestates")(fs.result_onehot)
actiontype_result_onehot = ftype("gamestates")(fs.actiontype_result_onehot)
bodypart = ftype("gamestates")(fs.bodypart)
bodypart_onehot = ftype("gamestates")(fs.bodypart_onehot)
startlocation = ftype("gamestates")(fs.startlocation)
endlocation = ftype("gamestates")(fs.endlocation)
startpolar = ftype("gamestates")(fs.startpolar)
endpolar = ftype("gamestates")(fs.endpolar)
team = ftype("gamestates")(fs.team)
movement = ftype("gamestates")(fs.movement)
time_delta = ftype("gamestates")(fs.time_delta)
space_delta = ftype("gamestates")(fs.space_delta)


@ftype("gamestates")
def speed(gamestates):
    """Get the movement speed of the ball between the last and previous actions.

    Parameters
    ----------
    gamestates : list(pd.DataFrame)
        The game states of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the horizontal ('speedx_a0i'), vertical
        ('speedy_a0i') and total ('speed_a0i') movement speed of the ball
        between each <nb_prev_actions> action ai and action a0.
    """
    a0 = gamestates[0]
    spaced = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dt = a0.time_seconds - a.time_seconds
        dt[dt < 1] = 1
        dx = a.end_x - a0.start_x
        spaced["speedx_a0" + (str(i + 1))] = dx.abs() / dt
        dy = a.end_y - a0.start_y
        spaced["speedy_a0" + (str(i + 1))] = dy.abs() / dt
        spaced["speed_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2) / dt
    return spaced


# ############################################################################
# Attributes on SPADL shots
# ############################################################################


@ftype("actions")
def goal_from_shot(actions, shot_mask):
    """Determine whether a goal was scored from the current action.

    This label can be used to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each shot set to
        True if a goal was scored from the current shot; otherwise False.
    """
    shots = actions.loc[shot_mask]
    goaldf = pd.DataFrame(index=shots.index)
    goaldf["goal"] = shots["result_name"] == "success"
    return goaldf


@ftype("actions")
def shot_dist(actions, shot_mask):
    """Compute the distance to the middle of the goal.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the distance to the middle of the goal
        ('dist_shot').
    """
    shots = actions.loc[shot_mask]
    distdf = pd.DataFrame(index=shots.index)
    dx = (_spadl_cfg["length"] - shots["start_x"]).values
    dy = (_spadl_cfg["width"] / 2 - shots["start_y"]).values
    distdf["dist_shot"] = np.sqrt(dx**2 + dy**2)
    return distdf


@ftype("actions")
def shot_location(actions, shot_mask):
    """Compute the distance to the mid line and goal line.

    This corresponds to the absolute x- and y-coordinates of the shot with the
    origin at the center of the goal.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the distance to the mid line ('dy_shot')
        and a column for the distance to the goal line ('dx_shot').
    """
    shots = actions.loc[shot_mask]
    locationdf = pd.DataFrame(index=shots.index)
    locationdf["dx_shot"] = _spadl_cfg["length"] - shots["start_x"]
    locationdf["dy_shot"] = (_spadl_cfg["width"] / 2 - shots["start_y"]).abs()
    return locationdf


@ftype("actions")
def shot_angle(actions, shot_mask):
    """Compute the angle to the middle of the goal.

    This corresponds to the angle in a polar coordinate system with the origin
    at the center of the goal.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the angle to the middle of the goal
        ('angle_shot').
    """
    shots = actions.loc[shot_mask]
    polardf = pd.DataFrame(index=shots.index)
    dx = (_spadl_cfg["length"] - shots["start_x"]).abs().values
    dy = (_spadl_cfg["width"] / 2 - shots["start_y"]).abs().values
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["angle_shot"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@ftype("actions")
def shot_visible_angle(actions, shot_mask):
    """Compute the angle formed between the shot location and the two goal posts.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the angle formed between the shot location
        and the two goal posts ('visible_angle_shot').

    References
    ----------
    .. [1] Sumpter, David. "The Geometry of Shooting", 8 January 2017,
       https://medium.com/@Soccermatics/the-geometry-of-shooting-ae7a67fdf760,
    """
    shots = actions.loc[shot_mask]
    dx = _spadl_cfg["length"] - shots["start_x"]
    dy = _spadl_cfg["width"] / 2 - shots["start_y"]
    angledf = pd.DataFrame(index=shots.index)
    angledf["visible_angle_shot"] = np.arctan(
        _spadl_cfg["goal_width"] * dx / (dx**2 + dy**2 - (_spadl_cfg["goal_width"] / 2) ** 2)
    )
    angledf.loc[angledf["visible_angle_shot"] < 0, "visible_angle_shot"] += np.pi
    angledf.loc[(shots["start_x"] >= _spadl_cfg["length"]), "visible_angle_shot"] = 0
    # Ball is on the goal line
    angledf.loc[
        (shots["start_x"] == _spadl_cfg["length"])
        & (
            shots["start_y"].between(
                _spadl_cfg["width"] / 2 - _spadl_cfg["goal_width"] / 2,
                _spadl_cfg["width"] / 2 + _spadl_cfg["goal_width"] / 2,
            )
        ),
        "visible_angle_shot",
    ] = np.pi
    return angledf


@ftype("actions")
def shot_relative_angle(actions, shot_mask):
    """Compute the relative angle to goal.

    If a player is in a central position, the angle is 1. If a player is wide
    of the posts, this feature takes the angle from the shot's location to the
    nearest post. For example, if a player is at a 45-degree angle to the
    nearest post, the angle is 0.5.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the relative angle to the goal
        ('relative_angle_shot').

    References
    ----------
    .. [1] Caley, Micheal. "Premier League Projections and New Expected Goals"
       Cartilage Freecaptain SBNation, 2015,
       https://cartilagefreecaptain.sbnation.com/2015/10/19/9295905/premier-leagueprojections-and-new-expected-goals
    """
    angledf = actions.loc[shot_mask, ["start_x", "start_y"]].copy()
    angledf["dx"] = (_spadl_cfg["length"] - angledf["start_x"]).abs().values
    left_post = _spadl_cfg["width"] / 2 + _spadl_cfg["goal_width"] / 2
    right_post = _spadl_cfg["width"] / 2 - _spadl_cfg["goal_width"] / 2
    angledf.loc[angledf.start_y > left_post, "dy"] = (left_post - angledf["start_y"]).abs().values
    angledf.loc[angledf.start_y < right_post, "dy"] = (
        (right_post - angledf["start_y"]).abs().values
    )
    is_center = (angledf.start_y <= left_post) & (angledf.start_y >= right_post)
    with np.errstate(divide="ignore", invalid="ignore"):
        angledf.loc[~is_center, "relative_angle_shot"] = 1 - (
            np.nan_to_num(np.arctan(angledf.dy / angledf.dx)) / (math.pi / 2)
        )
    angledf.loc[is_center, "relative_angle_shot"] = 1.0
    return angledf[["relative_angle_shot"]]


@ftype("actions")
def shot_bodypart(actions, shot_mask):
    """Return the body part used to take the shot.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the body part used to take the shot
        ('bodypart_name_shot').
    """
    shots = actions.loc[shot_mask]
    bodypartdf = pd.DataFrame(index=shots.index)
    bodypartdf["bodypart_name_shot"] = pd.Categorical(
        shots["bodypart_name"].replace(["foot_left", "foot_right"], "foot"),
        categories=["foot", "head", "other"],
        ordered=False,
    )
    return bodypartdf


@ftype("actions")
def shot_bodypart_detailed(actions, shot_mask):
    """Return the body part used to take the shot, distinguishing between
    the left and right foot.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column for the body part used to take the shot
        ('bodypart_name_shot').
    """
    shots = actions.loc[shot_mask]
    bodypartdf = pd.DataFrame(index=shots.index)
    bodypartdf["bodypart_name_shot"] = pd.Categorical(
        shots["bodypart_name"], categories=spadlcfg.bodyparts, ordered=False
    )
    return bodypartdf


@ftype("actions")
def shot_bodypart_onehot(actions, shot_mask):
    """Return the one-hot encoded body part used to take the shot.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a boolean column for each body part that can be used
        to take a shot.
    """
    shots = actions.loc[shot_mask]
    X = pd.DataFrame(index=shots.index)
    for bodypart_name in spadlcfg.bodyparts:
        col = "bodypart_" + bodypart_name + "_shot"
        if bodypart_name == "head/other":
            X[col] = shots["bodypart_name"].isin(["head", "other", "head/other"])
        elif bodypart_name == "foot":
            X[col] = shots["bodypart_name"].isin(["foot", "foot_left", "foot_right"])
        else:
            X[col] = shots["bodypart_name"] == bodypart_name
    return X


@ftype("actions")
def post_dribble(actions, shot_mask):
    """Compute features describing the dribble before the shot.

    Computes the following features:
        - post_dribble: whether the shot follows a previous attempt to beat
          a player
        - carry_length: The distance between the end location of the assisting
          pass and the location of the shot

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a boolean column containing whether the shot was
        preceded by a take-on attempt of the shot-taker ('post_dribble') and
        the distance between the end location of the assisting pass and the
        location of the shot ('carry_length').
    """
    df = {}
    for idx in actions.loc[shot_mask].index:
        carry_length = 0
        maybe_carry = actions.loc[:idx].iloc[-1]
        if maybe_carry.type_name == "dribble":
            dx = maybe_carry.end_x - maybe_carry.start_x
            dy = maybe_carry.end_y - maybe_carry.start_y
            carry_length = math.sqrt(dx**2 + dy**2)
        df[idx] = {"carry_length": carry_length}
    return pd.DataFrame.from_dict(df, orient="index")


@ftype("actions")
def assist_type(actions, shot_mask):
    """Return the assist type.

    One of a long ball, cross, through ball, danger-zone pass,
    and pull-back
    The assist type, which is one of pass, recovery, clearance, direct,
    or rebound

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column containing the assist type of each shot
        ('type_assist').
    """
    df = {}
    for idx, shot in actions.loc[shot_mask].iterrows():
        assist = None
        for _, maybe_assist in actions.loc[:idx].iloc[::-1].iterrows():
            if (
                maybe_assist.type_name.isin(
                    [
                        "pass",
                        "cross",
                        "throw_in",
                        "freekick_crossed",
                        "freekick_short",
                        "corner_crossed",
                        "corner_short",
                        "goalkick",
                    ]
                )
                and maybe_assist.team_id == shot.team_id
                and maybe_assist.period_id == shot.period_id
                and maybe_assist.result_name == "success"
            ):
                assist = maybe_assist
                break
            elif maybe_assist.player_id == shot.player_id and maybe_assist.type_name != "dribble":
                break
            elif maybe_assist.team_id != shot.team_id and maybe_assist.result_name == "success":
                break
        # (assist_type): The assist type
        assist_type = assist.type_name if assist else "direct"
        # TODO (assist_technique): The technique for crosses one of straight,
        # inswinging, or out swinging and whether the pass was a through ball
        df[idx] = {"type_assist": assist_type}
    return pd.DataFrame.from_dict(df, orient="index")


@ftype("actions")
def fastbreak(actions, shot_mask):
    """Get whether the shot was part of a counter attack.

    A fastbreak is defined as a pattern of play in which the team wins the
    ball in its own third and shoots in the last quarter of the pitch within 25
    seconds.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column containing whether shot was part of a fastbreak
        ('fastbreak').
    """
    df = {}
    for idx, shot in actions.loc[shot_mask].iterrows():
        prev_actions = actions[
            (actions.time_seconds < shot.time_seconds)
            & (actions.time_seconds > shot.time_seconds - 25)
            & (actions.period_id == shot.period_id)
        ]
        has_recovery_own_third = not prev_actions[
            (prev_actions.team_id == shot.team_id)
            & (prev_actions.start_x < 105 / 3)
            & (
                prev_actions.type_name.isin(
                    [
                        "tackle",
                        "interception",
                        "keeper_save",
                        "keeper_claim",
                        "keeper_punch",
                        "keeper_pick_up",
                        "clearance",
                    ]
                )
            )
        ].empty
        shot_in_last_quarter = shot.start_x > _spadl_cfg["length"] / 4
        df[idx] = {"fastbreak": has_recovery_own_third and shot_in_last_quarter}
    return pd.DataFrame.from_dict(df, orient="index")


@ftype("actions")
def rebound(actions, shot_mask):
    """Get whether the shot was a rebound.

    A shot is a rebound if one of the two preceding actions was also a shot.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game in SPADL format.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column containing whether shot was a rebound ('rebound')
        and the time since the previous shot ('time_prev_shot').
    """
    df = {}
    for idx, shot in actions.loc[shot_mask].iterrows():
        # (time_prev): Time in seconds from the last shot of the same team
        # in the same half of the same game. The idea here is try to add some
        # information about rebounds.
        _idx = actions.index.get_loc(idx)
        prev_actions = actions.iloc[:_idx]
        prev_shots = prev_actions[
            (prev_actions.type_name.isin(["shot", "shot_penalty", "shot_freekick"]))
            & (prev_actions.team_id == shot.team_id)
            & (prev_actions.period_id == shot.period_id)
        ]
        if not prev_shots.empty:
            time_prev_shot = shot["time_seconds"] - prev_shots.iloc[-1]["time_seconds"]
        else:
            time_prev_shot = float("inf")
        # (rebound): Shot in previous two actions
        rebound = (
            prev_actions.iloc[-2:].type_name.isin(["shot", "shot_penalty", "shot_freekick"]).any()
        )
        df[idx] = {
            "time_prev_shot": time_prev_shot,
            "rebound": rebound,
        }
    return pd.DataFrame.from_dict(df, orient="index")


# ############################################################################
# Transform cartesian coordinates to location in grid
# ############################################################################


def _caley_shot_matrix(cfg=_spadl_cfg):
    """
    https://cartilagefreecaptain.sbnation.com/2013/11/13/5098186/shot-matrix-i-shot-location-and-expected-goals
    """
    m = (cfg["origin_y"] + cfg["width"]) / 2

    zones = []
    # Zone 1 is the central area of the six-yard box
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = m - cfg["goal_width"] / 2
    y2 = m + cfg["goal_width"] / 2
    zones.append([(x1, y1, x2, y2)])
    # Zone 2 includes the wide areas, left and right, of the six-yard box.
    # left
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = m - cfg["six_yard_box_width"] / 2
    y2 = m - cfg["goal_width"] / 2
    zone_left = (x1, y1, x2, y2)
    # right
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = m + cfg["goal_width"] / 2
    y2 = m + cfg["six_yard_box_width"] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    # Zone 3 is the central area between the edges of the six- and
    # eighteen-yard boxes.
    x1 = cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"]
    x2 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"]
    y1 = m - cfg["six_yard_box_width"] / 2
    y2 = m + cfg["six_yard_box_width"] / 2
    zones.append([(x1, y1, x2, y2)])
    # Zone 4 comprises the wide areas in the eighteen-yard box, further from
    # the endline than the six-yard box extended.
    # left
    x1 = cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"]
    x2 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"] - 2
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m - cfg["six_yard_box_width"] / 2
    zone_left = (x1, y1, x2, y2)
    # right
    x1 = cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"]
    x2 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"] - 2
    y1 = m + cfg["six_yard_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    # Zone 5 includes the wide areas left and right in the eighteen yard box
    # within the six-yard box extended.
    # left
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"] - 2
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m - cfg["six_yard_box_width"] / 2
    zone_left = (x1, y1, x2, y2)
    # right
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"] - 2
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = m + cfg["six_yard_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    # Zone 6 is the eighteen-yard box extended out to roughly 35 yards (=32m).
    x1 = cfg["origin_x"] + cfg["length"] - 32
    x2 = cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"]
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    zones.append([(x1, y1, x2, y2)])
    # Zone 7 is the deep, deep area beyond that
    x1 = cfg["origin_x"]
    x2 = cfg["origin_x"] + cfg["length"] - 32
    y1 = cfg["origin_y"]
    y2 = cfg["origin_y"] + cfg["width"]
    zones.append([(x1, y1, x2, y2)])
    # Zone 8 comprises the regions right and left of the box.
    # left
    x1 = cfg["origin_x"] + cfg["length"] - 32
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = cfg["origin_y"] + cfg["width"]
    y2 = m + cfg["penalty_box_width"] / 2
    zone_left = (x1, y1, x2, y2)
    # right
    x1 = cfg["origin_x"] + cfg["length"] - 32
    x2 = cfg["origin_x"] + cfg["length"]
    y1 = cfg["origin_y"]
    y2 = m - cfg["penalty_box_width"] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    return zones


def _point_in_rect(rect):
    x1, y1, x2, y2 = rect

    def fn(point):
        x, y = point
        if x1 <= x and x <= x2:
            if y1 <= y and y <= y2:
                return True
        return False

    return fn


def triangular_grid(name, angle_bins, dist_bins, symmetrical=False):
    @simple
    def fn(actions):
        zonedf = startpolar(actions)
        if symmetrical:
            zonedf.loc[
                zonedf.start_angle_to_goal_a0 > np.pi / 2,
                "start_angle_to_goal_a0",
            ] -= (
                np.pi / 2
            )
        dist_bin = np.digitize(zonedf.start_dist_to_goal_a0, dist_bins)
        angle_bin = np.digitize(zonedf.start_angle_to_goal_a0, angle_bins)
        zonedf[name] = dist_bin * angle_bin + dist_bin
        zonedf[name] = pd.Categorical(
            zonedf[name],
            categories=list(range(len(dist_bins) * len(angle_bins))),
            ordered=False,
        )
        return zonedf[[name]]

    return fn


def rectangular_grid(name, x_bins, y_bins, symmetrical=False, cfg=_spadl_cfg):
    @simple
    def fn(actions):
        zonedf = actions[["start_x", "start_y"]].copy()
        if symmetrical:
            m = (cfg["origin_y"] + cfg["width"]) / 2
            zonedf.loc[zonedf.start_y > m, "start_y"] -= m
        x_bin = np.digitize(zonedf.start_x, x_bins)
        y_bin = np.digitize(zonedf.start_y, y_bins)
        zonedf[name] = x_bin * y_bin + y_bin
        zonedf[name] = pd.Categorical(
            zonedf[name],
            categories=list(range(len(x_bins) * len(y_bins))),
            ordered=False,
        )
        return zonedf[[name]]

    return fn


def custom_grid(name, zones, is_in_zone):
    @simple
    def fn(actions):
        zonedf = actions[["start_x", "start_y"]].copy()
        zonedf[name] = [0] * len(actions)  # zone 0 if no match
        for i, zone in enumerate(zones):
            for subzone in zone:
                zonedf.loc[
                    np.apply_along_axis(
                        is_in_zone(subzone),
                        1,
                        zonedf[["start_x", "start_y"]].values,
                    ),
                    name,
                ] = (
                    i + 1
                )
        zonedf[name] = pd.Categorical(
            zonedf[name], categories=list(range(len(zones) + 1)), ordered=False
        )
        return zonedf[[name]]

    return fn


caley_grid = ftype("gamestates")(custom_grid("caley_zone", _caley_shot_matrix(), _point_in_rect))

# ############################################################################
# StatsBomb-specific attributes
# ############################################################################


def _sb_to_spadl(sb_x, sb_y):
    spadl_x = ((sb_x - 1) / 119) * _spadl_cfg["length"]
    spadl_y = _spadl_cfg["width"] - ((sb_y - 1) / 79) * _spadl_cfg["width"]
    return spadl_x, spadl_y


def _get_intersect(a1, a2, b1, b2):
    """Returns the point of intersection of the lines passing through a2,a1 and b2,b1.

    Parameters
    ----------
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return (x / z, y / z)


def _overlap(min1, max1, min2, max2):
    start = max(min1, min2)
    end = min(max1, max2)
    d = end - start
    if d < 0:
        return False, None, None
    else:
        return d, start, end


def _is_inside_triangle(point, tri_points):
    Dx, Dy = point

    A, B, C = tri_points
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C

    M1 = np.array([[Dx - Bx, Dy - By, 0], [Ax - Bx, Ay - By, 0], [1, 1, 1]])
    M1 = np.linalg.det(M1)

    M2 = np.array([[Dx - Ax, Dy - Ay, 0], [Cx - Ax, Cy - Ay, 0], [1, 1, 1]])
    M2 = np.linalg.det(M2)

    M3 = np.array([[Dx - Cx, Dy - Cy, 0], [Bx - Cx, By - Cy, 0], [1, 1, 1]])
    M3 = np.linalg.det(M3)

    if M1 == 0 or M2 == 0 or M3 == 0:
        # lies on the arms of Triangle
        return True
    if (M1 > 0 and M2 > 0 and M3 > 0) or (M1 < 0 and M2 < 0 and M3 < 0):
        # if products is non 0 check if all of their sign is same
        # lies inside the Triangle
        return True
    # lies outside the Triangle
    return False


@ftype("events")
def statsbomb_open_goal(events, shot_mask):
    """Get whether the shot was taken into an open goal.

    This is derived from the 'open_goal' annotation in StatsBomb's event
    stream data.

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
    """
    output = {}
    for idx, shot in events.loc[shot_mask].iterrows():
        if "shot" in shot.extra:
            output[idx] = {"open_goal": "open_goal" in shot.extra['shot']}

    output = pd.DataFrame.from_dict(output, orient="index")
    return output


@ftype("events")
def statsbomb_first_touch(events, shot_mask):
    """Get whether the shot was a first-touch shot.

    This is derived from the 'first_time' annotation in StatsBomb's event
    stream data.

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
    """
    output = {}
    for idx, shot in events.loc[shot_mask].iterrows():
        if "shot" in shot.extra:
            output[idx] = {"first_touch": "first_time" in shot.extra['shot']}

    output = pd.DataFrame.from_dict(output, orient="index")
    return output


@ftype("events")
def statsbomb_free_projection(events, shot_mask):
    """Get the free projection area.

    This feature represents the proportion of the goal that is left uncovered by
    the goalkeeper and defenders. To capture the fact that players are not
    static and will react to the shot, we assume that the defending player has
    an effective span of one arm length (80 cm) and the goalkeeper has an
    effect of two arm lengths (160 cm).

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column containing the free projection area
        ('free_projection_pct') and the number of gaps in the projection area
        ('free_projection_gaps').

    References
    ----------
    .. [1] Cem Arslan, Pieter Robberechts and Jesse Davis. "Enhancing xG
       Models with Freeze Frame Data". DTAI Sports Analytics Lab, September 2,
       2020. https://dtai.cs.kuleuven.be/sports/blog/enhancing-xg-models-with-freeze-frame-data
    """
    output = {}
    # We have to use StatsBomb coordinates here
    corner1 = [120, 0]
    corner2 = [120, 80]
    goal = [36, 44]
    for idx, shot in events.loc[shot_mask].iterrows():
        if "shot" not in shot.extra or "freeze_frame" not in shot.extra["shot"]:
            # No freeze frame data available for this shot
            continue
        freeze_frame = shot.extra["shot"]["freeze_frame"]
        start_x, start_y = shot.location
        # By default, the entire goal is free
        free_projection = [goal]
        # Now we remove the area blocked by each defending player
        defenders = [t for t in freeze_frame if not t["teammate"]]
        for defender in defenders:
            def_x, def_y = defender["location"]
            # goalkeepers span 1.60m; defenders span 0.80m
            def_position = defender["position"]["name"]
            def_width = (
                160 / 91.44 if def_position == "Goalkeeper" else 80 / 91.44
            )  # convert to yards
            if def_x > start_x:
                _, left_bound_y = _get_intersect(
                    [start_x, start_y], [def_x, def_y - def_width / 2], corner1, corner2
                )
                _, right_bound_y = _get_intersect(
                    [start_x, start_y], [def_x, def_y + def_width / 2], corner1, corner2
                )
                new_free_projection = []
                for projection in free_projection:
                    d, overlap_start, overlap_end = _overlap(
                        left_bound_y, right_bound_y, *projection
                    )
                    if d:
                        new_free_projection.append([projection[0], overlap_start])
                        new_free_projection.append([overlap_end, projection[1]])
                    else:
                        new_free_projection.append(projection)
                free_projection = [p for p in new_free_projection if p[1] - p[0] > 0]
        output[idx] = {
            "free_projection_gaps": len(free_projection),
            "free_projection_pct": np.sum(np.diff(free_projection)) / np.diff(goal)[0],
        }
    output = pd.DataFrame.from_dict(output, orient="index")
    return output


@ftype("events")
def statsbomb_goalkeeper_position(events, shot_mask):
    """Get the goalkeeper's position.

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column containing the goalkeeper's x-coordinate
        ('goalkeeper_x'), y-coordinate ('goalkeeper_y'), distance to the ball
        ('goalkeeper_dist_to_ball'), distance to the center of the goal
        ('goalkeeper_dist_to_goal') and the angle to the center of the goal
        ('goalkeeper_angle_to_goal').
    """
    output = {}
    for idx, shot in events.loc[shot_mask].iterrows():
        if "shot" not in shot.extra or "freeze_frame" not in shot.extra["shot"]:
            # No freeze frame data available for this shot
            continue
        freeze_frame = shot.extra["shot"]["freeze_frame"]
        goalkeeper = next(
            (
                t
                for t in freeze_frame
                # Cartesian coordinates
                if not t["teammate"] and t["position"]["name"] == "Goalkeeper"
            ),
            None,
        )
        if goalkeeper is None:
            # The goalkeeper is not included in the freeze_frame
            continue

        # Cartesian coordinates
        goalkeeper_x, goalkeeper_y = _sb_to_spadl(
            goalkeeper["location"][0], goalkeeper["location"][1]
        )

        # Polar coordinates
        dx_gk = _spadl_cfg["length"] - goalkeeper_x
        dy_gk = _spadl_cfg["width"] / 2 - goalkeeper_y
        goalkeeper_dist_to_goal = math.sqrt(dx_gk**2 + dy_gk**2)
        goalkeeper_angle_to_goal = math.atan2(dy_gk, dx_gk)  # if dx_gk > 0 else 0

        ball_x, ball_y = _sb_to_spadl(shot["location"][0], shot["location"][1])
        dx_kb = goalkeeper_x - ball_x
        dy_kb = goalkeeper_y - ball_y
        goalkeeper_dist_to_ball = math.sqrt(dx_kb**2 + dy_kb**2)

        output[idx] = {
            "goalkeeper_x": goalkeeper_x,
            "goalkeeper_y": goalkeeper_y,
            "goalkeeper_dist_to_ball": goalkeeper_dist_to_ball,
            "goalkeeper_dist_to_goal": goalkeeper_dist_to_goal,
            "goalkeeper_angle_to_goal": goalkeeper_angle_to_goal,
        }
    output = pd.DataFrame.from_dict(output, orient="index")
    return output


@ftype("events")
def statsbomb_defenders_position(events, shot_mask):
    """Get features describing the position of the defending players.

    The following features are computed:
        - dist_to_defender: The distance to the closest defender.
        - under_pressure: Whether the shot was taken under pressure (StatsBomb
          definition).
        - nb_defenders_in_shot_line: The number of defenders in the visible
          angle to the goal.
        - nb_defenders_behind_ball: The number of defenders behind the ball.
        - one_on_one: Whether the shot was a 1 versus 1 situation. We define
          a 1v1 as a shot for which only the goalkeeper is inside the triangle
          formed by joining the shot location, the right post, and the left
          post. Also, the striker must be higher up the pitch than any other
          defending player apart from the opposition goalkeeper and the shot
          is made with the player's foot. [1]_

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame

    References
    ----------
    .. [1] Wear, Matthew, et al. "Learning from the Pros: Extracting
       Professional Goalkeeper Technique from Broadcast Footage." arXiv preprint
       arXiv:2202.12259 (2022).
    """
    output = {}
    left_post = (_spadl_cfg["length"], _spadl_cfg["width"] / 2 - _spadl_cfg["goal_width"] / 2)
    right_post = (_spadl_cfg["length"], _spadl_cfg["width"] / 2 + _spadl_cfg["goal_width"] / 2)
    for idx, shot in events.loc[shot_mask].iterrows():
        if "shot" not in shot.extra or "freeze_frame" not in shot.extra["shot"]:
            # No freeze frame data available for this shot
            continue
        freeze_frame = shot.extra["shot"]["freeze_frame"]
        defenders = [t for t in freeze_frame if not t["teammate"]]
        distances = []
        in_shot_line = []
        behind_ball = []
        for defender in defenders:
            if defender["teammate"] or defender["position"]["name"] == "Goalkeeper":
                continue
            defender_x, defender_y = _sb_to_spadl(defender["location"][0], defender["location"][1])
            ball_x, ball_y = _sb_to_spadl(shot["location"][0], shot["location"][1])
            distances.append(math.sqrt((ball_x - defender_x) ** 2 + (ball_y - defender_y) ** 2))
            in_shot_line.append(
                _is_inside_triangle(
                    (defender_x, defender_y), [left_post, (ball_x, ball_y), right_post]
                )
            )
            behind_ball.append(defender_x > ball_x)
        output[idx] = {
            "dist_to_defender": min(distances, default=float("inf")),
            "under_pressure": shot.under_pressure,
            "nb_defenders_in_shot_line": sum(in_shot_line),
            "nb_defenders_behind_ball": sum(behind_ball),
            "one_on_one": (
                sum(behind_ball) == 0
                and sum(in_shot_line) == 0
                and shot.extra["shot"]["body_part"]["name"] in ["Left Foot", "Right Foot"]
            ),
        }
    output = pd.DataFrame.from_dict(output, orient="index")
    output["one_on_one"] = output["one_on_one"].astype('boolean')
    output["under_pressure"] = output["under_pressure"].astype('boolean')
    return output


@ftype("events")
def statsbomb_assist(events, shot_mask):  # noqa: C901
    """Get features describing the assist.

    The following features are computed:
        - end_x_assist: The assisting pass' x-coordinate
        - end_y_assist: The assisting pass' y-coordinate
        - carry_dist: The distance between the end location of the assisting
          pass and the location of the shot.
        - type_assist: The assist type, which is one of 'standard_pass',
          'free_kick', 'corner', 'throw_in', 'cross', 'cut_back' or 'through_ball'.
        - height_assist: The peak height of the assisting pass, which is one of
          'ground', 'low' (under shoulder level) or 'high' (above shoulder
          level).

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame

    References
    ----------
    .. [1] Wear, Matthew, et al. "Learning from the Pros: Extracting
       Professional Goalkeeper Technique from Broadcast Footage." arXiv preprint
       arXiv:2202.12259 (2022).
    """
    output = {}
    for event_id, shot in events.loc[shot_mask].iterrows():
        if "shot" not in shot.extra or "key_pass_id" not in shot.extra["shot"]:
            # No assist for this shot
            continue
        assist = events.loc[shot.extra["shot"]["key_pass_id"]]
        assist_x, assist_y = _sb_to_spadl(
            assist.extra["pass"]["end_location"][0], assist.extra["pass"]["end_location"][1]
        )
        shot_x, shot_y = _sb_to_spadl(shot["location"][0], shot["location"][1])

        assist_type = "standard_pass"
        assist_height = "ground"
        if "pass" in assist.extra:
            # assist type
            if "cross" in assist.extra["pass"]:
                assist_type = "cross"
            elif "cut_back" in assist.extra["pass"]:
                assist_type = "cut_back"
            elif "technique" in assist.extra["pass"]:
                if assist.extra["pass"]["technique"]["name"] == "Through Ball":
                    assist_type = "through_ball"
            # special pass type
            if "type" in assist.extra["pass"]:
                if assist.extra["pass"]["type"]["name"] == "Free Kick":
                    assist_type = "free_kick"
                elif assist.extra["pass"]["type"]["name"] == "Corner":
                    assist_type = "corner"
                elif assist.extra["pass"]["type"]["name"] == "Throw-in":
                    assist_type = "throw_in"

            # assist height
            if "height" in assist.extra["pass"]:
                m = {
                    "Ground Pass": "ground",
                    "Low Pass": "low",
                    "High Pass": "high",
                }
                assist_height = m[assist.extra["pass"]["height"]["name"]]

        output[event_id] = {
            "end_x_assist": assist_x,
            "end_y_assist": assist_y,
            "carry_dist": math.sqrt((shot_x - assist_x) ** 2 + (shot_y - assist_y) ** 2),
            "type_assist": assist_type,
            "height_assist": assist_height,
        }

    output = pd.DataFrame.from_dict(output, orient="index")
    output["type_assist"] = pd.Categorical(
        output["type_assist"],
        categories=[
            "standard_pass",
            "free_kick",
            "corner",
            "throw_in",
            "cross",
            "cut_back",
            "through_ball",
        ],
        ordered=False,
    )
    output["height_assist"] = pd.Categorical(
        output["height_assist"], categories=["ground", "low", "high"], ordered=True
    )
    return output


@ftype("events")
def statsbomb_counterattack(events, shot_mask):
    """Get whether a shot was from a counterattack.

    This is derived from the 'play_pattern' annotation in StatsBomb's event
    stream data.

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
    """
    output = {}
    for idx, shot in events.loc[shot_mask].iterrows():
        output[idx] = {
            "from_counterattack": shot.play_pattern_name == "From Counter",
        }

    output = pd.DataFrame.from_dict(output, orient="index")
    return output


@ftype("events")
def statsbomb_shot_impact_height(events, shot_mask):
    """Get the height of the ball when the shot was taken.

    This is derived from the bodypart and technique that was used to take the
    shot. Possible values are 'ground', 'low' (below shoulder level) and
    'high' (above shoulder level).

    Parameters
    ----------
    events : pd.DataFrame
        The StatsBomb events of a game.
    shot_mask : pd.Series
        A boolean mask to select the shots for which attributes should be
        computed.

    Returns
    -------
    pd.DataFrame
    """
    # The height of the ball when the ball is touched is not included,
    # but we can use body part and technique as a proxy for this
    output = {}
    for idx, shot in events.loc[shot_mask].iterrows():
        if "shot" not in shot.extra or "technique" not in shot.extra["shot"]:
            # No freeze frame data available for this shot
            continue
        height = "ground"
        if shot.extra["shot"]["body_part"]["name"] == "Head":
            if shot.extra["shot"]["technique"]["name"] == "Diving Header":
                height = "low"
            else:
                height = "high"
        elif shot.extra["shot"]["body_part"]["name"] == "Other":
            height = "low"
        elif shot.extra["shot"]["technique"]["name"] == "Half Volley":
            height = "low"
        elif shot.extra["shot"]["technique"]["name"] == "Volley":
            height = "low"
        elif shot.extra["shot"]["technique"]["name"] == "Overhead Kick":
            height = "high"
        output[idx] = {"impact_height": height}

    output = pd.DataFrame.from_dict(output, orient="index")
    output["impact_height"] = pd.Categorical(
        output["impact_height"], categories=["ground", "low", "high"], ordered=True
    )
    return output


default_features = [
    actiontype,
    bodypart,
    result,
    startlocation,
    endlocation,
    movement,
    space_delta,
    startpolar,
    endpolar,
    team,
    time_delta,
    speed,
    shot_angle,
    caley_grid,
    ftype("gamestates")(
        triangular_grid(
            "angle_zone",
            [-50, -20, 20, 50],
            [2, 4, 8, 11, 16, 24, 34, 50],
            symmetrical=True,
        )
    ),
]

default_labels = [goal_from_shot]


def compute_attributes(
    game,
    actions,
    events=None,
    xfns=default_features,
    yfns=default_labels,
    shotfilter=None,
    nb_prev_actions=3,
):
    """Extract xG features for a given game.

    Parameters
    ----------
    game : pd.Series
        The game information.
    actions : pd.DataFrame
        A DataFrame containing SPADL actions.
    events: pd.DataFrame
        A DataFrame containing the raw provider-specific events corresponding
        to ``actions``. Can be used to calculate provider-specific features.
    xfns : list(callable)
        List of feature generators to apply. Defaults to ``default_features``.
    yfns : list(callable)
        List of label generators to apply. Defaults to ``default_labels``.
    shotfilter: callable(pd.Series) -> bool
        A function that takes a shot (in SPADL format) and returns True if the
        shot should be used for feature extraction. If None, all shots will be
        used (excluding own-goals).
    nb_prev_actions: int
        The number of previous actions to consider when calculating features
        to model the game state context preceding a shot.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the calculated features.
    """
    # add names for result, bodypart and type
    actions = spadl.utils.add_names(actions)

    # select shots
    if shotfilter is None:
        # filter shots and ignore own goals
        shot_mask = actions.type_name.isin(
            ["shot", "shot_penalty", "shot_freekick"]
        ) & actions.result_name.isin(["fail", "success"])
    else:
        shot_mask = actions.apply(lambda a: shotfilter(a), axis=1)
    shot_actions_idx = actions.index[shot_mask]
    shot_events_idx = actions.loc[shot_mask, "original_event_id"]

    # handle inputs with no shots or no attributes
    if shot_mask.sum() < 1:
        # TODO: create the expected columns
        return pd.DataFrame()
    if len(xfns + yfns) < 1:
        return pd.DataFrame(index=shot_actions_idx)

    # convert actions to ltr orientation
    actions_ltr = spadl.utils.play_left_to_right(actions, game.home_team_id)
    # convert actions to ltr shot gamestates
    gamestates = fs.gamestates(actions, nb_prev_actions)
    gamestates_ltr = fs.play_left_to_right(gamestates, game.home_team_id)
    shot_gamestates_ltr = [states.loc[shot_mask].copy() for states in gamestates_ltr]
    # remove post-shot attributes to avoid target leakage
    shot_gamestates_ltr[0]["end_x"] = float("NaN")
    shot_gamestates_ltr[0]["end_y"] = float("NaN")
    shot_gamestates_ltr[0]["result_id"] = float("NaN")

    # compute features and labels
    def _apply_fns(fns):
        attrs = []
        for fn in fns:
            if getattr(fn, "ftype", None) == "gamestates":
                attrs.append(fn(shot_gamestates_ltr).set_index(shot_events_idx))
            elif getattr(fn, "ftype", None) == "actions":
                attrs.append(fn(actions_ltr, shot_mask).set_index(shot_events_idx))
            elif getattr(fn, "ftype", None) == "events":
                attrs.append(fn(events, shot_events_idx))
            else:
                warnings.warn("Unknown attribute type for {}.".format(fn.__name__), stacklevel=2)
        attrs = pd.concat(attrs, axis=1).loc[shot_events_idx].set_index(shot_actions_idx)
        attrs.index.name = "action_id"
        # fill missing values
        missing_bool = attrs.select_dtypes(include=['boolean']).columns
        attrs[missing_bool] = attrs[missing_bool].fillna(False).astype(bool)
        # replace 'a0' by 'shot' in each feature name
        attrs.rename(columns=lambda s: s.replace("a0", "shot"), inplace=True)
        return attrs

    X = _apply_fns(xfns)
    y = _apply_fns(yfns)

    return X, y
