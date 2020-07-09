"""A collection of feature generators."""
import numpy as np
import pandas as pd
from socceraction.vaep.features import *

_spadl_cfg = {
    'length': 105,
    'width': 68,
    'penalty_box_length': 16.5,
    'penalty_box_width': 40.3,
    'six_yard_box_length': 5.5,
    'six_yard_box_width': 18.3,
    'goal_widht': 7.32,
    'penalty_spot_distance': 11,
    'goal_width': 7.3,
    'goal_length': 2,
    'origin_x': 0,
    'origin_y': 0,
    'circle_radius': 9.15,
}


@simple
def goalangle(actions, cfg=_spadl_cfg):
    dx = cfg['length'] - actions['start_x']
    dy = cfg['width'] / 2 - actions['start_y']
    angledf = pd.DataFrame()
    angledf['shot_angle'] = np.arctan(
        cfg['goal_width']
        * dx
        / (dx ** 2 + dy ** 2 - (cfg['goal_width'] / 2) ** 2)
    )
    angledf.loc[angledf['shot_angle'] < 0, 'shot_angle'] += np.pi
    angledf.loc[(actions['start_x'] >= cfg['length']), 'shot_angle'] = 0
    # Ball is on the goal line
    angledf.loc[
        (actions['start_x'] == cfg['length'])
        & (
            actions['start_y'].between(
                cfg['width'] / 2 - cfg['goal_width'] / 2,
                cfg['width'] / 2 + cfg['goal_width'] / 2,
            )
        ),
        'shot_angle',
    ] = np.pi
    return angledf


def speed(gamestates):
    a0 = gamestates[0]
    spaced = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dt = a0.time_seconds - a.time_seconds
        dt[dt < 1] = 1
        dx = a.end_x - a0.start_x
        spaced['speedx_a0' + (str(i + 1))] = dx.abs() / dt
        dy = a.end_y - a0.start_y
        spaced['speedy_a0' + (str(i + 1))] = dy.abs() / dt
        spaced['speed_a0' + (str(i + 1))] = np.sqrt(dx ** 2 + dy ** 2) / dt
    return spaced


def _caley_shot_matrix(cfg=_spadl_cfg):
    """
    https://cartilagefreecaptain.sbnation.com/2013/11/13/5098186/shot-matrix-i-shot-location-and-expected-goals
    """
    m = (cfg['origin_y'] + cfg['width']) / 2

    zones = []
    # Zone 1 is the central area of the six-yard box
    x1 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length']
    x2 = cfg['origin_x'] + cfg['length']
    y1 = m - cfg['goal_width'] / 2
    y2 = m + cfg['goal_width'] / 2
    zones.append([(x1, y1, x2, y2)])
    # Zone 2 includes the wide areas, left and right, of the six-yard box.
    ## Left
    x1 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length']
    x2 = cfg['origin_x'] + cfg['length']
    y1 = m - cfg['six_yard_box_width'] / 2
    y2 = m - cfg['goal_width'] / 2
    zone_left = (x1, y1, x2, y2)
    ## Right
    x1 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length']
    x2 = cfg['origin_x'] + cfg['length']
    y1 = m + cfg['goal_width'] / 2
    y2 = m + cfg['six_yard_box_width'] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    # Zone 3 is the central area between the edges of the six- and eighteen-yard boxes.
    x1 = cfg['origin_x'] + cfg['length'] - cfg['penalty_box_length']
    x2 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length']
    y1 = m - cfg['six_yard_box_width'] / 2
    y2 = m + cfg['six_yard_box_width'] / 2
    zones.append([(x1, y1, x2, y2)])
    # Zone 4 comprises the wide areas in the eighteen-yard box, further from the endline than the six-yard box extended.
    ## Left
    x1 = cfg['origin_x'] + cfg['length'] - cfg['penalty_box_length']
    x2 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length'] - 2
    y1 = m - cfg['penalty_box_width'] / 2
    y2 = m - cfg['six_yard_box_width'] / 2
    zone_left = (x1, y1, x2, y2)
    ## Right
    x1 = cfg['origin_x'] + cfg['length'] - cfg['penalty_box_length']
    x2 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length'] - 2
    y1 = m + cfg['six_yard_box_width'] / 2
    y2 = m + cfg['penalty_box_width'] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    # Zone 5 includes the wide areas left and right in the eighteen yard box within the six-yard box extended.
    ## Left
    x1 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length'] - 2
    x2 = cfg['origin_x'] + cfg['length']
    y1 = m - cfg['penalty_box_width'] / 2
    y2 = m - cfg['six_yard_box_width'] / 2
    zone_left = (x1, y1, x2, y2)
    ## Right
    x1 = cfg['origin_x'] + cfg['length'] - cfg['six_yard_box_length'] - 2
    x2 = cfg['origin_x'] + cfg['length']
    y1 = m + cfg['six_yard_box_width'] / 2
    y2 = m + cfg['penalty_box_width'] / 2
    zone_right = (x1, y1, x2, y2)
    zones.append([zone_left, zone_right])
    # Zone 6 is the eighteen-yard box extended out to roughly 35 yards (=32m).
    x1 = cfg['origin_x'] + cfg['length'] - 32
    x2 = cfg['origin_x'] + cfg['length'] - cfg['penalty_box_length']
    y1 = m - cfg['penalty_box_width'] / 2
    y2 = m + cfg['penalty_box_width'] / 2
    zones.append([(x1, y1, x2, y2)])
    # Zone 7 is the deep, deep area beyond that
    x1 = cfg['origin_x']
    x2 = cfg['origin_x'] + cfg['length'] - 32
    y1 = cfg['origin_y']
    y2 = cfg['origin_y'] + cfg['width']
    zones.append([(x1, y1, x2, y2)])
    # Zone 8 comprises the regions right and left of the box.
    ## Left
    x1 = cfg['origin_x'] + cfg['length'] - 32
    x2 = cfg['origin_x'] + cfg['length']
    y1 = cfg['origin_y'] + cfg['width']
    y2 = m + cfg['penalty_box_width'] / 2
    zone_left = (x1, y1, x2, y2)
    ## Right
    x1 = cfg['origin_x'] + cfg['length'] - 32
    x2 = cfg['origin_x'] + cfg['length']
    y1 = cfg['origin_y']
    y2 = m - cfg['penalty_box_width'] / 2
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
                'start_angle_to_goal_a0',
            ] -= (np.pi / 2)
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
        zonedf = actions[['start_x', 'start_y']].copy()
        if symmetrical:
            m = (cfg['origin_y'] + cfg['width']) / 2
            zonedf.loc[zonedf.start_y > m, 'start_y'] -= m
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
        zonedf = actions[['start_x', 'start_y']].copy()
        zonedf[name] = [0] * len(actions)  # zone 0 if no match
        for (i, zone) in enumerate(zones):
            for subzone in zone:
                zonedf.loc[
                    np.apply_along_axis(
                        is_in_zone(subzone),
                        1,
                        zonedf[['start_x', 'start_y']].values,
                    ),
                    name,
                ] = (i + 1)
        zonedf[name] = pd.Categorical(
            zonedf[name], categories=list(range(len(zones) + 1)), ordered=False
        )
        return zonedf[[name]]

    return fn


caley_grid = custom_grid('caley_zone', _caley_shot_matrix(), _point_in_rect)


all_features = [
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
    goalangle,
    caley_grid,
    triangular_grid(
        'angle_zone',
        [-50, -20, 20, 50],
        [2, 4, 8, 11, 16, 24, 34, 50],
        symmetrical=True,
    ),
]
