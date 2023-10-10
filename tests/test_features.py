import soccer_xg.features as fs
import pytest
import math


def test_shot_angle(shot):
    # Test output feature names
    df = fs.shot_angle(shot, [True])
    assert df.columns.tolist() == ["angle_shot"]
    assert len(df) == 1
    # Ball on goalline in center of goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 34]
    df = fs.shot_angle(shot, [True])
    assert df.loc[0, "angle_shot"] == 0
    # Ball on goalline next to goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 0]
    df = fs.shot_angle(shot, [True])
    assert df.loc[0, "angle_shot"] == math.pi / 2
    # Constant as ball moves away from goal
    shot.loc[:, ["start_x", "start_y"]] = [105 - 10, 34]
    df1 = fs.shot_angle(shot, [True])
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34]
    df2 = fs.shot_angle(shot, [True])
    assert df1.loc[0, "angle_shot"] == df2.loc[0, "angle_shot"]
    # Is the same left and right from the goal
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34 - 5]
    df1 = fs.shot_angle(shot, [True])
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34 + 5]
    df2 = fs.shot_angle(shot, [True])
    assert df1.loc[0, "angle_shot"] == df2.loc[0, "angle_shot"]


def test_shot_visible_angle(shot):
    # Test output feature names
    df = fs.shot_visible_angle(shot, [True])
    assert df.columns.tolist() == ["visible_angle_shot"]
    assert len(df) == 1
    # Ball on goalline in center of goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 34]
    df = fs.shot_visible_angle(shot, [True])
    assert df.loc[0, "visible_angle_shot"] == math.pi
    # Ball on goalline next to goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 0]
    df = fs.shot_visible_angle(shot, [True])
    assert df.loc[0, "visible_angle_shot"] == 0
    # Decreases as ball moves away from goal
    shot.loc[:, ["start_x", "start_y"]] = [105 - 10, 34]
    df1 = fs.shot_visible_angle(shot, [True])
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34]
    df2 = fs.shot_visible_angle(shot, [True])
    assert df1.loc[0, "visible_angle_shot"] > df2.loc[0, "visible_angle_shot"]
    # Is the same left and right from the goal
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34 - 5]
    df1 = fs.shot_visible_angle(shot, [True])
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34 + 5]
    df2 = fs.shot_visible_angle(shot, [True])
    assert df1.loc[0, "visible_angle_shot"] == df2.loc[0, "visible_angle_shot"]

def test_shot_relative_angle(shot):
    # Test output feature names
    df = fs.shot_relative_angle(shot, [True])
    assert df.columns.tolist() == ["relative_angle_shot"]
    assert len(df) == 1
    # Ball on goalline in center of goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 34]
    df = fs.shot_relative_angle(shot, [True])
    assert df.loc[0, "relative_angle_shot"] == 1
    # Ball on goalline next to goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 0]
    df = fs.shot_relative_angle(shot, [True])
    assert df.loc[0, "relative_angle_shot"] == 0
    shot.loc[:, ["start_x", "start_y"]] = [105, 68]
    df = fs.shot_relative_angle(shot, [True])
    assert df.loc[0, "relative_angle_shot"] == 0
    # Decreases as ball moves away from goal
    shot.loc[:, ["start_x", "start_y"]] = [100, 20]
    df1 = fs.shot_relative_angle(shot, [True])
    shot.loc[:, ["start_x", "start_y"]] = [100, 10]
    df2 = fs.shot_relative_angle(shot, [True])
    assert df1.loc[0, "relative_angle_shot"] > df2.loc[0, "relative_angle_shot"]
    # Is the same left and right from the goal
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34 - 5]
    df1 = fs.shot_relative_angle(shot, [True])
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34 + 5]
    df2 = fs.shot_relative_angle(shot, [True])
    assert df1.loc[0, "relative_angle_shot"] == pytest.approx(df2.loc[0, "relative_angle_shot"])


def test_shot_dist(shot):
    # Test output feature names
    df = fs.shot_dist(shot, [True])
    assert df.columns.tolist() == ["dist_shot"]
    assert len(df) == 1
    # Ball on goalline in center of goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 34]
    df = fs.shot_dist(shot, [True])
    assert df.loc[0, "dist_shot"] == 0
    # Ball on goalline next to goal
    shot.loc[:, ["start_x", "start_y"]] = [105, 0]
    df = fs.shot_dist(shot, [True])
    assert df.loc[0, "dist_shot"] == 34
    # Ball on penalty spot
    shot.loc[:, ["start_x", "start_y"]] = [105 - 11, 34]
    df = fs.shot_dist(shot, [True])
    assert df.loc[0, "dist_shot"] == 11


def test_shot_location(shot):
    # Test output feature names
    df = fs.shot_location(shot, [True])
    assert df.columns.tolist() == ["dx_shot", "dy_shot"]
    assert len(df) == 1
    # Test feature values
    assert df.loc[0, "dx_shot"] == 11
    assert df.loc[0, "dy_shot"] == 0


def test_shot_bodypart(shot):
    # Test output feature names
    df = fs.shot_bodypart(shot, [True])
    assert df.columns.tolist() == ["bodypart_name_shot"]
    assert len(df) == 1
    # Test feature values
    assert df.loc[0, "bodypart_name_shot"] == 'foot'


def test_shot_bodypart_onehot(shot):
    # Test output feature names
    df = fs.shot_bodypart_onehot(shot, [True])
    assert df.columns.tolist() == [
        "bodypart_foot_shot",
        "bodypart_head_shot",
        "bodypart_other_shot",
        "bodypart_head/other_shot",
    ]
    assert len(df) == 1
    # Test feature values
    shot.loc[:, ["bodpart_id", "bodypart_name"]] = [0, "foot"]
    df = fs.shot_bodypart_onehot(shot, [True])
    assert df.loc[0, "bodypart_foot_shot"]
    shot.loc[:, ["bodpart_id", "bodypart_name"]] = [1, "head"]
    df = fs.shot_bodypart_onehot(shot, [True])
    assert not df.loc[0, "bodypart_foot_shot"]
    assert df.loc[0, "bodypart_head_shot"]
    assert df.loc[0, "bodypart_head/other_shot"]


def test_extract_features_on_game(api):
    GAME_ID = 8658  # World Cup final
    game = api.games.loc[GAME_ID]
    events = api.get_events(GAME_ID)
    actions = api.get_actions(GAME_ID)
    features = fs.extract_features_on_game(game, actions, events)
    assert len(features) == 23
