"""A hand-built 88-second scenario with a known answer, to test the pipeline logic WITHOUT the
match data (which never leaves our machines). Run:

    .\\.venv\\Scripts\\python.exe solution\\test_scenario.py

Story (frames at 25 fps):
    0-100     kick-off setup, players still            -> D
    100       home kicks off, home dribbles            -> H
    ~600      away player intercepts, away dribbles    -> A
    ~894      away clears the ball over the touchline  -> D (ball out, then lost, players slow)
    ~1250     home throw-in, home keeps the ball       -> H
    1600-1900 ball lost in play; tags say away from 1700 -> A from 1700
    1900-2200 away dribbles                            -> A
    2200-2600 injury: players walk, the ball rolls slowly, no tag at all -> D
    2600      home restarts from a still ball, dribbles -> H
    ~2860     home dribbles over the goal line         -> D
    3100-3150 the ball is CARRIED back at a jog (not a restart), placed, then
    3200      away goal kick                           -> A
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import params as P  # noqa: E402
import tags as tagging  # noqa: E402
from deadball import detect, runs  # noqa: E402
from features import compute  # noqa: E402
from possession import assign  # noqa: E402

L, W = 105.3, 68.0
N = 3400
rng = np.random.default_rng(0)


def unit(v):
    norm = np.hypot(*v)
    return np.array(v) / norm if norm else np.zeros(2)


def build():
    ball = np.full((N, 2), np.nan)
    pos = np.array([L / 2, W / 2])
    segments = [(0, 100, (0, 0)), (100, 350, (3.5, 0)), (350, 600, (-3.5, 0.5)),
                (600, 860, (3.5, -1)), (860, 900, (0, -20)), (900, 925, (0, -1)),
                (1250, 1266, (0, 8)), (1266, 1600, (-3, 0.5)), (1900, 2200, (2, 0))]
    vel = np.zeros((N, 2))
    for s, e, v in segments:
        if s == 1250:
            pos = np.array([pos[0], 1.0])
        for t in range(s, e):
            ball[t] = pos
            vel[t] = v
            pos = pos + np.array(v) / P.FPS
        if s == 1266:
            last_live = pos.copy()
    ball[1900:2200] = np.array([50.0, 20.0]) + np.outer(np.arange(300), [2 / P.FPS, 0])

    players = np.full((N, 46, 2), np.nan)
    team = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    walk_speed = np.full(N, 2.5)
    walk_speed[:100] = 0.3
    walk_speed[925:1250] = 0.3
    walk_speed[2200:2600] = 0.8      # injury: everybody walks
    walk_speed[2870:3200] = 2.0      # jogging into position for the goal kick
    walkers = list(range(1, 11)) + list(range(24, 34))
    for k in walkers:
        p = rng.uniform([10, 5], [95, 63])
        angle = rng.uniform(0, 2 * np.pi)
        for t in range(N):
            angle += rng.normal(0, 0.05)
            speed = walk_speed[t]
            p = p + speed / P.FPS * np.array([np.cos(angle), np.sin(angle)])
            if not (2 < p[0] < L - 2):
                angle = np.pi - angle
            if not (2 < p[1] < W - 2):
                angle = -angle
            p = np.clip(p, 1, [L - 1, W - 1])
            players[t, k] = p

    H, A = 0, 23
    for t in range(N):
        b, v = ball[t], vel[t]
        if t < 100:
            players[t, H] = b - [0.5, 0]
            players[t, A] = b + [8, 0]
        elif t < 600:
            players[t, H] = b - 0.5 * unit(v)
            off = 4.0 if t < 560 else 4.0 - 3.7 * (t - 560) / 40
            players[t, A] = b + [0, off]
        elif t < 900:
            players[t, A] = b - 0.3 * unit(v)
            players[t, H] = b + [0, min(3.0, 0.5 + 2.5 * (t - 600) / 20)]
        elif t < 1250:
            a = (t - 900) / 350
            players[t, H] = (1 - a) * players[899, H] + a * np.array([94.65, 6.0])
            players[t, A] = players[899, A] + [0, 8 * a]
        elif t < 1600:
            if t < 1266:
                players[t, H] = [94.65, 6.0]
            else:
                players[t, H] = b - 0.5 * unit(v)
            players[t, A] = (b if np.isfinite(b).all() else players[t - 1, H]) + [0, 5]
        elif t < 1900:
            players[t, H] = players[t - 1, H] + [-2.5 / P.FPS, 0]
            players[t, A] = players[t - 1, A] + [-2.5 / P.FPS, 0]
        else:
            players[t, A] = b - 0.4 * unit(v)
            players[t, H] = b + [0, 5]
    ball[925:1250] = np.nan      # tracker loses the ball during the stoppage
    ball[1600:1900] = np.nan     # ... and in play

    # -- injury, restart, ball over the goal line, carried back, goal kick ----------------------
    centre = ball[2199] - [3.0, 0.0]
    for t in range(2200, 2575):                     # ball rolls slowly (1.5 m/s) on a circle
        theta = 0.5 * (t - 2200) / P.FPS
        ball[t] = centre + 3.0 * np.array([np.cos(theta), np.sin(theta)])
    ball[2575:2600] = ball[2574]                    # placed for the restart
    pos = ball[2599].copy()
    for t in range(2600, 2870):                     # kick, then dribble towards the goal line
        v = (9.0, 0.0) if t < 2610 else (3.0, 0.0)
        ball[t] = pos
        vel[t] = v
        pos = pos + np.array(v) / P.FPS
    ball[2870:3100] = np.nan                        # lost behind the goal
    pos = np.array([104.0, 34.0])
    for t in range(3100, 3150):                     # carried back at a jog: NOT a restart
        ball[t] = pos
        vel[t] = (-3.5, 0.0)
        pos = pos + np.array([-3.5, 0.0]) / P.FPS
    ball[3150:3200] = pos                           # placed in the six-yard box
    for t in range(3200, N):                        # goal kick, then away dribbles
        v = (-15.0, 0.0) if t < 3225 else (-3.0, 0.0)
        ball[t] = pos
        vel[t] = v
        pos = pos + np.array(v) / P.FPS

    for t in range(2200, N):
        b, v = ball[t], vel[t]
        if t < 2575:
            players[t, H] = players[t - 1, H] + [1.0 / P.FPS, 0] if t > 2200 else ball[2199] + [0, 6]
            players[t, A] = players[t - 1, A] + [0, 1.0 / P.FPS] if t > 2200 else ball[2199] + [0, -6]
        elif t < 2600:
            players[t, H] = b - [0.5, 0]
            players[t, A] = b + [0, 6]
        elif t < 2870:
            players[t, H] = b - 0.5 * unit(v)
            players[t, A] = b + [0, 5]
        elif t < 3100:
            players[t, H] = players[t - 1, H] + [-1.5 / P.FPS, 0]
            players[t, A] = players[t - 1, A] + ((np.array([104.5, 34.0]) - players[t - 1, A]) / max(3100 - t, 1))
        elif t < 3200:
            players[t, A] = (b + [0.5, 0]) if t < 3150 else (b + [0.6, 0])
            players[t, H] = players[t - 1, H]
        else:
            players[t, A] = b - 0.4 * unit(v)
            players[t, H] = players[t - 1, H] + [-2.0 / P.FPS, 0]

    arrays = {"ball": np.column_stack([ball, np.full(N, np.nan)]).astype(np.float32),
              "players": np.concatenate([players, np.full((N, 46, 1), np.nan)], axis=2).astype(np.float32),
              "team": team}
    events = [
        {"code": "SET PIECES", "start_frame": 0, "end_frame": 125,
         "tags": {"Team": "Barcelona", "Type": "Kick Off (Start)"}},
        {"code": "BUILD UP", "start_frame": 1300, "end_frame": 1650, "tags": {"Team": "Barcelona"}},
        {"code": "SET PIECES", "start_frame": 1230, "end_frame": 1280,
         "tags": {"Team": "N/A", "Type": "Throw-in"}},
        {"code": "PROGRESSION", "start_frame": 1700, "end_frame": 1950,
         "tags": {"Team": "Atletic Club"}},
        {"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 1700, "end_frame": 1950,
         "tags": {"Team": "Barcelona"}},
        {"code": "SET PIECES", "start_frame": 3150, "end_frame": 3230,
         "tags": {"Team": "Atletic Club", "Type": "Goal Kick"}},
    ]
    return arrays, events


def describe(labels):
    out = []
    for code, name in ((0, "H"), (1, "A"), (2, "D")):
        for s, l in zip(*runs(labels == code)):
            out.append((int(s), int(s + l), name))
    return sorted(out)


def main():
    arrays, events = build()
    halves = [(1, 0, N - 1)]
    tag_data = tagging.build(N, events)
    f = compute(arrays, L, W)
    dead = detect(f, tag_data, halves)
    labels = assign(f, dead, tag_data, halves)
    got = describe(labels)
    print("result:")
    for s, e, name in got:
        print("  {}  frame {:>5} - {:>5}".format(name, s, e))

    expected = [("D", 0, 100), ("H", 100, 600), ("A", 600, 894), ("D", 894, 1252),
                ("H", 1252, 1700), ("A", 1700, 2200), ("D", 2200, 2600), ("H", 2600, 2860),
                ("D", 2860, 3200), ("A", 3200, 3400)]
    tolerance = P.FPS   # 1 s; a stoppage is only seen once the 1 s activity smoothing catches it
    ok = len(got) == len(expected) and all(
        g[2] == x[0] and abs(g[0] - x[1]) <= tolerance and abs(g[1] - x[2]) <= tolerance
        for g, x in zip(got, expected))
    print("\n{}  (expected: {})".format("PASS" if ok else "FAIL",
                                      ", ".join("{} {}-{}".format(*x) for x in expected)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
