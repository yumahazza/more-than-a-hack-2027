"""Smart Tagging -> per-frame arrays the solution can use."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))
import load_smart_tagging  # noqa: E402

import params as P  # noqa: E402

HOME, AWAY = 0, 1
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


def side_of(value):
    v = (value or "").strip().lower()
    if "barcelona" in v:
        return HOME
    if "atletic" in v or "athletic" in v:
        return AWAY
    return None


def build(n, events=None):
    """Return dict:
    tag_team  (n,) int8   0/1 = team the reliable tags say has the ball, -1 = no clear answer
    setpieces list of {start, end, type, side}   side None for N/A (e.g. every throw-in)
    sp_start  (n,) bool   frame where a set-piece tag starts
    """
    events = load_smart_tagging.load() if events is None else events
    vote = np.zeros((2, n), dtype=np.float32)
    setpieces = []
    sp_start = np.zeros(n, dtype=bool)
    for e in events:
        code = (e["code"] or "").strip().lower()
        s, t = max(int(e["start_frame"]), 0), min(int(e["end_frame"]), n)
        if t <= s:
            continue
        side = side_of(e["tags"].get("Team"))
        if code == "set pieces":
            setpieces.append({"start": s, "end": t, "side": side,
                              "type": (e["tags"].get("Type") or "").strip().lower()})
            sp_start[s] = True
            continue
        w = P.TAG_WEIGHTS.get(code)
        if w is None or side is None:
            continue
        if code in DEFENDING:
            side = 1 - side   # the tagged team is defending, i.e. does NOT have the ball
        vote[side, s:t] += w

    lead = vote[HOME] - vote[AWAY]
    tag_team = np.full(n, -1, dtype=np.int8)
    tag_team[lead >= P.TAG_MARGIN] = HOME
    tag_team[lead <= -P.TAG_MARGIN] = AWAY
    return {"tag_team": tag_team, "setpieces": setpieces, "sp_start": sp_start}
