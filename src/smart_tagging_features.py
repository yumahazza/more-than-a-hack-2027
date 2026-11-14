"""Smart Tagging tactical feature extraction aligned to tactical video frames.

Converts Metrica's SportsCode XML intervals into frame-level soft priors and
structured set-piece / kickoff anchors, explicitly handling 'N/A' teams and
the polarity difference between attacking and defending phase tags.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES, load as load_xml  # noqa: E402
from src.atd_loader import AWAY, HOME  # noqa: E402

NOBODY = -1

# Phase codes where the tagged Team is attacking / in possession
ATTACK_CODE_WEIGHTS = {
    "BUILD UP": 1.0,
    "PROGRESSION": 1.0,
    "ATTACKING TRANSITION": 1.0,
    "CREATING CHANCES": 0.8,
    "BALL IN FINAL THIRD": 0.6,
    "BALL IN THE BOX": 0.6,
    "PLAYERS IN THE BOX": 0.5,
    "LONG BALL": 0.6,
}

# Phase codes where the tagged Team is defending (so the opponent likely has the ball)
DEFEND_CODE_WEIGHTS = {
    "DEFENDING IN ATTACKING THIRD": 0.8,
    "DEFENDING IN MIDDLE THIRD": 0.8,
    "DEFENDING IN DEFENSIVE THIRD": 0.8,
    "DEFENSIVE TRANSITION": 0.8,
}


def parse_team_name(team_text):
    """Map Smart Tagging 'Team' text to HOME (0), AWAY (1), or NOBODY (-1)."""
    if not team_text:
        return NOBODY
    norm = team_text.strip().lower()
    if norm in ("n/a", "none", ""):
        return NOBODY
    if "barca" in norm or "barcel" in norm:
        return HOME
    if "atlet" in norm or "athlet" in norm:
        return AWAY
    return NOBODY


def extract_smart_tagging_features(n_frames, game=GAME, data_dir=None):
    """Build frame-aligned Smart Tagging signals and structured restart anchors.

    Returns a dict with:
        home_prior: float64 (n_frames,) soft attacking prior for Home
        away_prior: float64 (n_frames,) soft attacking prior for Away
        st_bias: float64 (n_frames,) home_prior - away_prior
        in_set_piece: bool (n_frames,) True inside any SET PIECES window
        set_piece_team: int8 (n_frames,) expected restart team (-1 if unknown/N/A)
        set_pieces: list of structured set-piece dicts
        active_half_mask: bool (n_frames,) True inside graded halves
        pre_kickoff_dead_mask: bool (n_frames,) True inside graded half before kickoff
    """
    fps = GAMES[game]["fps"]
    events = load_xml(game=game, data_dir=data_dir, fps=fps)

    home_prior = np.zeros(n_frames, dtype=np.float64)
    away_prior = np.zeros(n_frames, dtype=np.float64)
    in_set_piece = np.zeros(n_frames, dtype=bool)
    set_piece_team = np.full(n_frames, NOBODY, dtype=np.int8)

    set_pieces = []
    goals = []

    for ev in events:
        code = (ev["code"] or "").strip().upper()
        s = max(0, int(ev["start_frame"]))
        e = min(n_frames, int(ev["end_frame"]))
        if e <= s:
            continue

        team_code = parse_team_name(ev["tags"].get("Team"))

        if code in ATTACK_CODE_WEIGHTS and team_code != NOBODY:
            w = ATTACK_CODE_WEIGHTS[code]
            if team_code == HOME:
                home_prior[s:e] += w
            else:
                away_prior[s:e] += w

        elif code in DEFEND_CODE_WEIGHTS and team_code != NOBODY:
            w = DEFEND_CODE_WEIGHTS[code]
            opp = AWAY if team_code == HOME else HOME
            if opp == HOME:
                home_prior[s:e] += w
            else:
                away_prior[s:e] += w

        elif code == "GOALS":
            goals.append({"start_frame": s, "end_frame": e, "team_code": team_code})

        elif code == "SET PIECES":
            sp_type = (ev["tags"].get("Type") or "").strip()
            set_pieces.append({
                "start_frame": s,
                "end_frame": e,
                "type": sp_type,
                "team_code": team_code,
                "half": ev["tags"].get("Half"),
            })

    # If a Kick Off (After Goal) follows a GOALS interval, the conceding team restarts play
    for sp in set_pieces:
        if sp["type"].lower() == "kick off (after goal)" and sp["team_code"] == NOBODY:
            preceding = [g for g in goals if g["start_frame"] <= sp["start_frame"] <= g["end_frame"] + 50]
            if preceding and preceding[-1]["team_code"] != NOBODY:
                scorer = preceding[-1]["team_code"]
                sp["team_code"] = AWAY if scorer == HOME else HOME

        s, e = sp["start_frame"], sp["end_frame"]
        in_set_piece[s:e] = True
        if sp["team_code"] != NOBODY:
            set_piece_team[s:e] = sp["team_code"]

    # Graded match halves and pre-kickoff dead-ball stretches
    active_half_mask = np.zeros(n_frames, dtype=bool)
    pre_kickoff_dead_mask = np.zeros(n_frames, dtype=bool)
    blocks = half_video_blocks(game)
    half_names = {1: "1st Half", 2: "2nd Half"}

    for half_num, first_f, last_f in blocks:
        first_f = max(0, first_f)
        last_f = min(n_frames - 1, last_f)
        active_half_mask[first_f:last_f + 1] = True

        ko_frame = KICKOFF_VIDEO_FRAMES.get(half_names.get(half_num))
        if ko_frame is not None and ko_frame > first_f:
            pre_kickoff_dead_mask[first_f:min(ko_frame, last_f + 1)] = True

    return {
        "home_prior": home_prior,
        "away_prior": away_prior,
        "st_bias": home_prior - away_prior,
        "in_set_piece": in_set_piece,
        "set_piece_team": set_piece_team,
        "set_pieces": set_pieces,
        "active_half_mask": active_half_mask,
        "pre_kickoff_dead_mask": pre_kickoff_dead_mask,
    }
