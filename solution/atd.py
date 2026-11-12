"""Load the tactical ATD into numpy arrays, once, and cache them.

Every later step of our solution needs the same thing: for each video frame, where the ball is and
where every tracked player is. Parsing the 195 MB text file takes a while, so the result is saved
to data/cache/ (gitignored, because it is derived match data) and reused on the next run.

    from atd import load_arrays
    a = load_arrays()
    a["ball"]      # (n_frames, 3)      x, y in metres, speed; NaN where the ball is missing
    a["players"]   # (n_frames, 46, 3)  x, y in metres, speed per track slot; NaN where absent
    a["team"]      # (46,)              0 = home (H), 1 = away (A), per track slot

Row i is video frame i, i.e. submission row i (ATD frame_id = i + 1).
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1
CACHE_VERSION = 1


def team_of_slots(meta_path):
    """Team per track slot from the ATD metadata. The first team listed is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_groups(text, n_groups):
    """'x,y,s;x,y,s;...' -> (n_groups, 3) float32, NaN-padded."""
    out = np.full((n_groups, 3), np.nan, dtype=np.float32)
    groups = text.split(";")
    flat = text.replace(";", ",").split(",")
    if len(flat) == 3 * len(groups):  # the common case: every group has exactly 3 fields
        values = np.array(flat, dtype=np.float32).reshape(-1, 3)
        k = min(n_groups, len(values))
        out[:k] = values[:k]
        return out
    for i, group in enumerate(groups[:n_groups]):  # fallback for irregular lines
        fields = group.split(",")[:3]
        out[i, :len(fields)] = [float(f) for f in fields]
    return out


def parse(meta_path, raw_path, n_frames, pitch):
    team = team_of_slots(meta_path)
    n_slots = len(team)
    length, width = pitch
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= row < n_frames:
                continue
            players[row] = _parse_groups(parts[1], n_slots)
            ball[row] = _parse_groups(parts[2], 1)[0]

    # normalised 0-1 -> metres, so distances mean the same thing along both axes
    for arr in (ball[..., 0], players[..., 0]):
        arr *= length
    for arr in (ball[..., 1], players[..., 1]):
        arr *= width
    return {"ball": ball, "players": players, "team": team}


def load_arrays(game=GAME, data_dir=None, use_cache=True, verbose=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache_dir = os.path.join(data_dir or DATA_DIR, "cache")
    cache_path = os.path.join(cache_dir, "atd_tactical_v{}.npz".format(CACHE_VERSION))

    if use_cache and os.path.isfile(cache_path):
        with np.load(cache_path) as z:
            return {k: z[k] for k in ("ball", "players", "team")}

    if verbose:
        print("membaca {} (sekali saja, lalu disimpan ke cache) ...".format(
            os.path.basename(raw_path)))
    arrays = parse(meta_path, raw_path, source_frames("tactical", game), GAMES[game]["pitch"])
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, **arrays)
        if verbose:
            print("cache disimpan: {}".format(os.path.relpath(cache_path, ROOT)))
    return arrays
