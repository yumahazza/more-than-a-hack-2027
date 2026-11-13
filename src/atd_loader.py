"""Fast streaming ATD parser with local numpy caching.

Parses Metrica's FIFA/EPTS raw tracking file into video-frame-aligned NumPy
arrays (0-indexed: row i corresponds to submission frame i) without the
memory and runtime overhead of loading 199,319 frames through kloppy.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, DEFAULT_SOURCE, GAME, GAMES, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1


def load_team_mapping(meta_path):
    """Return int8 array of shape (n_slots,) mapping track slot -> HOME (0) or AWAY (1)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def _parse_triplet(token):
    """Parse 'x,y,s' token where missing values start with 'N' ('NaN')."""
    if token[0] == "N":
        return np.nan, np.nan, np.nan
    c1 = token.index(",")
    c2 = token.index(",", c1 + 1)
    x = float(token[:c1])
    y = float(token[c1 + 1:c2])
    spd_str = token[c2 + 1:]
    spd = np.nan if spd_str[0] == "N" else float(spd_str)
    return x, y, spd


def parse_atd_raw(raw_path, n_frames, n_slots):
    """Stream raw ATD file into float32 arrays aligned to 0-based video frames."""
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            vframe = int(parts[0]) - 1
            if vframe < 0 or vframe >= n_frames:
                continue

            bx_val, by_val, _ = _parse_triplet(parts[2])
            bx[vframe] = bx_val
            by[vframe] = by_val

            for slot, token in enumerate(parts[1].split(";")):
                if slot >= n_slots or token[0] == "N":
                    continue
                x, y, spd = _parse_triplet(token)
                px[vframe, slot] = x
                py[vframe, slot] = y
                ps[vframe, slot] = spd

    return px, py, ps, bx, by


def load_atd_arrays(game=GAME, data_dir=None, source=DEFAULT_SOURCE, use_cache=True):
    """Load video-aligned ATD arrays, caching as compressed npz under data/_cache/.

    Returns a dict with keys:
        player_x, player_y, player_s: float32 arrays of shape (n_frames, n_slots)
        ball_x, ball_y: float32 arrays of shape (n_frames,)
        team_of_slot: int8 array of shape (n_slots,) with 0=Home, 1=Away
        pitch: tuple (length_m, width_m)
        fps: int
        n_frames: int
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    team_of_slot = load_team_mapping(meta_path)
    n_slots = len(team_of_slot)

    cache_root = data_dir or DATA_DIR
    cache_dir = os.path.join(cache_root, "_cache")
    cache_path = os.path.join(cache_dir, f"atd_{game}_{source}.npz")

    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        return {
            "player_x": cached["player_x"],
            "player_y": cached["player_y"],
            "player_s": cached["player_s"],
            "ball_x": cached["ball_x"],
            "ball_y": cached["ball_y"],
            "team_of_slot": cached["team_of_slot"],
            "pitch": tuple(GAMES[game]["pitch"]),
            "fps": GAMES[game]["fps"],
            "n_frames": n_frames,
        }

    px, py, ps, bx, by = parse_atd_raw(raw_path, n_frames, n_slots)

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(
            cache_path,
            player_x=px,
            player_y=py,
            player_s=ps,
            ball_x=bx,
            ball_y=by,
            team_of_slot=team_of_slot,
        )

    return {
        "player_x": px,
        "player_y": py,
        "player_s": ps,
        "ball_x": bx,
        "ball_y": by,
        "team_of_slot": team_of_slot,
        "pitch": tuple(GAMES[game]["pitch"]),
        "fps": GAMES[game]["fps"],
        "n_frames": n_frames,
    }
