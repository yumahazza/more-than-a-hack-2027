"""External validation on a DIFFERENT, public match: Metrica Sports Sample Game 3.

    .\\.venv\\Scripts\\python.exe solution\\external_validation.py <folder with Sample_Game_3_*>

Data: https://github.com/metrica-sports/sample-data (data/Sample_Game_3): EPTS tracking (same
format as the competition ATD, players x,y only), metadata XML and Metrica event JSON. RULES.md
allows labelled data from other games; this match is not the competition match.

Why: the public window has 17 possessions, so v9 vs v12 cannot be separated beyond noise there.
This match has several hundred possessions. Labels are built from its human event data, so their
definition differs from the competition ground truth (corrected tracking) and absolute scores are
not comparable; the comparison BETWEEN versions is what this is for. Nothing here is used to set a
value: v9 and v12 are run exactly as they are.

Labels from events (in time order):
  restart (SET PIECE, or a goal-kick pass) -> live, team = its team
  on-ball event (PASS, CARRY, RECOVERY, SHOT) while live -> team = its team
  BALL OUT, a shot that goes out or in, a foul -> dead from that moment until the next restart
  CHALLENGE and BALL LOST do not change the team (the regaining team's own event follows)
  before each half's kick-off -> dead
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import params as P  # noqa: E402
import possession_metrics as pm  # noqa: E402
import tags as tagging  # noqa: E402
from compare import macro_score, segment_contrib, spell_flags, spell_score  # noqa: E402
from deadball import detect  # noqa: E402
from features import compute  # noqa: E402
from possession import assign  # noqa: E402

HOME, AWAY, DEAD, NONE = 0, 1, 2, -1


def names(sub):
    if sub is None:
        return []
    if isinstance(sub, dict):
        return [sub.get("name", "")]
    return [s.get("name", "") for s in sub]


def load_tracking(folder):
    meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    fs = next(meta.iter("FieldSize"))
    length, width = float(fs.findtext("Width")), float(fs.findtext("Height"))
    team_ids = [t.get("id") for t in meta.iter("Team")]
    player_team = {p.get("id"): (HOME if p.get("teamId") == team_ids[0] else AWAY) for p in meta.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    slot_of = {pid: i for i, pid in enumerate(player_team)}
    team = np.array([player_team[pid] for pid in slot_of], dtype=np.int8)
    specs = []
    for s in meta.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in s.iter("PlayerChannelRef")]
        order = [slot_of[channel_player[r]] for r in refs[0::2]]       # x channel of each player
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), order))

    path = os.path.join(folder, "Sample_Game_3_tracking.txt")
    n = max(e for _, e, _ in specs)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    players = np.full((n, len(team), 3), np.nan, dtype=np.float32)
    with open(path) as handle:
        for line in handle:
            head, body, b = line.rstrip("\n").split(":")
            fid = int(head)
            order = next(o for s, e, o in specs if s <= fid <= e)
            vals = np.array(body.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 2)
            players[fid - 1, order, 0] = vals[:, 0] * length
            players[fid - 1, order, 1] = vals[:, 1] * width
            bxy = b.split(",")
            ball[fid - 1, 0] = float(bxy[0]) * length
            ball[fid - 1, 1] = float(bxy[1]) * width
    return {"ball": ball, "players": players, "team": team}, length, width, team_ids


def build_labels(folder, n, team_ids):
    data = json.load(open(os.path.join(folder, "Sample_Game_3_events.json"), encoding="utf-8"))["data"]
    data.sort(key=lambda e: (e["start"]["frame"], e["index"]))
    side = lambda e: HOME if e["team"]["id"] == team_ids[0] else AWAY  # noqa: E731
    halves = []
    for period in (1, 2):
        ev = [e for e in data if e["period"] == period]
        ko = [e for e in ev if e["type"]["name"] == "SET PIECE" and "KICK OFF" in names(e.get("subtypes"))]
        start = (ko[0] if ko else ev[0])["start"]["frame"] - 1
        end = max(e["end"]["frame"] or e["start"]["frame"] for e in ev) - 1
        halves.append((period, start, end))

    gt = np.full(n, NONE, dtype=np.int8)
    for period, first, last in halves:
        ev = [e for e in data if e["period"] == period]
        state, mark = DEAD, first
        for e in ev:
            t = e["start"]["frame"] - 1
            kind, subs = e["type"]["name"], names(e.get("subtypes"))
            restart = kind == "SET PIECE" or (kind == "PASS" and "GOAL KICK" in subs)
            onball = kind in ("PASS", "CARRY", "RECOVERY", "SHOT")
            new = state
            when = t
            # Once the ball is dead only a restart revives it: Metrica logs the last touch (a CARRY)
            # on the same frame as BALL OUT, which must not count as play. A dead stretch with no
            # logged restart for 90 s is taken as resumed by the next on-ball event.
            stale = state == DEAD and t - mark > 90 * P.FPS
            if restart or (onball and (state != DEAD or stale)):
                new = side(e)
            if kind == "BALL OUT":
                new, when = DEAD, (e["end"]["frame"] or e["start"]["frame"]) - 1
            elif kind == "SHOT" and ("OUT" in subs or "GOAL" in subs):
                new, when = DEAD, (e["end"]["frame"] or e["start"]["frame"]) - 1
            elif kind == "FAULT RECEIVED" or (kind == "CHALLENGE" and "FAULT" in subs):
                new = DEAD
            when = min(max(when, mark), last + 1)
            if new != state:
                gt[mark:when] = state
                mark, state = when, new
        gt[mark:last + 1] = state
    return gt, halves


LETTER = {HOME: "H", AWAY: "A", DEAD: "D", NONE: "X"}


def to_codes(labels):
    return pm.label_codes(np.array([LETTER[int(c)] for c in labels]))


def run(arrays, length, width, halves, flags):
    saved = {k: getattr(P, k) for k in flags}
    for k, v in flags.items():
        setattr(P, k, v)
    try:
        tag_data = tagging.build(len(arrays["ball"]), events=[])
        f = compute(arrays, length, width, halves)
        dead = detect(f, tag_data, halves)
        return assign(f, dead, tag_data, halves), f
    finally:
        for k, v in saved.items():
            setattr(P, k, v)


def summary(name, gt_codes, pred_codes, n_frames_graded):
    r = pm.evaluate(gt_codes, pred_codes, bootstrap_n=0)
    print("  {:<4} spellF1 {:.4f}  macroF1 {:.4f}  team acc {:.1%}  alive/dead F1 {:.3f}  possessions {:>4} "
          "(truth {})  mean {:.1f} s (truth {:.1f} s)".format(
              name, r["spell_f1_mean"], r["macro_f1"], r["team_acc_gt_alive"], r["alive_dead_f1"],
              r["n_pred_spells"], r["n_gt_spells"], r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"]))
    return r


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    folder = sys.argv[1]
    arrays, length, width, team_ids = load_tracking(folder)
    n = len(arrays["ball"])
    gt, halves = build_labels(folder, n, team_ids)
    graded = gt != NONE
    vis = np.isfinite(arrays["ball"][:, 0])
    print("Metrica Sample Game 3: {} frames, halves {}, pitch {} x {} m".format(n, halves, length, width))
    print("event labels on graded frames: H {:.1%}  A {:.1%}  D {:.1%};  ball visible {:.1%}".format(
        np.mean(gt[graded] == HOME), np.mean(gt[graded] == AWAY), np.mean(gt[graded] == DEAD), np.mean(vis[graded])))

    v9_flags = {"USE_GK_FIX": False, "USE_OOB_GAP": False, "USE_TOUCH_GATE": False}
    v12_flags = {"USE_GK_FIX": True, "USE_OOB_GAP": True, "USE_TOUCH_GATE": True}
    gt_codes = to_codes(gt)
    preds = {}
    for name, flags in (("v9", v9_flags), ("v12", v12_flags)):
        labels, f = run(arrays, length, width, halves, flags)
        preds[name] = to_codes(labels)
        summary(name, gt_codes, preds[name], int(graded.sum()))
    print("  v12 extras: goalkeeper slots {}, slot-frames reassigned {}, extra out-of-play events {}".format(
        len(f["gk"]["gk_slots"]), f["gk"]["reassigned_frames"], f.get("oob_gap_events", 0)))

    runs = pm.valid_runs(gt_codes)
    fa, n_gt, npa = spell_flags(gt_codes, preds["v9"], runs)
    fb, _, npb = spell_flags(gt_codes, preds["v12"], runs)
    codes_seg, ca = segment_contrib(gt_codes, preds["v9"], runs)
    _, cb = segment_contrib(gt_codes, preds["v12"], runs)
    rng = np.random.RandomState(2027)
    d_spell, d_macro = [], []
    for _ in range(2000):
        p = rng.randint(0, n_gt, n_gt)
        d_spell.append(spell_score(fb, n_gt, npb, p) - spell_score(fa, n_gt, npa, p))
        q = rng.randint(0, len(codes_seg), len(codes_seg))
        d_macro.append(macro_score(codes_seg, cb, q) - macro_score(codes_seg, ca, q))
    for name, d in (("spellF1", np.array(d_spell)), ("macroF1", np.array(d_macro))):
        lo, hi = np.percentile(d, [5, 95])
        print("  paired bootstrap {} v12 - v9: 5-95% {:+.4f} .. {:+.4f}   P(diff <= 0) = {:.1%}  ({} possessions)".format(
            name, lo, hi, np.mean(d <= 0), n_gt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
