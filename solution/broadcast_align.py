"""Map the broadcast recording onto the tactical timeline, automatically, from the sound.

RESULT: NOT USED. No feature located the broadcast: confident chunks 1/218 (loudness), 0/218
(loudness 2-4.5 kHz), 0/214 (whistle-band events); the best shift beat every other shift by only
~3-4% (median lead 1.03-1.04). The two sound tracks do not share enough structure (the broadcast is
a TV mix dominated by commentary). Kept as the record of the attempt; nothing feeds the pipeline.

    .\\.venv\\Scripts\\python.exe solution\\broadcast_align.py            # decode both (minutes, once)
    .\\.venv\\Scripts\\python.exe solution\\broadcast_align.py --cached   # reuse the saved envelopes

Why: the broadcast feed cuts to replays, close-ups and crowd shots, and a TV director does that
almost only while the ball is dead (after a foul, an injury, a goal). Its homography is invalid on
exactly those shots (DATA.md: 65.8% valid, 122 stretches). That is a dead-ball cue we do not have.
But the broadcast has its own timeline and no mapping is provided (FAQ), so first: where does each
broadcast moment sit on the tactical video?

How (no hand anchors, RULES.md #1): both recordings hear the same stadium. Crowd swells, whistles
and silences give both sound tracks the same loudness pattern. TV replays keep the LIVE sound under
the old pictures, so the sound clock of the broadcast stays continuous even through replays, and
the mapping should be one constant shift per stretch between real cuts (e.g. half-time).
  1. Loudness per 40 ms block (one video frame) for both tracks, minus its own 5 s running mean
     (removes the different mixing levels), standardised.
  2. Every 60 s of broadcast (every 30 s) is slid along the whole tactical envelope; the shift with
     the highest normalised correlation is its position. Confidence = that peak and how far it
     stands above the next best shift.
  3. Consecutive confident chunks with the same shift (+/- 2 frames) form a segment.

Only aggregates are printed: the segments (broadcast span, shift, confidence) and how the invalid-
homography stretches fall on our own v14 labels. No ground truth is read.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, half_video_blocks, source_frames, video_path  # noqa: E402

import params as P  # noqa: E402
from deadball import runs  # noqa: E402
from whistle import HOP, OUT as WHISTLE, decode, narrow, pick_pitch, spectrum  # noqa: E402

ENV = os.path.join(DATA_DIR, "cache", "{}_audio_env.npz".format(GAME))
MAP = os.path.join(DATA_DIR, "cache", "{}_broadcast_map.npz".format(GAME))
CHUNK = 60 * P.FPS          # 60 s of broadcast per position estimate
STEP = 30 * P.FPS
DETREND = 5 * P.FPS         # running mean removed from the loudness
SAME = 2                    # frames: two chunks with shifts this close belong to one segment
MIN_PEAK = 0.3              # normalised correlation a chunk needs to count (reported, not tuned)
MIN_LEAD = 1.3              # its peak over the best shift more than 1 s away


def envelope(audio, n):
    blocks = min(n, len(audio) // HOP)
    e = audio[:blocks * HOP].reshape(blocks, HOP)
    db = 10 * np.log10(np.mean(e.astype(np.float64) ** 2, axis=1) + 1e-10)
    c = np.concatenate(([0.0], np.cumsum(db)))
    i = np.arange(blocks)
    lo, hi = np.maximum(i - DETREND // 2, 0), np.minimum(i + DETREND // 2 + 1, blocks)
    x = db - (c[hi] - c[lo]) / (hi - lo)
    x = (x - x.mean()) / (x.std() + 1e-9)
    out = np.zeros(n, np.float32)
    out[:blocks] = x
    return out


def locate(chunk, tac, tac_fft, nfft, energy):
    """Best start of `chunk` in `tac`: (start, peak NCC, lead over the next best >1 s away)."""
    m = len(chunk)
    c = chunk - chunk.mean()
    norm = np.sqrt(np.sum(c ** 2)) + 1e-9
    corr = np.fft.irfft(tac_fft * np.conj(np.fft.rfft(c, nfft)), nfft)[:len(tac) - m + 1]
    ncc = corr / (norm * np.sqrt(np.maximum(energy, 1e-9)))
    k = int(np.argmax(ncc))
    peak = float(ncc[k])
    masked = ncc.copy()
    masked[max(k - P.FPS, 0):k + P.FPS + 1] = -np.inf
    second = float(masked.max())
    return k, peak, peak / max(second, 1e-6)


def detrend(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=np.nanmedian(x))
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo, hi = np.maximum(i - DETREND // 2, 0), np.minimum(i + DETREND // 2 + 1, len(x))
    y = x - (c[hi] - c[lo]) / (hi - lo)
    return ((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32)


def whistle_events(bins, k):
    """Sparse whistle-band events: tonality above that recording's own median + 3 robust sd,
    smoothed over 5 frames so a whistle heard 1-2 frames apart still overlaps."""
    score = narrow(bins, k).astype(np.float64)
    g = score[np.isfinite(score)]
    med = np.median(g)
    thr = med + 3 * 1.4826 * np.median(np.abs(g - med))
    ev = np.clip(np.nan_to_num(score, nan=thr) - thr, 0, None)
    ev = np.convolve(ev, np.ones(5) / 5, mode="same")
    return ((ev - ev.mean()) / (ev.std() + 1e-9)).astype(np.float32)


def align(tac, bc, chunk, step):
    tac64 = tac.astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(len(tac64) + chunk)))
    tac_fft = np.fft.rfft(tac64, nfft)
    c2 = np.concatenate(([0.0], np.cumsum(tac64 ** 2)))
    c1 = np.concatenate(([0.0], np.cumsum(tac64)))
    starts = np.arange(len(tac64) - chunk + 1)
    s1 = c1[starts + chunk] - c1[starts]
    energy = (c2[starts + chunk] - c2[starts]) - s1 ** 2 / chunk   # centred window energy
    rows = []
    for b in range(0, len(bc) - chunk + 1, step):
        k, peak, lead = locate(bc[b:b + chunk].astype(np.float64), tac64, tac_fft, nfft, energy)
        rows.append((b, k - b, peak, lead))
    return np.array(rows)


def segments(rows, chunk, step):
    ok = (rows[:, 2] >= MIN_PEAK) & (rows[:, 3] >= MIN_LEAD)
    segs = []
    for b, shift, peak, lead in rows[ok]:
        if segs and abs(shift - segs[-1]["shift"]) <= SAME and b - segs[-1]["b1"] <= 2 * step:
            segs[-1]["b1"] = b
            segs[-1]["peaks"].append(peak)
        else:
            segs.append({"b0": b, "b1": b, "shift": shift, "peaks": [peak], "chunk": chunk})
    return ok, segs


def main():
    n_t = source_frames("tactical", GAME)
    n_b = source_frames("broadcast", GAME)
    graded = np.zeros(n_t, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    cached = "--cached" in sys.argv
    if cached and os.path.exists(ENV) and "b_bins" in np.load(ENV).files:
        d = np.load(ENV)
        tac, bc, b_bins, b_level = d["tactical"], d["broadcast"], d["b_bins"], d["b_level"]
    else:
        if cached and os.path.exists(ENV):
            d = np.load(ENV)
            tac, bc = d["tactical"], d["broadcast"]
        else:
            print("decoding the tactical sound track ...", flush=True)
            tac = envelope(decode(video_path(GAME, source="tactical")), n_t)
            bc = None
        print("decoding the broadcast sound track (a few minutes) ...", flush=True)
        audio_b = decode(video_path(GAME, source="broadcast"))
        if bc is None:
            bc = envelope(audio_b, n_b)
        b_bins, _, b_level = spectrum(audio_b, n_b)
        del audio_b
        os.makedirs(os.path.dirname(ENV), exist_ok=True)
        np.savez_compressed(ENV, tactical=tac, broadcast=bc, b_bins=b_bins, b_level=b_level)
    if not os.path.exists(WHISTLE):
        sys.exit("run solution/whistle.py first (it saves the tactical spectrum)")
    w = np.load(WHISTLE)
    t_bins, t_level = w["bins"], w["level"]
    k = pick_pitch(t_bins, graded)[0]

    features = {
        "loudness, all frequencies": (tac, bc, CHUNK, STEP),
        "loudness, 2-4.5 kHz": (detrend(t_level), detrend(b_level), CHUNK, STEP),
        "whistle-band events": (whistle_events(t_bins, k), whistle_events(b_bins, k), 3 * CHUNK, STEP),
    }
    best = None
    for name, (ft, fb, chunk, step) in features.items():
        rows = align(ft, fb, chunk, step)
        ok, segs = segments(rows, chunk, step)
        print("\n[{}] chunks of {} s: {} of {} confident ({:.0%})".format(
            name, chunk // P.FPS, ok.sum(), len(rows), ok.mean()))
        print("  peak NCC p10 {:.2f} p50 {:.2f} p90 {:.2f};  lead p10 {:.2f} p50 {:.2f} p90 {:.2f}".format(
            *np.percentile(rows[:, 2], [10, 50, 90]), *np.percentile(rows[:, 3], [10, 50, 90])))
        if best is None or ok.sum() > best[1].sum():
            best = (name, ok, segs)
    name, ok, segs = best
    print("\nbest feature: {}".format(name))
    print("segments (broadcast minutes -> tactical minutes, shift in frames):")
    for s in segs:
        print("  bc {:6.1f}-{:6.1f} min  ->  tac {:6.1f}-{:6.1f} min   shift {:+8d}   chunks {:>3}  median peak {:.2f}".format(
            s["b0"] / P.FPS / 60, (s["b1"] + s["chunk"]) / P.FPS / 60, (s["b0"] + s["shift"]) / P.FPS / 60,
            (s["b1"] + s["chunk"] + s["shift"]) / P.FPS / 60, int(s["shift"]), len(s["peaks"]),
            np.median(s["peaks"])))
    CHUNK_USED = segs[0]["chunk"] if segs else CHUNK

    # broadcast frame -> tactical frame, from the segments with 3+ chunks (piecewise constant)
    big = [s for s in segs if len(s["peaks"]) >= 3]
    to_tac = np.full(n_b, -1, dtype=np.int64)
    for s in big:
        lo, hi = int(s["b0"]), int(min(s["b1"] + CHUNK_USED, n_b))
        to_tac[lo:hi] = np.arange(lo, hi) + int(s["shift"])
    to_tac[(to_tac < 0) | (to_tac >= n_t)] = -1
    mapped = to_tac >= 0
    print("\nbroadcast frames mapped: {:.1%} (segments with 3+ chunks: {})".format(mapped.mean(), len(big)))
    np.savez_compressed(MAP, to_tac=to_tac)

    from load_homography import load as load_h
    _, _, valid = load_h(source="broadcast")
    valid = np.asarray(valid).astype(bool)[:n_b]
    inv_s, inv_l = runs(~valid)
    print("broadcast invalid-homography stretches: {} (>= 1 s: {}), {:.1%} of frames".format(
        len(inv_s), int(np.sum(inv_l >= P.FPS)), np.mean(~valid)))

    sub = os.path.join(ROOT, "evaluation", "results", "v14.csv")
    if not os.path.exists(sub):
        print("(run solution/run.py first to compare with v14)")
        return 0
    lab = np.loadtxt(sub, delimiter=",", skiprows=1, dtype=str, usecols=1)
    inv_t = np.zeros(n_t, dtype=bool)
    m = mapped & ~valid
    inv_t[to_tac[m]] = True
    g = graded
    print("\non the tactical timeline (graded frames), against our v14 labels:")
    print("  invalid-broadcast frames: {:.1%} of graded; of these, v14 says D {:.0%}".format(
        np.mean(inv_t[g]), np.mean(lab[inv_t & g] == "D") if np.any(inv_t & g) else 0))
    print("  v14 D frames covered by invalid broadcast: {:.0%};  v14 live frames covered: {:.1%}".format(
        np.mean(inv_t[g & (lab == "D")]), np.mean(inv_t[g & (lab != "D")])))
    cand = 0
    cand_s = 0.0
    for s, l in zip(*runs(inv_t & g)):
        if l >= 5 * P.FPS and np.mean(lab[s:s + l] == "D") < 0.2:
            cand += 1
            cand_s += l / P.FPS
    print("  invalid stretches >= 5 s that v14 calls live (candidate missed stoppages): {} ({:.0f} s)".format(
        cand, cand_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
