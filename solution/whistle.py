"""Referee whistles from the tactical video's sound track, found automatically.

    .\\.venv\\Scripts\\python.exe solution\\whistle.py            # decode + detect (a few minutes)
    .\\.venv\\Scripts\\python.exe solution\\whistle.py --cached   # detect again from the saved spectrum

Why: fouls, injuries and other stoppages that Smart Tagging does not tag are the largest error
left in the dead ball. The referee stops play with the whistle: one steady, narrow tone that
stands out from crowd noise, which is broadband.

Method, no frame-specific rule (RULES.md #1) and no hand label:
  1. ffmpeg decodes the sound to mono 16 kHz. The tactical sound track has the video's timeline
     (same duration), so 40 ms audio block k = video frame k.
  2. Per 40 ms block, a 64 ms Hann-windowed spectrum. For every 100 Hz bin of 2-4.5 kHz: its
     strongest line over the median of the whole band, in dB ("how much one tone stands out").
  3. v1 took the strongest line anywhere in 2-4.5 kHz and found 9 whistles in 106 minutes. The
     aggregate view (whistle_diag.py) showed why: the peak usually sits at 2.0-2.5 kHz on EVERY
     frame (a steady background tone and low-frequency spill), which hides the whistle.
  4. v2: the referee uses one whistle all match, so its pitch is the 100 Hz bin most
     over-represented among the most tonal frames (share in the top WHISTLE_PICK_TOP over share on
     all frames).
     It is chosen by this rule from the sound itself, then detection uses only that bin and its
     neighbours (+/- WHISTLE_HALF_BAND).
  5. Threshold = median + WHISTLE_K robust sd of that narrow-band tonality over the graded frames.
     A whistle = a run above it lasting WHISTLE_MIN_S or more, short dips closed.

Output: data/derived/BAR-ATH_whistle.npz (per-frame spectrum summary and whistle flag), and only
aggregate numbers on screen.
"""

import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames, video_path  # noqa: E402

import params as P  # noqa: E402
from deadball import close_holes, runs  # noqa: E402

RATE = 16000
HOP = RATE // P.FPS          # 640 samples = 40 ms = one video frame
WIN = 1024                   # 64 ms window, 15.6 Hz lines
BAND = (2000.0, 4500.0)
BIN_HZ = 100
EDGES = np.arange(BAND[0], BAND[1] + 1, BIN_HZ)
CHUNK = 20000                # frames per spectrum batch (memory)
OUT = os.path.join(ROOT, "data", "derived", "{}_whistle.npz".format(GAME))


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: pip install imageio-ffmpeg (bundles an ffmpeg binary)")


def decode(path):
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(RATE),
           "-f", "s16le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def spectrum(audio, n):
    """Per video frame: tonality per 100 Hz bin (dB over band median), peak Hz, band level dB."""
    freqs = np.fft.rfftfreq(WIN, 1.0 / RATE)
    band = (freqs >= BAND[0]) & (freqs < BAND[1])
    bf = freqs[band]
    which = np.digitize(bf, EDGES) - 1
    n_bins = len(EDGES) - 1
    window = np.hanning(WIN).astype(np.float32)
    pad = np.concatenate([np.zeros(WIN // 2, np.float32), audio, np.zeros(WIN, np.float32)])
    bins = np.full((n, n_bins), np.nan, np.float16)
    peak = np.full(n, np.nan, np.float32)
    level = np.full(n, np.nan, np.float32)
    last = min(n, (len(pad) - WIN) // HOP)
    for a in range(0, last, CHUNK):
        b = min(a + CHUNK, last)
        idx = (np.arange(a, b) * HOP)[:, None] + np.arange(WIN)[None, :]
        spec = np.abs(np.fft.rfft(pad[idx] * window, axis=1)[:, band]) ** 2 + 1e-12
        med = np.median(spec, axis=1, keepdims=True)
        rel = 10 * np.log10(spec / med)
        for k in range(n_bins):
            bins[a:b, k] = rel[:, which == k].max(axis=1)
        peak[a:b] = bf[spec.argmax(axis=1)]
        level[a:b] = 10 * np.log10(spec.mean(axis=1))
    return bins, peak, level


def pick_pitch(bins, graded):
    """Index of the 100 Hz bin most over-represented among the most tonal frames."""
    b = bins[graded].astype(np.float32)
    ok = np.all(np.isfinite(b), axis=1)
    b = b[ok]
    top = b.max(axis=1)
    where = b.argmax(axis=1)
    share_all = np.bincount(where, minlength=b.shape[1]) / len(where)
    n_top = max(int(P.WHISTLE_PICK_TOP * len(top)), 2 * P.WHISTLE_PICK_MIN)
    strong = np.argsort(top)[-n_top:]
    counts = np.bincount(where[strong], minlength=b.shape[1])
    share_top = counts / n_top
    ratio = np.where(counts >= P.WHISTLE_PICK_MIN, share_top / np.maximum(share_all, 1e-6), 0)
    return int(np.argmax(ratio)), share_all, share_top, ratio


def narrow(bins, k):
    lo, hi = max(k - P.WHISTLE_HALF_BAND, 0), min(k + P.WHISTLE_HALF_BAND + 1, bins.shape[1])
    sub = np.nan_to_num(bins[:, lo:hi].astype(np.float32), nan=-np.inf).max(axis=1)
    return np.where(np.isfinite(sub), sub, np.nan)


def detect(score, graded, k_sd):
    g = score[graded & np.isfinite(score)]
    med = np.median(g)
    mad = 1.4826 * np.median(np.abs(g - med))
    thr = med + k_sd * mad
    above = close_holes(np.where(graded, np.nan_to_num(score, nan=-np.inf), -np.inf) > thr, P.WHISTLE_GAP)
    flag = np.zeros(len(score), dtype=bool)
    durations = []
    for s, l in zip(*runs(above)):
        if l >= P.WHISTLE_MIN_S * P.FPS:
            flag[s:s + l] = True
            durations.append(l / P.FPS)
    return flag, thr, med, mad, np.array(durations)


def load(n, graded, verbose=True):
    """Per-frame whistle flag for the pipeline. Uses the saved spectrum when there is one (the
    detection is redone with the current params); otherwise decodes the sound and saves it."""
    if os.path.exists(OUT) and "bins" in np.load(OUT).files:
        bins = np.load(OUT)["bins"]
    else:
        if verbose:
            print("whistle: decoding the tactical sound track (a few minutes, once) ...", flush=True)
        bins, peak, level = spectrum(decode(video_path(GAME, source="tactical")), n)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        np.savez_compressed(OUT, bins=bins, peak=peak, level=level)
    k = pick_pitch(bins, graded)[0]
    flag, thr, _, _, dur = detect(narrow(bins, k), graded, P.WHISTLE_K)
    info = {"pitch_lo": float(EDGES[max(k - P.WHISTLE_HALF_BAND, 0)]),
            "pitch_hi": float(EDGES[min(k + P.WHISTLE_HALF_BAND + 1, len(EDGES) - 1)]),
            "threshold": float(thr), "n": int(len(dur))}
    return flag, info


def main():
    n = source_frames("tactical", GAME)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    if "--cached" in sys.argv and os.path.exists(OUT):
        d = np.load(OUT)
        bins, peak, level = d["bins"], d["peak"], d["level"]
    else:
        print("decoding sound track (a few minutes) ...", flush=True)
        audio = decode(video_path(GAME, source="tactical"))
        print("  {:.1f} s of sound for {:.1f} s of video".format(len(audio) / RATE, n / P.FPS))
        bins, peak, level = spectrum(audio, n)

    k, share_all, share_top, ratio = pick_pitch(bins, graded)
    print("\nwhistle pitch, chosen from the sound (share of frames whose strongest line is in the bin)")
    print("  bin Hz      all frames  top {:.0%}     ratio".format(P.WHISTLE_PICK_TOP))
    for i in np.argsort(ratio)[::-1][:6]:
        print("  {:>4.0f}-{:<4.0f}   {:7.2%}   {:7.2%}  {:6.1f}{}".format(
            EDGES[i], EDGES[i + 1], share_all[i], share_top[i], ratio[i], "  <- chosen" if i == k else ""))
    lo_hz = EDGES[max(k - P.WHISTLE_HALF_BAND, 0)]
    hi_hz = EDGES[min(k + P.WHISTLE_HALF_BAND + 1, len(EDGES) - 1)]
    score = narrow(bins, k)

    minutes = graded.sum() / P.FPS / 60
    print("\nnarrow band {:.0f}-{:.0f} Hz: whistles per minute at each threshold (runs >= {:.2f} s)".format(
        lo_hz, hi_hz, P.WHISTLE_MIN_S))
    for ks in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        fl, thr, _, _, dur = detect(score, graded, ks)
        print("  K {:>3}  thr {:5.1f} dB  {:5.2f} per minute ({:>4})  median {:.2f} s".format(
            ks, thr, len(dur) / minutes, len(dur), np.median(dur) if len(dur) else 0))

    flag, thr, med, mad, dur = detect(score, graded, P.WHISTLE_K)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, bins=bins, peak=peak, level=level, score=score, whistle=flag,
                        threshold=thr, pitch_bin=k)
    gw = flag & graded
    starts = runs(gw)[0]
    print("\nused: K {} -> threshold {:.1f} dB (median {:.1f} + {} x {:.2f}); {} whistles, {:.2f} per minute".format(
        P.WHISTLE_K, thr, med, P.WHISTLE_K, mad, len(starts), len(starts) / minutes))
    if len(dur):
        print("  duration s: p10 {:.2f}  p50 {:.2f}  p90 {:.2f}".format(*np.percentile(dur, [10, 50, 90])))
    print("  (plausibility: a referee blows roughly 1-2.5 times a minute of play, mostly 0.2-1 s)")

    sub = os.path.join(ROOT, "evaluation", "results", "v12.csv")
    if os.path.exists(sub) and len(starts):
        lab = np.loadtxt(sub, delimiter=",", skiprows=1, dtype=str, usecols=1)
        at = lab[starts]
        soon = np.array([np.any(lab[s:min(s + 5 * P.FPS, n)] == "D") for s in starts])
        print("\nagainst our own v12 labels (no ground truth):")
        print("  whistle starts on D {:.0%}; on live {:.0%}, of which D follows within 5 s {:.0%}".format(
            np.mean(at == "D"), np.mean(at != "D"), np.mean(soon[at != "D"]) if np.any(at != "D") else 0))
        print("  whistles in live play with no D within 5 s (candidate untagged stoppages): {}".format(
            int(np.sum((at != "D") & ~soon))))
    print("\nwritten: {}".format(os.path.relpath(OUT, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
