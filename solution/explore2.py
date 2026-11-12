"""Tahap 1.3 (lanjutan) -- dua pertanyaan dari explore.py.

    .\\.venv\\Scripts\\python.exe solution\\explore2.py

A. Kecepatan bola di ATD tidak pernah > 10 m/s. Apakah kolomnya aneh, atau posisi bolanya sendiri
   yang diperhalus / menempel ke pemain? -> hitung kecepatan sendiri dari perpindahan posisi.
B. Bola hilang dalam lubang panjang (median ~11 s). Selama lubang itu bola mati atau masih main?
C. Di mana bola terakhir terlihat sebelum lubang, dan di mana muncul lagi sesudahnya?

Sama seperti explore.py: hanya angka ringkasan, tanpa ground truth, tanpa output per frame.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from explore import (ATTACKING_CODES, DEFENDING_CODES, FPS, LENGTH, WIDTH,  # noqa: E402
                     graded_mask, header, load_smart_tagging, pct, percentiles, runs,
                     side_of_team_tag)
from atd import load_arrays  # noqa: E402

LONG_GAP = 5 * FPS          # a "long" gap: 5 s or more
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16


def dist_to_line(x, y):
    """Distance to the nearest touch/goal line in metres (negative = outside the pitch)."""
    return np.minimum.reduce([x, LENGTH - x, y, WIDTH - y])


def in_box(x, y):
    return ((x < BOX_DEPTH) | (x > LENGTH - BOX_DEPTH)) & (np.abs(y - WIDTH / 2) < BOX_HALF_WIDTH)


def near_centre(x, y, radius=3.0):
    return np.hypot(x - LENGTH / 2, y - WIDTH / 2) < radius


def main():
    a = load_arrays()
    ball, players = a["ball"], a["players"]
    n = len(ball)
    graded = graded_mask(n)
    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    bx, by = ball[:, 0], ball[:, 1]

    # -- A. kecepatan bola dihitung sendiri ------------------------------------------------------
    header("A. KECEPATAN BOLA DIHITUNG DARI PERPINDAHAN POSISI")
    for lag, name in ((1, "antar 1 frame (0,04 s)"), (5, "antar 5 frame (0,2 s)"),
                      (25, "antar 25 frame (1 s)")):
        ok = visible[lag:] & visible[:-lag] & graded[lag:] & graded[:-lag]
        step = np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag])[ok]
        speed = step * FPS / lag
        print("{}: {}".format(name, percentiles(speed)))
        print("    >10 m/s: {}   >20 m/s: {}   <1 m/s: {}".format(
            pct(np.mean(speed > 10)), pct(np.mean(speed > 20)), pct(np.mean(speed < 1))))

    ok1 = visible[1:] & visible[:-1] & graded[1:]
    step1 = np.hypot(np.diff(bx), np.diff(by))[ok1]
    print("lompatan antar frame > 2 m (= >50 m/s, tidak mungkin secara fisik): {}".format(
        pct(np.mean(step1 > 2))))
    print("bola tidak bergerak sama sekali antar frame (< 1 cm): {}".format(
        pct(np.mean(step1 < 0.01))))

    col = ball[1:, 2][ok1]
    calc = (step1 * FPS)
    good = np.isfinite(col) & (calc > 0.05)
    if good.any():
        ratio = col[good] / calc[good]
        corr = np.corrcoef(col[good], calc[good])[0, 1]
        print("kolom kecepatan bola / kecepatan dihitung: " + percentiles(ratio))
        print("korelasi kolom vs dihitung: {:.2f}  (1 = searah sempurna)".format(corr))

    dist = np.hypot(players[:, :, 0] - bx[:, None], players[:, :, 1] - by[:, None])
    nearest = np.nanmin(dist, axis=1)
    okn = visible & graded & np.isfinite(nearest)
    print("bola 'menempel' pemain (jarak < 0,1 m): {}   (< 0,3 m): {}".format(
        pct(np.mean(nearest[okn] < 0.1)), pct(np.mean(nearest[okn] < 0.3))))
    near_speed = np.take_along_axis(
        players[:, :, 2], np.nanargmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)[:, None],
        axis=1)[:, 0]
    both = okn & np.isfinite(near_speed) & np.isfinite(ball[:, 2])
    if both.any():
        print("korelasi kecepatan bola (kolom) vs kecepatan pemain terdekat: {:.2f}".format(
            np.corrcoef(ball[both, 2], near_speed[both])[0, 1]))

    # -- B. isi lubang panjang -------------------------------------------------------------------
    header("B. LUBANG PANJANG (bola hilang >= 5 s, frame dinilai)")
    events = load_smart_tagging.load()
    live_tag = np.zeros(n, dtype=bool)          # frame covered by any team-tagged phase
    sp_start = np.zeros(n, dtype=bool)          # frame where a set-piece tag starts
    for e in events:
        code = (e["code"] or "").strip().lower()
        s, t = max(e["start_frame"], 0), min(e["end_frame"], n)
        if t <= s:
            continue
        if code == "set pieces":
            sp_start[s] = True
        elif code in ATTACKING_CODES | DEFENDING_CODES and side_of_team_tag(e["tags"].get("Team")) is not None:
            live_tag[s:t] = True

    starts, lengths = runs(graded & ~visible)
    long_sel = lengths >= LONG_GAP
    starts, lengths = starts[long_sel], lengths[long_sel]
    mean_speed = np.nanmean(players[:, :, 2], axis=1)
    oob = (bx < 0) | (bx > LENGTH) | (by < 0) | (by > WIDTH)

    rows = []
    for s, l in zip(starts, lengths):
        e = s + l
        before = slice(max(s - FPS, 0), s)                  # last second before the gap
        last = np.flatnonzero(visible[:s])
        nxt = np.flatnonzero(visible[e:e + 10 * FPS])
        lx, ly = (bx[last[-1]], by[last[-1]]) if last.size else (np.nan, np.nan)
        rx, ry = (bx[e + nxt[0]], by[e + nxt[0]]) if nxt.size else (np.nan, np.nan)
        rows.append(dict(
            length=l,
            live_frac=live_tag[s:e].mean(),
            sp_near_end=sp_start[max(e - 10 * FPS, s):min(e + 5 * FPS, n)].any(),
            sp_inside=sp_start[s:e].any(),
            out_before=oob[before].any(),
            last_line=dist_to_line(lx, ly), last_box=bool(in_box(lx, ly)),
            reap_line=dist_to_line(rx, ry), reap_centre=bool(near_centre(rx, ry)),
            speed=np.nanmean(mean_speed[s:e]),
        ))

    if not rows:
        print("tidak ada lubang panjang")
        return
    L = np.array([r["length"] for r in rows])
    live = np.array([r["live_frac"] for r in rows])
    spe = np.array([r["sp_near_end"] for r in rows])
    spi = np.array([r["sp_inside"] for r in rows])
    outb = np.array([r["out_before"] for r in rows])
    gspeed = np.array([r["speed"] for r in rows])
    print("jumlah lubang panjang: {}   total frame: {}".format(len(rows), int(L.sum())))

    vis_g = visible & graded
    miss_g = ~visible & graded
    print("\nporsi frame yang tertutup tag fase permainan (= indikasi bola masih main):")
    print("  saat bola terlihat: {}   saat bola hilang: {}".format(
        pct(live_tag[vis_g].mean()), pct(live_tag[miss_g].mean())))
    print("rata-rata kecepatan pemain (m/s):")
    print("  saat bola terlihat: {}".format(percentiles(mean_speed[vis_g])))
    print("  saat bola hilang:   {}".format(percentiles(mean_speed[miss_g])))

    print("\nper lubang (n = jumlah lubang, frame = porsi frame lubang):")
    print("  {:<52} {:>5} {:>8}".format("ciri", "n", "frame"))

    def line(name, sel):
        print("  {:<52} {:>5} {:>8}".format(name, int(sel.sum()), pct(L[sel].sum() / L.sum())))

    line("set piece dimulai di dalam lubang", spi)
    line("set piece dimulai dekat akhir lubang (-10 s..+5 s)", spe)
    line("bola keluar garis 1 s sebelum lubang", outb)
    line(">= 50% lubang tertutup tag fase permainan", live >= .5)
    line("< 10% lubang tertutup tag fase permainan", live < .1)
    line("pemain rata-rata pelan (< 1,5 m/s) selama lubang", gspeed < 1.5)
    dead_like = spe | spi | outb
    line("ADA tanda bola mati (set piece / bola keluar)", dead_like)
    line("TIDAK ada tanda bola mati DAN >= 50% tertutup tag", ~dead_like & (live >= .5))

    print("\nmenurut panjang lubang:")
    print("  {:<14} {:>4} {:>18} {:>16} {:>16}".format(
        "panjang", "n", "ada tanda mati", "tag main >=50%", "pemain pelan"))
    for lo, hi, name in ((125, 250, "5-10 s"), (251, 500, "10-20 s"), (501, 1500, "20-60 s"),
                         (1501, 10 ** 9, "> 60 s")):
        sel = (L >= lo) & (L <= hi)
        if sel.any():
            print("  {:<14} {:>4} {:>18} {:>16} {:>16}".format(
                name, int(sel.sum()), int(dead_like[sel].sum()), int((live[sel] >= .5).sum()),
                int((gspeed[sel] < 1.5).sum())))

    # -- C. sebelum dan sesudah lubang -----------------------------------------------------------
    header("C. POSISI BOLA SEBELUM DAN SESUDAH LUBANG PANJANG")
    last_line = np.array([r["last_line"] for r in rows])
    last_box = np.array([r["last_box"] for r in rows])
    reap_line = np.array([r["reap_line"] for r in rows])
    reap_centre = np.array([r["reap_centre"] for r in rows])
    print("terakhir terlihat - jarak ke garis (m): " + percentiles(last_line))
    print("  dalam 2 m dari garis / di luar: {}   di kotak penalti: {}".format(
        pct(np.nanmean(last_line < 2)), pct(np.mean(last_box))))
    print("muncul lagi (dalam 10 s) - jarak ke garis (m): " + percentiles(reap_line))
    print("  dalam 2 m dari garis / di luar: {}   dekat titik tengah (kick-off): {}".format(
        pct(np.nanmean(reap_line < 2)), pct(np.mean(reap_centre))))
    print("  tidak muncul dalam 10 s: {}".format(pct(np.mean(np.isnan(reap_line)))))

    print("\nselesai. Semua angka di atas ringkasan; aman dibagikan ke tim.")


if __name__ == "__main__":
    main()
