"""Does a video file carry an audio track? Reads only the MP4 container header (track list),
never the picture or the sound itself, and needs no extra package.

    .\\.venv\\Scripts\\python.exe solution\\probe_audio.py            # tactical and broadcast
    .\\.venv\\Scripts\\python.exe solution\\probe_audio.py file.mp4

Step 1 of whistle detection: without a sound track there is nothing to detect.
"""

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def boxes(fh, start, end):
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        size, kind = struct.unpack(">I4s", fh.read(8))
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", fh.read(8))[0], 16
        elif size == 0:
            size = end - pos
        if size < head:
            return
        yield kind, pos + head, pos + size
        pos += size


def tracks(path):
    out = []
    with open(path, "rb") as fh:
        end = os.path.getsize(path)
        for kind, a, b in boxes(fh, 0, end):
            if kind != b"moov":
                continue
            for k2, a2, b2 in boxes(fh, a, b):
                if k2 != b"trak":
                    continue
                info = {}
                for k3, a3, b3 in boxes(fh, a2, b2):
                    if k3 != b"mdia":
                        continue
                    for k4, a4, b4 in boxes(fh, a3, b3):
                        fh.seek(a4)
                        data = fh.read(min(b4 - a4, 64))
                        if k4 == b"hdlr":
                            info["type"] = data[8:12].decode("latin-1")
                        elif k4 == b"mdhd":
                            v = data[0]
                            if v == 1:
                                scale, dur = struct.unpack(">IQ", data[20:32])
                            else:
                                scale, dur = struct.unpack(">II", data[12:20])
                            info["seconds"] = dur / scale if scale else 0
                            info["rate"] = scale
                        elif k4 == b"minf":
                            for k5, a5, b5 in boxes(fh, a4, b4):
                                if k5 == b"stbl":
                                    for k6, a6, _ in boxes(fh, a5, b5):
                                        if k6 == b"stsd":
                                            fh.seek(a6 + 8)
                                            fh.read(4)
                                            info["codec"] = fh.read(4).decode("latin-1")
                out.append(info)
    return out


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        from games import GAME, video_path
        paths = [video_path(GAME, source=s) for s in ("tactical", "broadcast")]
    for path in paths:
        print(os.path.basename(path))
        if not os.path.exists(path):
            print("  not found")
            continue
        found = tracks(path)
        for t in found:
            label = {"soun": "AUDIO", "vide": "video"}.get(t.get("type"), t.get("type"))
            print("  {:<6} codec {:<5} {:>8.1f} s   timescale {}".format(
                label, t.get("codec", "?"), t.get("seconds", 0), t.get("rate", "?")))
        has = any(t.get("type") == "soun" for t in found)
        print("  -> {}".format("has an audio track" if has else "NO audio track"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
