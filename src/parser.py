"""Parsers for vspipe / ffmpeg output and source/clip probing.

Design:
- Parsing happens line-by-line inside the runner so we never buffer
  the full output; only the needed metrics are extracted.
- Two public dataclasses (VspipeResult, FfmpegResult) carry the parsed
  values back to the caller.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field


# =====================================================================
# Result containers
# =====================================================================

@dataclass
class VspipeResult:
    frames: int | None = None
    fps: float | None = None
    time_s: float | None = None
    ok: bool = False          # True when all three metrics are present
    stderr_log: str = ""      # kept only on failure for diagnostics


@dataclass
class FfmpegResult:
    frames: int | None = None
    fps: float | None = None
    rtime: float | None = None
    median_fps: float | None = None   # median of progress fps samples
    ok: bool = False
    stderr_log: str = ""


# =====================================================================
# Line-level parsers (pure functions, easy to unit-test)
# =====================================================================

# vspipe final line: "Output 2882 frames in 29.72 seconds (96.99 fps)"
_VS_FINAL = re.compile(
    r"Output\s+(\d+)\s+frames\s+in\s+([\d.]+)\s+seconds\s+\(([\d.]+)\s+fps\)"
)


def parse_vspipe_line(line: str) -> VspipeResult | None:
    """Return VspipeResult if *line* is the vspipe summary line, else None."""
    m = _VS_FINAL.search(line)
    if m is None:
        return None
    return VspipeResult(
        frames=int(m.group(1)),
        time_s=float(m.group(2)),
        fps=float(m.group(3)),
        ok=True,
    )


# ffmpeg progress:  "frame= 2882 fps= 86 q=… Lsize=… time=… …"
_FF_PROGRESS_FPS = re.compile(r"fps=\s*([\d.]+)")

# ffmpeg bench line: "bench: utime=…s stime=…s rtime=33.383s"
_FF_BENCH = re.compile(r"rtime=([\d.]+)s")

# ffmpeg final stats after "[out#…]":
# "frame= 2882 fps= 86 q=19.0 Lsize=N/A time=00:01:00.03 …"
_FF_FRAME = re.compile(r"frame=\s*(\d+)")


class FfmpegLineParser:
    """Stateful line parser for ffmpeg stderr.

    Call ``feed(line)`` for every stderr line.
    After the process ends call ``result()`` to get the FfmpegResult.

    State machine:
      - Before seeing ``[out#`` we are in the *progress* phase → collect fps
        samples for median.
      - After ``[out#`` we switch to *final* phase → extract frames, fps,
        and rtime from the subsequent summary/bench lines.
    """

    def __init__(self) -> None:
        self._fps_samples: list[float] = []
        self._final_phase = False
        self._frames: int | None = None
        self._fps: float | None = None
        self._rtime: float | None = None
        self._stderr_lines: list[str] = []

    def feed(self, line: str) -> None:
        self._stderr_lines.append(line)

        if not self._final_phase:
            # Progress phase: accumulate fps.
            if "[out#" in line:
                self._final_phase = True
                return
            m = _FF_PROGRESS_FPS.search(line)
            if m:
                val = float(m.group(1))
                if val > 0:
                    self._fps_samples.append(val)
        else:
            # Final phase: look for summary frame line and bench rtime.
            fm = _FF_FRAME.search(line)
            if fm:
                self._frames = int(fm.group(1))
                fpm = _FF_PROGRESS_FPS.search(line)
                if fpm:
                    self._fps = float(fpm.group(1))
            bm = _FF_BENCH.search(line)
            if bm:
                self._rtime = float(bm.group(1))

    def result(self) -> FfmpegResult:
        median_fps = _median(self._fps_samples) if self._fps_samples else None
        ok = all(v is not None for v in (self._frames, self._fps, self._rtime))
        return FfmpegResult(
            frames=self._frames,
            fps=self._fps,
            rtime=self._rtime,
            median_fps=median_fps,
            ok=ok,
            stderr_log="" if ok else "".join(self._stderr_lines),
        )


# =====================================================================
# Source / Clip probing
# =====================================================================

def probe_source(source: str) -> dict[str, str]:
    """Try to extract video properties from *source* using ffmpeg-python.

    Returns dict with keys: Width, Height, Frames, FPS, Format.
    Falls back to 'N/A' per-field on any failure.
    """
    na = {k: "N/A" for k in ("Width", "Height", "Frames", "FPS", "Format")}
    try:
        import ffmpeg as ffmpeg_python  # optional dep
    except ImportError:
        return na
    try:
        info = ffmpeg_python.probe(source, select_streams="v:0")
        stream = info["streams"][0]
        # Frame count: some containers store nb_frames, others need
        # nb_read_frames from -count_frames.  We try the cheap way first.
        frames = stream.get("nb_frames", "N/A")
        # FPS from r_frame_rate or avg_frame_rate.
        fps_str = stream.get("r_frame_rate") or stream.get("avg_frame_rate", "0/0")
        try:
            num, den = fps_str.split("/")
            fps_val = f"{int(num) / int(den):.3f}" if int(den) else "N/A"
        except (ValueError, ZeroDivisionError):
            fps_val = "N/A"
        return {
            "Width": str(stream.get("width", "N/A")),
            "Height": str(stream.get("height", "N/A")),
            "Frames": str(frames),
            "FPS": fps_val,
            "Format": stream.get("pix_fmt", "N/A"),
        }
    except Exception:
        return na


def probe_clip(
    script: str,
    source: str,
    models: list[int],
    vspipe_path: str = "vspipe",
) -> dict[str, str]:
    """Run ``vspipe -i script -a source=…`` and parse the clip properties.

    Returns dict with keys: Width, Height, Frames, FPS, Format Name.
    """
    na = {k: "N/A" for k in ("Width", "Height", "Frames", "FPS", "Format Name")}
    try:
        proc = None
        for m in models:
            cmd = [vspipe_path, "-i", script, "-a", f"source={source}", "-a", f"model={m}"]
            tmp_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if tmp_proc.returncode != 0:
                continue
            else:
                proc = tmp_proc

        if proc == None:
            return na
    except Exception:
        return na

    result = dict(na)
    for line in proc.stdout.splitlines():
        line = line.strip()
        # Lines look like "Width: 1920" or "FPS: 48000/1001 (47.952 fps)"
        for key in ("Width", "Height", "Frames", "Format Name"):
            if line.startswith(f"{key}:"):
                result[key] = line.split(":", 1)[1].strip()
        if line.startswith("FPS:"):
            # Extract the human-readable value from parentheses if present.
            m = re.search(r"\(([\d.]+)\s*fps\)", line)
            if m:
                result["FPS"] = m.group(1)
            else:
                result["FPS"] = line.split(":", 1)[1].strip()
    return result


# =====================================================================
# Utility: get VapourSynth version from ``vspipe -v``
# =====================================================================

def get_vs_version(vspipe_path: str = "vspipe") -> str:
    """Return e.g. 'Core R73' parsed from ``vspipe -v``, or 'N/A'."""
    try:
        proc = subprocess.run(
            [vspipe_path, "-v"], capture_output=True, text=True, timeout=10,
        )
        for line in proc.stdout.splitlines():
            m = re.search(r"(Core R\d+)", line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return "N/A"


# =====================================================================
# Helpers
# =====================================================================

def _median(data: list[float]) -> float:
    """Median without numpy (keeps deps minimal)."""
    s = sorted(data)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
