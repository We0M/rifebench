"""Runner — builds commands, executes vspipe / ffmpeg, streams output
to the line-level parsers, and coordinates the GPU sampler.

Key design choices:
- subprocess.Popen is used (not run()) so we can read stderr line-by-
  line without buffering the entire output, and so the GPU sampler
  thread runs concurrently with the child process.
- In ffmpeg mode the two processes are connected via an OS pipe managed
  by Popen (stdout of vspipe → stdin of ffmpeg).
- Warmup uses the same command but limits frames with vspipe -e flag.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .parser import (
    FfmpegLineParser,
    FfmpegResult,
    VspipeResult,
    parse_vspipe_line,
)
from .sampler import GpuSampler, GpuSnapshot


# =====================================================================
# Run result (one invocation of a model)
# =====================================================================

@dataclass
class RunResult:
    model: int
    run_id: int               # 1-based
    is_warmup: bool = False

    # Metrics from parser.
    frames: int | None = None
    fps: float | None = None
    time_s: float | None = None          # vspipe time or ffmpeg rtime
    median_fps: float | None = None      # ffmpeg only

    # GPU samples.
    gpu_samples: list[GpuSnapshot] = field(default_factory=list)

    # Error bookkeeping.
    ok: bool = False
    error: str = ""


# =====================================================================
# Command builders
# =====================================================================

def _vpy_args(args) -> list[str]:
    """Build the ``-a key=value`` portion forwarded to the .vpy script.

    Every parameter is passed as a string — the .vpy is responsible for
    casting them to the correct types.
    """
    pairs = [
        ("gpu_id",         args.gpu_id),
        ("source",         args.source),
        # model is set per-run, so it is NOT included here.
        ("format",         args.format),
        ("multi",          args.multi),
        ("rife_v",         args.rife_v),
        ("streams",        args.streams),
        ("trt_fp16",       args.trt_fp16),
        ("use_cuda_graph", args.use_cuda_graph),
        ("trt_out_fmt",    args.trt_out_fmt),
        ("sc_threshold",   args.sc_threshold),
    ]
    parts: list[str] = []
    for k, v in pairs:
        parts += ["-a", f"{k}={v}"]
    return parts


def build_vspipe_cmd(
    args,
    model: int,
    *,
    warmup_e_frame: int | None = None,
) -> list[str]:
    """Construct the full vspipe command list.

    In pure-vspipe mode outfile is ``--`` (no output written, maximum
    speed).  In ffmpeg mode the caller replaces it with ``-`` (stdout).
    """
    cmd = [args.vspipe_path, "-c", "y4m"]
    # Progress flag so vspipe writes progress to stderr.
    cmd.append("-p")
    cmd.append(args.script)
    # Outfile: ``--`` = discard frames (fastest for benchmarking).
    cmd.append("--")
    # VPY arguments.
    cmd += _vpy_args(args)
    cmd += ["-a", f"model={model}"]

    # Optional frame range.
    s_frame = args.s_frame
    # For warmup we override -e to limit frames.
    e_frame = warmup_e_frame if warmup_e_frame is not None else args.e_frame
    if s_frame is not None:
        cmd += ["-s", str(s_frame)]
    if e_frame is not None:
        cmd += ["-e", str(e_frame)]

    return cmd


def build_ffmpeg_cmd(args) -> list[str]:
    """Build the ffmpeg command (the part after the pipe).

    If the user provided --command we use it verbatim (split by spaces).
    Otherwise we fall back to a sensible hevc_nvenc default.

    NUL on Windows, /dev/null on Linux — automatically chosen.
    """
    if args.command:
        # User-provided fragment.  We prepend the ffmpeg binary path.
        parts = args.command.strip().split()
        # If the user already included 'ffmpeg' at the start, don't duplicate.
        if parts and parts[0].lower().rstrip(".exe") == "ffmpeg":
            parts[0] = args.ffmpeg_path
        else:
            parts = [args.ffmpeg_path] + parts
        return parts

    null = "NUL" if platform.system() == "Windows" else "/dev/null"
    return [
        args.ffmpeg_path,
        "-benchmark",
        "-i", "-",
        "-c:v", "hevc_nvenc",
        "-preset", "p5",
        "-cq", "18",
        "-profile:v", "main10",
        "-pix_fmt", "p010le",
        "-f", "null",
        null,
    ]


# =====================================================================
# Single-run executors
# =====================================================================

def _run_vspipe(cmd: list[str], sampler: GpuSampler | None) -> RunResult:
    """Execute a pure-vspipe benchmark run.

    vspipe writes progress to stderr and the final summary to stdout.
    We only need the summary line from stdout.
    """
    result = RunResult(model=0, run_id=0)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        result.error = f"Failed to start vspipe: {exc}"
        return result

    if sampler:
        sampler.start()

    # Read stdout and stderr in parallel.
    stdout_data, stderr_data = proc.communicate()

    if sampler:
        result.gpu_samples = sampler.stop()

    if proc.returncode != 0:
        result.error = stderr_data or f"vspipe exited with code {proc.returncode}"
        return result

    # Parse the summary line from stdout.
    # vspipe writes the summary to stdout after all frame data.
    # However, when outfile is "--" it also goes to stderr.
    # Check both.
    parsed = None
    for src in (stdout_data, stderr_data):
        for line in src.splitlines():
            p = parse_vspipe_line(line)
            if p and p.ok:
                parsed = p
                break
        if parsed:
            break

    if parsed is None or not parsed.ok:
        result.error = stderr_data or "Could not parse vspipe summary"
        return result

    result.frames = parsed.frames
    result.fps = parsed.fps
    result.time_s = parsed.time_s
    result.ok = True
    return result


def _run_ffmpeg(
    vspipe_cmd: list[str],
    ffmpeg_cmd: list[str],
    sampler: GpuSampler | None,
) -> RunResult:
    """Execute vspipe | ffmpeg pipeline.

    vspipe stdout is piped into ffmpeg stdin.  ffmpeg writes progress
    to stderr, so we read that line-by-line and feed it to FfmpegLineParser.
    """
    result = RunResult(model=0, run_id=0)

    # Replace outfile with ``-`` so vspipe writes to stdout.
    vspipe_cmd = list(vspipe_cmd)
    try:
        idx = vspipe_cmd.index("--")
        vspipe_cmd[idx] = "-"
    except ValueError:
        pass

    try:
        vs_proc = subprocess.Popen(
            vspipe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ff_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=vs_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Allow vspipe to receive SIGPIPE if ffmpeg exits.
        if vs_proc.stdout:
            vs_proc.stdout.close()
    except Exception as exc:
        result.error = f"Failed to start pipeline: {exc}"
        return result

    if sampler:
        sampler.start()

    parser = FfmpegLineParser()
    # Read ffmpeg stderr line-by-line (it writes progress there).
    if ff_proc.stderr:
        for raw in ff_proc.stderr:
            # ffmpeg progress lines use \r without \n; splitlines helps.
            for line in raw.splitlines():
                parser.feed(line)

    ff_proc.wait()
    vs_proc.wait()

    if sampler:
        result.gpu_samples = sampler.stop()

    if ff_proc.returncode != 0:
        ff_res = parser.result()
        result.error = ff_res.stderr_log or f"ffmpeg exited with code {ff_proc.returncode}"
        return result

    ff_res = parser.result()
    if not ff_res.ok:
        result.error = ff_res.stderr_log or "Could not parse ffmpeg metrics"
        return result

    result.frames = ff_res.frames
    result.fps = ff_res.fps
    result.time_s = ff_res.rtime
    result.median_fps = ff_res.median_fps
    result.ok = True
    return result


# =====================================================================
# Orchestrator
# =====================================================================

def run_benchmarks(args, on_result=None) -> list[RunResult]:
    """Execute the full benchmark session.

    *on_result* is an optional callback ``(RunResult) -> None`` invoked
    after every single run, enabling incremental reporting.

    Returns all RunResult objects (including warmup, which are marked
    with ``is_warmup=True``).
    """
    results: list[RunResult] = []
    models: list[int] = list(args.models)
    sampler = GpuSampler(args.gpu_id, args.period_ms)

    # Pre-compute warmup end frame.
    warmup_e: int | None = None
    if args.warmup_frames > 0:
        start = args.s_frame or 0
        warmup_e = start + args.warmup_frames - 1

    # Helper to execute one run.
    def _exec(model: int, run_id: int, *, is_warmup: bool) -> RunResult:
        e_override = warmup_e if is_warmup else None
        vs_cmd = build_vspipe_cmd(args, model, warmup_e_frame=e_override)

        # GPU sampler only for real (non-warmup) runs.
        active_sampler = None if is_warmup else sampler

        if args.mode == "vspipe":
            res = _run_vspipe(vs_cmd, active_sampler)
        else:
            ff_cmd = build_ffmpeg_cmd(args)
            res = _run_ffmpeg(vs_cmd, ff_cmd, active_sampler)

        res.model = model
        res.run_id = run_id
        res.is_warmup = is_warmup
        return res

    # -- Warmup ----------------------------------------------------
    for wi in range(args.warmup_runs):
        for model in models:
            tag = f"warmup {wi + 1}/{args.warmup_runs}"
            print(f"[{tag}] model={model} …", flush=True)
            res = _exec(model, run_id=wi + 1, is_warmup=True)
            results.append(res)
            if on_result:
                on_result(res)
            status = "OK" if res.ok else f"FAIL: {res.error[:80]}"
            print(f"  -> {status}", flush=True)
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

    # -- Test runs -------------------------------------------------
    for ti in range(args.tests):
        order = list(models)
        if args.shuffle:
            random.shuffle(order)
        for model in order:
            tag = f"test {ti + 1}/{args.tests}"
            print(f"[{tag}] model={model} …", flush=True)
            res = _exec(model, run_id=ti + 1, is_warmup=False)
            results.append(res)
            if on_result:
                on_result(res)

            # status = f"{res.fps:.2f} fps" if res.ok else f"FAIL: {res.error[:80]}"
            col_def = "\033[0m"
            col_red = "\033[31m"
            col_gre = "\033[32m"
            if res.ok:
                st_ok = f"{col_gre} PASS:{col_def}"
                status = f"{st_ok} {res.time_s} seconds | {res.fps:.2f} fps"
            else:
                st_fa = f"{col_red} FAIL:"
                status = f"{st_fa} {res.error[:130]}{col_def}"
            print(f"  ->{status}", flush=True)
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

    return results
