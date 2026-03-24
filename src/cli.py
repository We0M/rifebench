"""CLI argument definitions.

Kept in a separate module so __main__ stays tiny and the arg spec
is easy to review / modify in one place.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

def sc_threshold_type(value: str) -> float:
    v = float(value)
    if not (0.05 <= v <= 0.25):
        raise argparse.ArgumentTypeError(
            "sc_threshold must be in range [0.05, 0.25]"
        )
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rifebench",
        description="Benchmark RIFE models via VapourSynth + vs-mlrt.",
    )

    # -- required ------------------------------------------------------
    p.add_argument("--script", type=str, required=True,
                   help="Path to .vpy script")
    p.add_argument("--source", type=str, required=True,
                   help="Path to source video file.")
    p.add_argument("--models", type=str, required=True,
                   help="Comma-separated model IDs, e.g. 47,425,417")

    # -- mode --------------------------------------------------
    p.add_argument("--mode", choices=("vspipe", "ffmpeg"), default="vspipe",
                   help="Runner mode (default: vspipe)")

    # -- VPY forwarded parameters (passed as -a strings) --------------
    p.add_argument("--format", choices=("RGBS", "RGBH"), default="RGBH",
                   help="Pixel format for RIFE (default: RGBH)")
    p.add_argument("--multi", type=int, default=2,
                   help="Frame multiplier (default: 2)")
    p.add_argument("--rife_v", type=int, choices=(1, 2), default=2,
                   help="RIFE version .onnx (default: 2)")
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--trt_fp16", type=int, choices=(0, 1), default=1)
    p.add_argument("--use_cuda_graph", type=int, choices=(0, 1), default=1)
    p.add_argument("--trt_out_fmt", type=int, choices=(0, 1), default=1,
                   help="output_format 0: fp32, 1: fp16 (default: 1)")
    p.add_argument("--gpu_id", type=int, choices=(0, 15), default=0,
                   help="CUDA device index (default: 0)")
    p.add_argument("--sc_threshold", type=sc_threshold_type, default=0.10,
                   metavar="FLOAT", help="Scene-cut threshold for RIFE "
                   "(lower=more sensitive, higher=less sensitive). "
                   "Typical: 0.08-0.10 anime, 0.10-0.14 live-action, "
                   "0.14-0.20 noisy sources (default: %(default)s)")

    # -- benchmark parameters -----------------------------------------
    p.add_argument("--warmup_frames", type=int, default=1000,
                   help="Frames for warmup; 0 = full clip (default: 1000)")
    p.add_argument("--warmup_runs", type=int, default=1,
                   help="Warmup iterations before real tests (default: 1)")
    p.add_argument("--tests", type=int, default=1,
                   help="Test iterations per model (default: 1)")
    p.add_argument("--shuffle", type=int, choices=(0, 1), default=1,
                   help="Randomise model order between repetitions (default: 1)")
    p.add_argument("--cooldown_s", type=int, default=4,
                   help="Seconds to sleep between runs to reduce jitter (default: 4)")

    # -- output --------------------------------------------------------
    p.add_argument("--output_dir", type=str, default="./bench_results")
    p.add_argument(
        "--bench_name", type=str,
        default=None,  # filled below if not provided
        help="Base name for report file (default: bench_YY-MM-DD_HH-MM)",
    )

    # -- GPU sampling -------------------------------------------------
    p.add_argument("--period_ms", type=int, default=100,
                   help="GPU sampling period in ms (default: 100)")

    # -- ffmpeg-specific ----------------------------------------------
    p.add_argument(
        "--command", type=str, default=None,
        help="Custom ffmpeg command fragment (everything after the pipe).",
    )
    p.add_argument("--ffmpeg_path", type=str, default="ffmpeg")
    p.add_argument("--vspipe_path", type=str, default="vspipe")

    # -- frame range (optional, forwarded to vspipe -s / -e) ----------
    p.add_argument("--s_frame", type=int, default=None,
                   help="First output frame (vspipe -s)")
    p.add_argument("--e_frame", type=int, default=None,
                   help="Last output frame (vspipe -e)")

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and post-process CLI arguments."""
    p = build_parser()
    args = p.parse_args(argv)

    # Normalise paths.
    args.source = str(Path(args.source).resolve())
    args.script = str(Path(args.script).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())

    # Split model list into ints.
    args.models = [int(m.strip()) for m in args.models.split(",")]

    # Default bench name uses current timestamp.
    if args.bench_name is None:
        args.bench_name = datetime.now().strftime("bench_%y-%m-%d_%H-%M")

    return args
