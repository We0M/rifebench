"""Entry point for ``python -m rifebench`` (and the ``rifebench`` console script).

Responsibilities:
1. Parse CLI args.
2. Validate that the requested GPU exists (early exit if not).
3. Create the Reporter and write the header (specs, source/clip info).
4. Hand off to the runner, passing ``reporter.append_run`` as the
   incremental callback so each run is recorded as it finishes.
5. Finalize the report with summary statistics.
"""

from __future__ import annotations

import sys

from . import __version__
from .cli import parse_args
from .reporter import Reporter
from .runner import run_benchmarks
from .sampler import gpu_exists


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    s_fp16 = "True" if args.rife_v == 1 else "False"
    s_fmt = "fp32" if args.trt_out_fmt == 0 else "fp16"

    print(f"\n{'='*62}", flush=True)
    print(f"  rifebench v{__version__}", flush=True)
    print(f"  Tests: {args.tests} |  Mode: {args.mode} |  Multi: {args.multi}   |  Format: {args.format}", flush=True)
    print(f"  RIFE: v{args.rife_v} |  fp16: {s_fp16}  |  Streams: {args.streams} |  Out format: {s_fmt}", flush=True)
    print(f"  Models: {args.models}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*62}\n", flush=True)

    # -- GPU validation --------------------------------------------
    if not gpu_exists(args.gpu_id):
        print(
            f"Error: GPU with index {args.gpu_id} not found.  "
            "Check nvidia-smi or pass a valid --gpu_id.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Reporter init (writes header + specs) ---------------------
    reporter = Reporter(args)
    reporter.write_header()

    # -- Run benchmarks --------------------------------------------
    results = run_benchmarks(args, on_result=reporter.append_run)

    # -- Finalize --------------------------------------------------
    reporter.finalize(results)

if __name__ == "__main__":
    main()
