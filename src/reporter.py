"""Reporter — generates the Markdown benchmark report.

Design:
- The report is written incrementally: the header/specs are written once
  at the start, and each run's data is appended as soon as it finishes.
- The Summary table is rewritten at the end, once all runs are done,
  because it needs aggregate statistics.
- If a file with the chosen bench_name already exists, a numeric suffix
  ``_01``, ``_02``, … is added to avoid overwriting.
- All numeric aggregations (median, percentile, avg, max) are computed
  without numpy to keep dependencies minimal.
"""

from __future__ import annotations

import os
import platform
import subprocess
import re
from pathlib import Path
from typing import Any

from .parser import probe_clip, probe_source, get_vs_version
from .runner import RunResult
from .sampler import GpuSnapshot, query_gpu_specs


# =====================================================================
# Math helpers (no numpy)
# =====================================================================

def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _percentile(vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (same as numpy default)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(v: float | None, prec: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{prec}f}"


# =====================================================================
# Aggregate GPU samples for one run
# =====================================================================

def _agg_gpu(samples: list[GpuSnapshot]) -> dict[str, str]:
    """Aggregate GPU samples into summary strings for the All-Runs table."""
    na = "N/A"
    if not samples:
        keys = [
            "gpu_util_avg", "gpu_util_p99",
            "mem_util_avg", "mem_util_p99",
            "vram_avg_mb", "vram_max_mb",
            "pwr_avg_w", "pwr_max_w",
            "rxpci_avg_mb_s", "rxpci_max_mb_s",
            "txpci_avg_mb_s", "txpci_max_mb_s",
            "gpu_clock", "mem_clock",
        ]
        return {k: na for k in keys}

    gpu_u = [s.gpu_util for s in samples]
    mem_u = [s.mem_util for s in samples]
    vram  = [s.vram_mb for s in samples]
    pwr   = [s.power_w for s in samples]
    rx    = [s.rxpci_mb_s for s in samples]
    tx    = [s.txpci_mb_s for s in samples]
    gclk  = [float(s.gpu_clock) for s in samples]
    mclk  = [float(s.mem_clock) for s in samples]

    return {
        "gpu_util_avg":     _fmt(_avg(gpu_u), 1),
        "gpu_util_p99":     _fmt(_percentile(gpu_u, 99), 1),
        "mem_util_avg":     _fmt(_avg(mem_u), 1),
        "mem_util_p99":     _fmt(_percentile(mem_u, 99), 1),
        "vram_avg_mb":      _fmt(_avg(vram), 0),
        "vram_max_mb":      _fmt(max(vram), 0),
        "pwr_avg_w":        _fmt(_avg(pwr), 1),
        "pwr_max_w":        _fmt(max(pwr), 1),
        "rxpci_avg_mb_s":   _fmt(_avg(rx), 1),
        "rxpci_max_mb_s":   _fmt(max(rx), 1),
        "txpci_avg_mb_s":   _fmt(_avg(tx), 1),
        "txpci_max_mb_s":   _fmt(max(tx), 1),
        "gpu_clock":        _fmt(_median(gclk), 0),
        "mem_clock":        _fmt(_median(mclk), 0),
    }


# =====================================================================
# File-name deduplication
# =====================================================================

def _unique_path(directory: str, base: str, ext: str = ".md") -> Path:
    """Return a path that does not yet exist, adding _01, _02 … if needed."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    candidate = d / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        candidate = d / f"{base}_{i:02d}{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Too many existing benchmark files")


# =====================================================================
# CPU info (cross-platform, best-effort)
# =====================================================================

def _cpu_info() -> str:
    """Return a short CPU description.  Falls back to platform.processor()."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
            cores = os.cpu_count() or "?"
            return f"{name.strip()} ┆ {cores} cores"
        else:
            # Linux: /proc/cpuinfo
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        cores = os.cpu_count() or "?"
                        return f"{name} ┆ {cores} cores"
    except Exception:
        pass
    proc = platform.processor() or "Unknown"
    cores = os.cpu_count() or "?"
    return f"{proc} ┆ {cores} cores"


def _os_info() -> str:
    """E.g. 'Windows 11 10.0.26100' or 'Linux 6.5.0-arch1-1'."""
    system = platform.system()
    ver = platform.version()
    release = platform.release()
    if system == "Windows":
        # platform.release() gives '10' even on Win 11; check build.
        build = int(ver.split(".")[-1]) if ver else 0
        win_ver = "11" if build >= 22000 else "10"
        return f"Windows {win_ver} {ver}"
    return f"{system} {release}"


# =====================================================================
# Reporter class
# =====================================================================

class Reporter:
    """Builds and writes the Markdown report file.

    Usage::

        reporter = Reporter(args)
        reporter.write_header()            # once, before runs
        reporter.append_run(run_result)    # after each run
        reporter.finalize(all_results)     # once, after all runs
    """

    def __init__(self, args):
        self.args = args
        self.path = _unique_path(args.output_dir, args.bench_name)
        self.log_path = _unique_path((args.output_dir + "/logs"), args.bench_name)
        self._errors: list[tuple[int, int, str]] = []  # (model, run_id, msg)

    # -- header ----------------------------------------------------

    def write_header(self) -> None:
        """Write everything above the Summary table."""
        a = self.args

        gpu = query_gpu_specs(a.gpu_id)
        vs_ver = get_vs_version(a.vspipe_path)
        source_info = probe_source(a.source)
        clip_info = probe_clip(a.script, a.source, a.models, a.vspipe_path)

        src_str = " ┆ ".join(f"{k}: {v}" for k, v in source_info.items())
        clip_str = " ┆ ".join(f"{k}: {v}" for k, v in clip_info.items())

        lines = [
            f"# {a.bench_name}\n",
            "## Specs\n",
            f"| Name                     | Info                      |",
            f"| ------------------------ | ------------------------- |",
            f"| **OS:**                  | {_os_info()}              |",
            f"| **GPU model:**           | {gpu['gpu_model']}        |",
            f"| **Driver version:**      | {gpu['driver_version']}   |",
            f"| **CUDA version:**        | {gpu['cuda_version']}     |",
            f"| **TDP:**                 | {gpu['tdp_w']} W          |",
            f"| **Total VRAM:**          | {gpu['total_vram_mb']} MB |",
            f"| **Used VRAM:**           | {gpu['used_vram_mb']} MB  |",
            f"| **Free VRAM:**           | {gpu['free_vram_mb']} MB  |",
            f"| **CPU:**                 | {_cpu_info()}             |",
            f"| **Bench Mode:**          | {a.mode}                  |",
            f"| **VapourSynth version:** | {vs_ver}                  |",
            f"| **RIFE:**                | v{a.rife_v}               |",
            f"| **Source:**              | {src_str}                 |",
            f"| **Clip:**                | {clip_str}                |",
            "",
            "## Summary",
            "",
            f"**Models tested:** {', '.join(str(m) for m in a.models)}  ",
            f"**Total tests:** {a.tests * len(a.models)}  ",
            f"**Total errors:** {{TOTAL_ERRORS}}  ",  # placeholder, filled in finalize
            "",
        ]
        self.path.write_text("\n".join(lines), encoding="utf-8")
        # print(f"[reporter] Report: {self.path}", flush=True)

    # -- incremental run append ------------------------------------

    def append_run(self, r: RunResult) -> None:
        """Append one run's data to the file.  Called after every run."""
        if r.is_warmup:
            return  # warmup runs are not recorded in the report

        if not r.ok:
            self._errors.append((r.model, r.run_id, r.error))

        # We don't write the All-Runs table row-by-row because the
        # table header must come first.  Instead we accumulate and
        # write everything in finalize().  But we *do* write errors
        # immediately so they are saved even if the process is killed.
        if not r.ok:
            with open(self.path, "a", encoding="utf-8") as f:
                # Append error immediately (will also appear in Errors section).
                pass  # errors are collected and written in finalize()

    # -- finalize --------------------------------------------------

    def finalize(self, all_results: list[RunResult]) -> None:
        """Rewrite the file with complete Summary, All Runs, and Errors."""
        a = self.args
        # Filter to test-only (non-warmup) results.
        test_runs = [r for r in all_results if not r.is_warmup]
        ok_runs = [r for r in test_runs if r.ok]
        fail_runs = [r for r in test_runs if not r.ok]

        # Read existing header.
        text = self.path.read_text(encoding="utf-8")
        # Replace error placeholder.
        text = text.replace("{TOTAL_ERRORS}", str(len(fail_runs)))

        lines = text.rstrip("\n").split("\n")

        # -- Summary table ----------------------------------------
        lines.append("")
        lines.append(
            "| Model | FPS (avg┆median┆min) "
            "| Time s (avg┆mdn┆max) "
            "| GPU avg % | VRAM MB (mdn┆max) "
            "| PWR W (mdn┆max) "
            "| rxpci_max MB/s | txpci_max MB/s "
            "| Frames | Runs (tot┆succ) |"
        )
        lines.append("|" + " :---: |" * 10)

        for model in a.models:
            m_runs = [r for r in ok_runs if r.model == model]
            total = len([r for r in test_runs if r.model == model])
            succ = len(m_runs)
            if not m_runs:
                lines.append(f"| {model} |" + " N/A |" * 8 + f" {total}/{succ} |")
                continue

            fps_list = [r.fps for r in m_runs if r.fps is not None]
            time_list = [r.time_s for r in m_runs if r.time_s is not None]
            frames_list = [r.frames for r in m_runs if r.frames is not None]

            # GPU aggregates across all samples from all successful runs.
            all_samples: list[GpuSnapshot] = []
            for r in m_runs:
                all_samples.extend(r.gpu_samples)
            g = _agg_gpu(all_samples)

            lines.append(
                f"| {model} "
                f"| {_fmt(_avg(fps_list))}┆{_fmt(_median(fps_list))}┆{_fmt(min(fps_list) if fps_list else None)} "
                f"| {_fmt(_avg(time_list))}┆{_fmt(_median(time_list))}┆{_fmt(max(time_list) if time_list else None)} "
                f"| {g['gpu_util_avg']} "
                f"| {_fmt(_median([s.vram_mb for s in all_samples]) if all_samples else None, 0)} ┆ {g['vram_max_mb']} "
                f"| {_fmt(_median([s.power_w for s in all_samples]) if all_samples else None, 1)} ┆ {g['pwr_max_w']} "
                f"| {g['rxpci_max_mb_s']} "
                f"| {g['txpci_max_mb_s']} "
                f"| {_fmt(_median([float(f) for f in frames_list]) if frames_list else None, 0)} "
                f"| {total}┆{succ} |"
            )

        # -- All Runs table ---------------------------------------
        lines.append("")
        lines.append("## All Runs")
        lines.append("")
        lines.append(
            "| Model | Run | Frames | FPS | Time s "
            "| GPU util % (avg┆p99) "
            "| MEM util % (avg┆p99) "
            "| VRAM Mb (avg┆max) "
            "| PWR W (avg┆max) "
            "| rxpci Mb/s (avg┆max) "
            "| txpci Mb/s (avg┆max) "
            "| gpu_clock | mem_clock |"
        )
        lines.append("|" + " :---: |" * 13)

        for r in test_runs:
            g = _agg_gpu(r.gpu_samples)
            status = "" if r.ok else " ❌"
            lines.append(
                f"| {r.model} "
                f"| {r.run_id}{status} "
                f"| {r.frames if r.frames is not None else 'N/A'} "
                f"| {_fmt(r.fps)} "
                f"| {_fmt(r.time_s)} "
                f"| {g['gpu_util_avg']} ┆ {g['gpu_util_p99']} "
                f"| {g['mem_util_avg']} ┆ {g['mem_util_p99']} "
                f"| {g['vram_avg_mb']} ┆ {g['vram_max_mb']} "
                f"| {g['pwr_avg_w']} ┆ {g['pwr_max_w']} "
                f"| {g['rxpci_avg_mb_s']} ┆ {g['rxpci_max_mb_s']} "
                f"| {g['txpci_avg_mb_s']} ┆ {g['txpci_max_mb_s']} "
                f"| {g['gpu_clock']} "
                f"| {g['mem_clock']} |"
            )

        # -- Errors section ---------------------------------------
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        if not fail_runs:
            lines.append("No errors.")
        else:
            err_report = []
            for r in fail_runs:
                err_title = f"**Error in model:** `{r.model}` **test:** `{r.run_id}`"
                err_report.append("")
                err_report.append(f"## {err_title}")
                err_report.append("")
                err_report.append(f"```bash\n{r.error}\n```")

                title_md = err_title.lstrip('#').strip()
                title_md = title_md.lower()
                title_md = title_md.replace(" ", "-")
                title_md = re.sub(r'[^a-z0-9\-]', '', title_md)
                title_md = re.sub(r'-+', '-', title_md)

                rel_path = os.path.relpath(self.log_path, os.path.dirname(self.path))
                rel_path = rel_path.replace("\\", "/")

                lines.append(f"{err_title} | **log:** [log]({rel_path}#{title_md})  ")

        # -- ArgLine section ------------------------------------
        arg_line = " ┆ ".join(f"**{k}**: `{v}`" for k, v in vars(self.args).items())
        lines.append(
            "\n## Runtime Arguments:\n\n"
            f"{arg_line}"
        )
        lines.append("")

        if fail_runs:
            self.log_path.write_text("\n".join(err_report), encoding="utf-8")

        self.path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n{'='*62}", flush=True)
        print(f"  Done. Markdown report saved to: {self.path}", flush=True)
        print(f"  Total: {len(test_runs)}  OK: {len(ok_runs)}  Failed: {len(fail_runs)}", flush=True)
        print(f"{'='*62}", flush=True)
