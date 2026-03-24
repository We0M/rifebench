"""GPU sampler — collects metrics via pynvml at a fixed interval.

Runs in a daemon thread so it dies automatically if the main process exits.
Uses a simple list-of-dicts accumulator; the thread only appends, the main
thread only reads after join(), so no lock is needed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


# -- Snapshot dataclass ------------------------------------------------

@dataclass
class GpuSnapshot:
    """Single point-in-time GPU reading."""
    gpu_util: int        # %  (0-100)
    mem_util: int        # %  (0-100)
    power_w: float       # watts
    vram_mb: float       # used VRAM in MiB
    rxpci_mb_s: float    # PCIe RX throughput MiB/s
    txpci_mb_s: float    # PCIe TX throughput MiB/s
    gpu_clock: int       # MHz
    mem_clock: int       # MHz


# -- Public helpers for one-off GPU info queries ----------------------

def get_gpu_handle(gpu_id: int):
    """Return an NVML handle for *gpu_id*, or None on failure."""
    if not _HAS_NVML:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if gpu_id >= count:
            return None
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    except pynvml.NVMLError:
        return None


def gpu_exists(gpu_id: int) -> bool:
    """Check whether *gpu_id* is reachable via NVML."""
    if not _HAS_NVML:
        return False
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        return gpu_id < count
    except pynvml.NVMLError:
        return False


def query_gpu_specs(gpu_id: int) -> dict[str, str]:
    """Return a dict with static GPU info for the report header.

    Keys: gpu_model, driver_version, cuda_version, tdp_w,
          total_vram_mb, used_vram_mb, free_vram_mb.
    All values are strings; on failure they become 'N/A'.
    """
    na = "N/A"
    info: dict[str, str] = {
        "gpu_model": na, "driver_version": na, "cuda_version": na,
        "tdp_w": na, "total_vram_mb": na, "used_vram_mb": na,
        "free_vram_mb": na,
    }
    handle = get_gpu_handle(gpu_id)
    if handle is None:
        return info
    try:
        info["gpu_model"] = pynvml.nvmlDeviceGetName(handle)
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        # NVML returns CUDA version as int, e.g. 12020 -> "12.2"
        cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        info["cuda_version"] = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
        # Power limit (milliwatts -> watts).
        pl = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        info["tdp_w"] = f"{pl / 1000:.0f}"
        # VRAM.
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = mem.total / (1024 ** 2)
        used = mem.used / (1024 ** 2)
        info["total_vram_mb"] = f"{total:.0f}"
        info["used_vram_mb"] = f"{used:.0f}"
        info["free_vram_mb"] = f"{total - used:.0f}"
    except pynvml.NVMLError:
        pass
    return info


# -- Sampler thread ---------------------------------------------------

def _sample_once(handle) -> GpuSnapshot | None:
    """Take one GPU reading.  Returns None on any NVML error."""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # PCIe throughput — returns KB/s; convert to MiB/s.
        # NVML_PCIE_UTIL_TX_BYTES = 0, NVML_PCIE_UTIL_RX_BYTES = 1
        try:
            rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
            tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
        except pynvml.NVMLError:
            rx = tx = 0.0
        gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        return GpuSnapshot(
            gpu_util=util.gpu,
            mem_util=util.memory,
            power_w=power,
            vram_mb=mem.used / (1024 ** 2),
            rxpci_mb_s=rx,
            txpci_mb_s=tx,
            gpu_clock=gpu_clock,
            mem_clock=mem_clock,
        )
    except pynvml.NVMLError:
        return None


class GpuSampler:
    """Collects GPU snapshots in a background thread.

    Usage::

        sampler = GpuSampler(gpu_id=0, period_ms=100)
        sampler.start()
        # … run benchmark …
        samples = sampler.stop()   # list[GpuSnapshot]
    """

    def __init__(self, gpu_id: int, period_ms: int = 100):
        self._gpu_id = gpu_id
        self._period = period_ms / 1000.0  # seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[GpuSnapshot] = []

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[GpuSnapshot]:
        """Signal the thread and wait for it; return collected samples."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return list(self._samples)

    # -- internal -----------------------------------------------------

    def _loop(self) -> None:
        handle = get_gpu_handle(self._gpu_id)
        if handle is None:
            return
        while not self._stop_event.is_set():
            snap = _sample_once(handle)
            if snap is not None:
                self._samples.append(snap)
            # Event-based sleep so stop() wakes us immediately.
            self._stop_event.wait(self._period)
