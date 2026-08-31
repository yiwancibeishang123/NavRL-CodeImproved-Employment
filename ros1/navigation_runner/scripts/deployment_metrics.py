#!/usr/bin/env python3
"""Metrics helpers shared by simulation and physical deployment runs.

The navigation node writes one JSON object per line so a run can be inspected
while it is still active.  Timing has two meanings which must remain separate:
``inference_wall_ms`` is the caller-observed latency, while
``inference_device_ms`` is the CUDA kernel time when CUDA is used.
"""

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path, flush_every=1):
        self.path = Path(path) if path else None
        self.flush_every = max(1, int(flush_every))
        self._count = 0
        self._file = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")

    def write(self, payload):
        if self._file is None:
            return
        self._file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self._count += 1
        if self._count % self.flush_every == 0:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def vector_list(vector):
    return [round(float(value), 6) for value in vector]


def _rss_mib():
    """Return current resident memory on Linux without adding psutil."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as file_obj:
            resident_pages = int(file_obj.read().split()[1])
        return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0), 3)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def runtime_snapshot(torch_module=None, device=None):
    """Collect cheap process/device measurements suitable for control logging."""
    snapshot = {
        "process_rss_mib": _rss_mib(),
        "process_max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
    }
    if torch_module is None:
        return snapshot

    try:
        resolved_device = torch_module.device(device) if device is not None else None
        use_cuda = (
            torch_module.cuda.is_available()
            and resolved_device is not None
            and resolved_device.type == "cuda"
        )
        if use_cuda:
            index = resolved_device.index
            snapshot.update({
                "cuda_device": torch_module.cuda.get_device_name(index),
                "cuda_allocated_mib": round(torch_module.cuda.memory_allocated(index) / (1024.0 * 1024.0), 3),
                "cuda_reserved_mib": round(torch_module.cuda.memory_reserved(index) / (1024.0 * 1024.0), 3),
            })
    except Exception:
        # Metrics must never prevent a safety hold or a control command.
        pass
    return snapshot


def runtime_metadata(torch_module=None, device=None):
    """Return immutable run metadata, recorded once at node startup."""
    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
    }
    if torch_module is None:
        return metadata
    try:
        metadata["torch"] = str(torch_module.__version__)
        metadata["cuda_available"] = bool(torch_module.cuda.is_available())
        if device is not None:
            resolved_device = torch_module.device(device)
            metadata["device"] = str(resolved_device)
            if resolved_device.type == "cuda" and torch_module.cuda.is_available():
                metadata["cuda_device"] = torch_module.cuda.get_device_name(resolved_device.index)
                metadata["cuda_runtime"] = str(torch_module.version.cuda)
    except Exception:
        pass
    return metadata


def timed_call(function, torch_module=None, device=None):
    """Execute ``function`` and return its result plus wall/device timing.

    CUDA calls are asynchronous.  Synchronizing around the measured call is
    intentional: the wall time then represents the latency a control loop has
    to budget, while CUDA events expose the device-only portion separately.
    """
    use_cuda = False
    cuda_device = None
    if torch_module is not None:
        try:
            cuda_device = torch_module.device(device) if device is not None else None
            use_cuda = (
                torch_module.cuda.is_available()
                and cuda_device is not None
                and cuda_device.type == "cuda"
            )
        except Exception:
            use_cuda = False

    start_event = end_event = None
    if use_cuda:
        torch_module.cuda.synchronize(cuda_device)
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)
        start_event.record()
    start = time.perf_counter()
    result = function()
    wall_ms = (time.perf_counter() - start) * 1000.0
    device_ms = None
    if use_cuda:
        end_event.record()
        torch_module.cuda.synchronize(cuda_device)
        device_ms = float(start_event.elapsed_time(end_event))
    return result, wall_ms, device_ms
