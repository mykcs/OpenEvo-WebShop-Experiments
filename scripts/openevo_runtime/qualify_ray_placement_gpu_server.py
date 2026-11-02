#!/usr/bin/env python3
"""Opt-in real Ray + CUDA qualification on the authorized experiment-host GPU host.

This is an isolated engineering smoke, NEVER an automatic GPU reservation,
production scheduler, scientific experiment, or hand-back receipt. It uses the
existing pinned Ray 2.57.0 host runtime, a disposable local two-node Ray
cluster, and precisely pinned idle GPU5/GPU7 UUIDs. No Docker or training.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid


RAY_RUNTIME = "/workspace/control/ray-shadow-runtime-2.57.0"
SCHEMA = "openevo.ray.gpu-server-qualification.v1"
# Bound to the existing versioned 4-7 pool. These are NOT a GPU grant.
GPU = {
    5: "GPU-00000000-0000-0000-0000-000000000000",
    7: "GPU-00000000-0000-0000-0000-000000000000",
}


def assert_idle(indices: tuple[int, ...] = (5, 7)) -> None:
    output = subprocess.check_output(
        ["/usr/local/bin/gpu-who"], text=True, timeout=20
    )
    for index in indices:
        line = next(
            (s for s in output.splitlines() if s.startswith(f"GPU{index} |")),
            None,
        )
        if not line or "| idle | -" not in line:
            raise RuntimeError(f"GPU{index} not independently owner-idle: {line}")


def cuda_worker(index: int, expected: str) -> dict[str, object]:
    import ray

    # Every worker repeats attribution just before creating a CUDA context.
    assert_idle((index,))
    os.environ["CUDA_VISIBLE_DEVICES"] = expected
    lib = ctypes.CDLL("libcuda.so.1")

    def api(name: str, types: list[object]):
        fn = getattr(lib, name)
        fn.argtypes = types
        fn.restype = ctypes.c_int
        return fn

    def ck(name: str, code: int) -> None:
        if code:
            raise RuntimeError(f"{name}: CUDA Driver API returned {code}")

    ck("cuInit", api("cuInit", [ctypes.c_uint])(0))
    count = ctypes.c_int()
    ck("cuDeviceGetCount", api("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])(
        ctypes.byref(count)
    ))
    if count.value != 1:
        raise RuntimeError(f"CUDA mask exposed {count.value} devices")
    device = ctypes.c_int()
    ck("cuDeviceGet", api("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])(
        ctypes.byref(device), 0
    ))

    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    identity = CUuuid()
    ck("cuDeviceGetUuid", api("cuDeviceGetUuid", [
        ctypes.POINTER(CUuuid), ctypes.c_int,
    ])(ctypes.byref(identity), device.value))
    actual = "GPU-" + str(uuid.UUID(bytes=bytes(identity.bytes)))
    if actual != expected:
        raise RuntimeError(f"CUDA UUID drift: {actual} != {expected}")

    context = ctypes.c_void_p()
    ck("cuCtxCreate_v2", api("cuCtxCreate_v2", [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int,
    ])(ctypes.byref(context), 0, device.value))
    try:
        memory = ctypes.c_uint64()
        ck("cuMemAlloc_v2", api("cuMemAlloc_v2", [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ])(ctypes.byref(memory), 16 * 1024 * 1024))
        try:
            time.sleep(0.3)
        finally:
            ck("cuMemFree_v2", api("cuMemFree_v2", [
                ctypes.c_uint64,
            ])(memory.value))
    finally:
        ck("cuCtxDestroy_v2", api("cuCtxDestroy_v2", [
            ctypes.c_void_p,
        ])(context))
    return {
        "gpu_index": index,
        "verified_cuda_uuid": actual,
        "allocated_then_freed_bytes": 16 * 1024 * 1024,
        "ray_node_id": ray.get_runtime_context().get_node_id(),
    }


def run() -> dict[str, object]:
    if socket.gethostname().split(".")[0] != "experiment-host" or os.geteuid() != 0:
        raise RuntimeError("requires the authorized experiment-host root host")
    assert_idle()
    if not Path("/dev/shm").is_dir():
        raise RuntimeError("bounded /dev/shm temp space unavailable")
    if not Path(RAY_RUNTIME, "ray", "_raylet.so").is_file():
        raise RuntimeError("pinned Ray host runtime missing")
    os.environ["PYTHONPATH"] = RAY_RUNTIME + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.pop("RAY_ADDRESS", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, RAY_RUNTIME)
    import ray
    from ray.cluster_utils import Cluster
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if ray.__version__ != "2.57.0" or ray.is_initialized():
        raise RuntimeError("unexpected Ray version/active local client")
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="oe-", dir="/dev/shm") as root:
        work = Path(root, "w")
        work.mkdir()
        # Keep Ray's session/log tree OUTSIDE the working directory; otherwise
        # Ray may recursively package private files and its own log tree.
        os.chdir(work)
        os.environ["RAY_TMPDIR"] = str(Path(root, "r"))
        cluster = Cluster(shutdown_at_exit=True)
        groups = []
        try:
            for index in GPU:
                # OpenEVO's virtual GPU model: one Ray node per physical GPU,
                # zero Ray-native GPUs and exact custom index resources.
                cluster.add_node(
                    num_cpus=1, num_gpus=0,
                    resources={f"gpu_index_{index}": 1},
                    include_dashboard=False,
                    object_store_memory=84 * 1024 * 1024,
                )
            # Use Ray's discovered cluster address. Overriding node IP to
            # localhost on experiment-host causes GCS advertised-IP drift.
            ray.init(address=cluster.address, include_dashboard=False, log_to_driver=False)
            assert_idle()
            bundles = [
                {"CPU": 0.5, "gpu_index_5": 1},
                {"CPU": 0.5, "gpu_index_7": 1},
            ]
            gang = placement_group(bundles, strategy="STRICT_SPREAD")
            groups.append(gang)
            ray.get(gang.ready(), timeout=30)
            task = ray.remote(num_cpus=0.5)(cuda_worker)
            rows = ray.get([
                task.options(
                    resources={f"gpu_index_{index}": 1},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=gang, placement_group_bundle_index=i,
                    ),
                ).remote(index, GPU[index])
                for i, index in enumerate(GPU)
            ], timeout=40)
            if {r["gpu_index"] for r in rows} != set(GPU):
                raise RuntimeError("gang resource assignment drift")
            if len({r["ray_node_id"] for r in rows}) != 2:
                raise RuntimeError("STRICT_SPREAD did not use separate Ray nodes")
            waiting = placement_group([{"gpu_index_5": 1}], strategy="PACK")
            groups.append(waiting)
            if ray.wait([waiting.ready()], timeout=1)[0]:
                raise RuntimeError("Ray overbooked an already-reserved GPU5")
            remove_placement_group(gang)
            groups.remove(gang)
            ray.get(waiting.ready(), timeout=25)
            result = {
                "schema": SCHEMA,
                "status": "PASS",
                "host": "experiment-host",
                "ray_version": ray.__version__,
                "scheduler": "real_two_node_STRICT_SPREAD",
                "gpu_bindings": rows,
                "overbooking_blocked": True,
                "waiting_group_ready_after_release": True,
                "scientific_work_consumed": False,
                "gpu_handoff_verified": False,
            }
        finally:
            if ray.is_initialized():
                for group in groups:
                    try:
                        remove_placement_group(group)
                    except Exception:
                        pass
                ray.shutdown()
            cluster.shutdown()
            os.chdir(previous)
    # Tempdirs/Ray workers must be gone; post-live attribution is authoritative.
    assert_idle()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-ref", default="")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": SCHEMA, "status": "PLAN_ONLY", "host": "experiment-host",
            "gpu_indices": list(GPU), "gpu_handoff_verified": False,
        }, sort_keys=True))
        return 0
    if not re.fullmatch(r"[A-Za-z0-9:._/#-]{8,180}", args.authorization_ref):
        parser.error("--execute requires a non-secret authorization reference")
    try:
        result = run()
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA, "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "scientific_work_consumed": False,
            "gpu_handoff_verified": False,
        }, sort_keys=True))
        return 1
    result["authorization_ref"] = args.authorization_ref
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
