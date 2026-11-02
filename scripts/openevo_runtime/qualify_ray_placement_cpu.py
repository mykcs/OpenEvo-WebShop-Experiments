#!/usr/bin/env python3
"""Real Ray placement-group qualification without GPUs or an existing cluster.

Run only in an isolated, disposable Python environment with pinned ray==2.57.0.
This script starts a new LOCAL, CPU-only Ray cluster. It does not connect to a
host Ray service, import the project's Docker/GPU launchers, or create scientific
receipts. PASS qualifies Ray's basic gang reservation/release semantics only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


RAY_VERSION = "2.57.0"
SCHEMA = "openevo.ray.cpu-placement-qualification.v1"


def check(timeout_seconds: int = 30) -> dict[str, object]:
    # Disable any inherited cluster auto-discovery or GPU visibility.
    os.environ.pop("RAY_ADDRESS", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import ray
    from ray.util.placement_group import (
        get_placement_group, placement_group, remove_placement_group,
    )
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    # Import the *existing production broker helper*, so this is a genuine
    # integration qualification rather than a generic standalone Ray demo.
    import sys
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.openevo_webshop import ceiling1_ray_placement_backend_202609042025 as broker

    if ray.__version__ != RAY_VERSION:
        raise RuntimeError(
            f"Ray runtime drift: expected {RAY_VERSION}, got {ray.__version__}"
        )
    if ray.is_initialized():
        raise RuntimeError("Refuse an existing Ray client/cluster; a local sandbox is required")

    groups = []
    original_cwd = Path.cwd()
    # Keep AF_UNIX Ray sockets below the macOS 103-byte path limit.
    with tempfile.TemporaryDirectory(prefix="oe-ray-", dir="/tmp") as root:
        try:
            # Ray may auto-package the working directory. Keep the EMPTY
            # published workspace separate from Ray's log/session directory:
            # packaging the latter recursively can hang or grow without bound.
            ray_session = str(Path(root) / "r")
            empty_workspace = Path(root) / "empty-workspace"
            empty_workspace.mkdir()
            os.chdir(empty_workspace)
            ray.init(
                address="local", num_cpus=2, num_gpus=0,
                include_dashboard=False, log_to_driver=False, _temp_dir=ray_session,
            )
            if any(key.startswith("GPU") and value > 0
                   for key, value in ray.cluster_resources().items()):
                raise RuntimeError("CPU-only Ray cluster unexpectedly advertises GPUs")

            primary_name = "oe-local-cpu-qualification"
            pg = placement_group(
                [{"CPU": 1}, {"CPU": 1}], strategy="STRICT_PACK",
                name=primary_name,
            )
            groups.append(pg)
            ray.get(pg.ready(), timeout=timeout_seconds)

            @ray.remote(num_cpus=1)
            def node_id() -> str:
                return ray.get_runtime_context().get_node_id()

            ids = ray.get(
                [node_id.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=i
                    )
                ).remote() for i in range(2)],
                timeout=timeout_seconds,
            )
            if len(ids) != 2 or ids[0] != ids[1]:
                raise RuntimeError("Ray placement group failed same-node bundle test")

            # All two CPUs are reserved by pg; another one-CPU group must not
            # become ready until pg is released.
            pending = placement_group([{"CPU": 1}], strategy="PACK")
            groups.append(pending)
            ready, _ = ray.wait([pending.ready()], timeout=1)
            if ready:
                raise RuntimeError("Ray allowed an overbooked placement group")

            if not broker.remove_pg_quiet(
                get_placement_group, remove_placement_group, primary_name
            ):
                raise RuntimeError("existing broker cannot request Ray PG removal")
            groups.remove(pg)
            ray.get(pending.ready(), timeout=timeout_seconds)
            if not broker.remove_pg_quiet(
                get_placement_group, remove_placement_group, primary_name
            ):
                raise RuntimeError("Ray PG idempotent cleanup failed")

            return {
                "schema": SCHEMA,
                "status": "PASS",
                "ray_version": ray.__version__,
                "execution": "isolated-local-ray-cpu-only",
                "logical_gpu_resources": 0,
                "same_node_two_bundles": True,
                "overbooking_blocked": True,
                "ready_after_release": True,
                "broker_backend_cleanup_exercised": True,
                "scientific_work_consumed": False,
                "host_gpu_release_verified": False,
            }
        finally:
            for group in groups:
                try:
                    remove_placement_group(group)
                except Exception:
                    pass
            if ray.is_initialized():
                ray.shutdown()
            os.chdir(original_cwd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 5 <= args.timeout_seconds <= 120:
        parser.error("--timeout-seconds must be between 5 and 120")
    try:
        receipt = check(args.timeout_seconds)
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA, "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "scientific_work_consumed": False,
            "host_gpu_release_verified": False,
        }, sort_keys=True))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
