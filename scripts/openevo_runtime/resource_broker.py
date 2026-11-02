"""Single Ray placement API for prospective OpenEVO GPU work.

Science decides whether an OperationIntent is ready.  This module only translates a
validated ResourceRequest into the repository's existing Ray placement backend.  It
is prospective runtime code: it does not hot-migrate the active formal campaign and
it grants neither scientific nor topology authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

from scripts.openevo_runtime.legacy_result_adapter import classify_legacy_process_exit
from scripts.openevo_runtime.qualified_topology import (
    QualifiedTopologyError,
    load_qualified_topology_registry,
    validate_request_topology,
    validate_selection_topology,
)
from scripts.openevo_runtime.resource_records import (
    build_resource_blocked,
    build_resource_lease,
    validate_resource_request,
)
from scripts.openevo_runtime.result_codes import OperationResult, OperationStatus, ReasonCode, RetryClass
from scripts.openevo_webshop import ceiling1_ray_placement_backend_202609042025 as ray_backend

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py"


class ResourceBrokerError(RuntimeError):
    """A prospective broker request disagrees with its frozen resource inputs."""


@dataclass(frozen=True)
class RayBrokerBinding:
    config_path: Path
    state_root: Path
    workload: str
    profile_registry_path: Path | None = None
    topology_registry_path: Path | None = None
    python_executable: str = sys.executable
    backend_path: Path = BACKEND


@dataclass(frozen=True)
class BrokerDecision:
    outcome: OperationResult
    request_id: str
    selection_receipt: Mapping[str, Any] | None = None
    resource_blocked: Mapping[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self.outcome.status is OperationStatus.READY


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceBrokerError(f"unreadable broker input: {path}") from exc
    if not isinstance(value, dict):
        raise ResourceBrokerError(f"broker input must be an object: {path}")
    return value


class RayResourceBroker:
    """Typed facade over the existing Ray placement/reservation backend."""

    def __init__(
        self,
        binding: RayBrokerBinding,
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.binding = binding
        self.command_runner = command_runner
        self.config_path = binding.config_path.resolve()
        self.state_root = binding.state_root.resolve()
        self.registry_path = binding.profile_registry_path.resolve() if binding.profile_registry_path else None
        self.topology_registry_path = binding.topology_registry_path.resolve() if binding.topology_registry_path else None
        self.backend_path = binding.backend_path.resolve()
        self._config, self._registry, self._profile, self._topology_registry = self._load_backend_inputs()

    def _load_backend_inputs(self) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
        if not self.config_path.is_file() or not self.backend_path.is_file():
            raise ResourceBrokerError("Ray backend/config is missing")
        config = _load_json(self.config_path)
        if config.get("schema") != "openevo.ceiling1-ray-placement-backend.202609042025":
            raise ResourceBrokerError("Ray base config schema drift")
        registry = None
        if self.registry_path is not None:
            registry = ray_backend.load_profile_registry(self.registry_path, _sha(self.config_path))
        try:
            profile = ray_backend.profile(config, self.binding.workload, registry)
        except RuntimeError as exc:
            raise ResourceBrokerError(str(exc)) from exc
        if self.topology_registry_path is None:
            raise ResourceBrokerError("qualified topology registry is required")
        try:
            topology_registry = load_qualified_topology_registry(
                self.topology_registry_path,
                base_config_sha256=_sha(self.config_path),
                profile_registry_sha256=registry.get("_sha256") if registry else None,
            )
        except QualifiedTopologyError as exc:
            raise ResourceBrokerError(str(exc)) from exc
        return config, registry, profile, topology_registry

    def _validate_request_against_profile(self, request: Mapping[str, Any]) -> None:
        validate_resource_request(request)
        gpu = request["envelope"]["gpu_scope"]
        count = int(gpu["gpu_count"])
        minimum = int(self._profile["min_gpus"])
        maximum = int(self._profile["max_gpus"])
        if not minimum <= count <= maximum:
            raise ResourceBrokerError("request gpu_count outside frozen workload profile")
        if int(gpu["vram_mib_per_gpu"]) != int(self._profile["vram_request_mib"]):
            raise ResourceBrokerError("request VRAM disagrees with frozen workload profile")
        managed = set(map(int, self._config["managed_gpus"]))
        preferred = set(map(int, request["preferred_gpu_indices"]))
        if not preferred.issubset(managed):
            raise ResourceBrokerError("preferred GPU set contains unmanaged GPU")
        if int(gpu["process_slots_per_gpu"]) != 1:
            raise ResourceBrokerError("current Ray backend consumes exactly one process slot per GPU")
        try:
            validate_request_topology(self._topology_registry, request, workload=self.binding.workload)
        except QualifiedTopologyError as exc:
            raise ResourceBrokerError(str(exc)) from exc
        host = request["envelope"]["host_scope"]
        if int(host["ram_mib"]) or int(host["cpu_slots"]):
            raise ResourceBrokerError("host-global hard admission is not implemented by this Ray backend yet")

    def _validate_selection(self, request: Mapping[str, Any], selection_path: Path) -> dict[str, Any]:
        doc = _load_json(selection_path)
        if doc.get("schema") != "openevo.ceiling1-ray-selection.202609042025":
            raise ResourceBrokerError("Ray selection schema drift")
        if doc.get("workload") != self.binding.workload:
            raise ResourceBrokerError("Ray selection workload drift")
        if doc.get("config_sha256") != _sha(self.config_path):
            raise ResourceBrokerError("Ray selection base-config drift")
        expected_registry = self._registry.get("_sha256") if self._registry else None
        if doc.get("profile_registry_sha256") != expected_registry:
            raise ResourceBrokerError("Ray selection profile-registry drift")
        selected = list(map(int, doc.get("selected_gpus") or []))
        expected_count = int(request["envelope"]["gpu_scope"]["gpu_count"])
        if len(selected) != expected_count:
            raise ResourceBrokerError("Ray selection silently changed requested GPU count")
        if not set(selected).issubset(set(request["preferred_gpu_indices"])):
            raise ResourceBrokerError("Ray selection escaped frozen preferred GPU set")
        try:
            topology = validate_request_topology(self._topology_registry, request, workload=self.binding.workload)
            validate_selection_topology(topology, selected)
        except QualifiedTopologyError as exc:
            raise ResourceBrokerError(str(exc)) from exc
        if topology["placement_mode"] == "GANG_ALL_OR_NOTHING":
            if len(selected) <= 1:
                raise ResourceBrokerError("gang topology must reserve more than one GPU as one placement group")
            if not isinstance(doc.get("placement_group_name"), str) or not doc["placement_group_name"]:
                raise ResourceBrokerError("gang topology requires an immutable Ray placement-group identity")
            if not isinstance(doc.get("reservation_id"), str) or not doc["reservation_id"]:
                raise ResourceBrokerError("gang topology requires an immutable Ray reservation identity")
            if self._config.get("rollout_strategy") != "STRICT_SPREAD":
                raise ResourceBrokerError("gang rollout placement strategy drift")
        uuids = doc.get("selected_gpu_uuids") or []
        if len(uuids) != len(selected) or any(not isinstance(value, str) for value in uuids):
            raise ResourceBrokerError("Ray selection GPU UUID vector drift")
        if int(doc.get("vram_request_mib_per_gpu", -1)) != int(self._profile["vram_request_mib"]):
            raise ResourceBrokerError("Ray selection VRAM profile drift")
        live_gate = doc.get("live_gate") or {}
        for index, uuid in zip(selected, uuids):
            row = live_gate.get(str(index))
            if not isinstance(row, Mapping) or row.get("pass") is not True:
                raise ResourceBrokerError(f"GPU{index} missing PASS live gate")
            if row.get("uuid") != uuid:
                raise ResourceBrokerError(f"GPU{index} UUID/live-gate disagreement")
        return doc

    def _select_command(self, request: Mapping[str, Any], selection_path: Path) -> list[str]:
        count = int(request["envelope"]["gpu_scope"]["gpu_count"])
        cmd = [
            self.binding.python_executable,
            str(self.backend_path),
            "--config", str(self.config_path),
            "--state-root", str(self.state_root),
        ]
        if self.registry_path is not None:
            cmd += ["--profile-registry", str(self.registry_path)]
        cmd += [
            "select", "--workload", self.binding.workload,
            "--count", str(count), "--exact-count",
            "--output", str(selection_path.resolve()),
        ]
        return cmd

    def request_placement(self, request: Mapping[str, Any], *, selection_path: Path) -> BrokerDecision:
        """Reserve exact Ray placement or return a typed resource wait.

        Existing immutable selection receipts are adopted instead of reselecting.  A
        backend wait is translated at the legacy compatibility edge into OperationResult.
        """
        self._validate_request_against_profile(request)
        selection_path = selection_path.resolve()
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        if not selection_path.is_file():
            completed = self.command_runner(
                self._select_command(request, selection_path),
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                outcome = classify_legacy_process_exit(
                    operation_id=str(request["operation_id"]),
                    returncode=int(completed.returncode),
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                )
                blocked = None
                if outcome.retry_class is RetryClass.WAIT_RESOURCE:
                    blocked = build_resource_blocked(request, block_class="PLACEMENT_UNAVAILABLE")
                return BrokerDecision(outcome=outcome, request_id=str(request["request_id"]), resource_blocked=blocked)
        selection = self._validate_selection(request, selection_path)
        outcome = OperationResult(
            status=OperationStatus.READY,
            reason_code=ReasonCode.NONE,
            operation_id=str(request["operation_id"]),
            scientific_state_mutated=False,
            side_effect_materialized=True,
            message="qualified resource request has an exact Ray reservation",
            details={
                "resource_only": True,
                "selection_sha256": _sha(selection_path),
                "reservation_id": selection.get("reservation_id"),
                "placement_group_name": selection.get("placement_group_name"),
                "scientific_authority_granted": False,
            },
        )
        return BrokerDecision(
            outcome=outcome,
            request_id=str(request["request_id"]),
            selection_receipt=selection,
        )

    def _bind_backend_lease_receipt(
        self,
        request: Mapping[str, Any],
        *,
        selection_path: Path,
        receipt_path: Path,
        recovered: bool,
    ) -> dict[str, Any]:
        """Convert one exact backend launch/recovery receipt into a typed lease."""
        self._validate_request_against_profile(request)
        selection = self._validate_selection(request, selection_path.resolve())
        receipt_path = receipt_path.resolve()
        launch = _load_json(receipt_path)
        if recovered:
            if launch.get("schema") != "openevo.ceiling1-ray-launch-recovery.202609042102" or launch.get("status") != "RECOVERED_RUNNING_LEASE":
                raise ResourceBrokerError("Ray recovery receipt is not a canonical recovered running lease")
            lease = launch.get("lease")
            if not isinstance(lease, Mapping) or lease.get("recovered_from_launching_reservation") is not True:
                raise ResourceBrokerError("Ray recovery receipt missing recovered lease identity")
            live_gate = selection.get("live_gate")
        else:
            if launch.get("schema") != "openevo.ceiling1-ray-launch.202609042025" or launch.get("status") != "RUNNING":
                raise ResourceBrokerError("Ray launch receipt is not a canonical running lease")
            if launch.get("selection") != selection:
                raise ResourceBrokerError("Ray launch receipt selection/readback drift")
            lease = launch.get("lease")
            live_gate = launch.get("live_gate")
            if not isinstance(lease, Mapping) or not isinstance(live_gate, Mapping):
                raise ResourceBrokerError("Ray launch receipt missing lease/live UUID readback")
        if not isinstance(live_gate, Mapping):
            raise ResourceBrokerError("Ray lease binding is missing UUID readback evidence")
        selected = list(map(int, selection["selected_gpus"]))
        if list(map(int, lease.get("gpus") or [])) != selected:
            raise ResourceBrokerError("Ray launch lease GPU indices drift from selection")
        if lease.get("workload") != self.binding.workload:
            raise ResourceBrokerError("Ray launch lease workload drift")
        if lease.get("placement_group_name") != selection.get("placement_group_name"):
            raise ResourceBrokerError("Ray launch lease placement-group drift")
        if lease.get("reservation_id") != selection.get("reservation_id"):
            raise ResourceBrokerError("Ray launch lease reservation drift")
        if int(lease.get("vram_request_mib_per_gpu", -1)) != int(self._profile["vram_request_mib"]):
            raise ResourceBrokerError("Ray launch lease VRAM profile drift")
        selected_rows: list[dict[str, Any]] = []
        for index, selection_uuid in zip(selected, selection["selected_gpu_uuids"]):
            expected_uuid = (self._config.get("managed_gpu_uuids") or {}).get(str(index))
            live_row = live_gate.get(str(index))
            if expected_uuid != selection_uuid:
                raise ResourceBrokerError(f"GPU{index} selection UUID differs from frozen physical identity")
            if not isinstance(live_row, Mapping) or live_row.get("pass") is not True:
                raise ResourceBrokerError(f"GPU{index} lease binding missing PASS UUID readback")
            if live_row.get("uuid") != expected_uuid or live_row.get("expected_uuid") not in (None, expected_uuid):
                raise ResourceBrokerError(f"GPU{index} launch live UUID readback drift")
            selected_rows.append({"index": index, "uuid": expected_uuid})
        return build_resource_lease(
            request,
            selected_gpus=selected_rows,
            lease_receipt_identity=_sha(receipt_path),
            placement_group_name=str(selection["placement_group_name"]),
        )

    def bind_launch_receipt_to_lease(
        self, request: Mapping[str, Any], *, selection_path: Path, launch_receipt_path: Path,
    ) -> dict[str, Any]:
        """Bind a normal backend RUNNING receipt; readback only, never relaunch."""
        return self._bind_backend_lease_receipt(
            request, selection_path=selection_path, receipt_path=launch_receipt_path, recovered=False
        )

    def bind_recovery_receipt_to_lease(
        self, request: Mapping[str, Any], *, selection_path: Path, recovery_receipt_path: Path,
    ) -> dict[str, Any]:
        """Adopt backend recovery after launch ACK loss without reselecting or relaunching.

        The backend reconciler is the existing #525 primitive that proves a container is
        already running and promotes the launching reservation exactly once. This method
        only verifies that immutable recovery receipt against the original selection and
        emits the typed ResourceLease. It grants no scientific retry/replay authority.
        """
        return self._bind_backend_lease_receipt(
            request, selection_path=selection_path, receipt_path=recovery_receipt_path, recovered=True
        )

    def reconcile_command(self) -> list[str]:
        cmd = [
            self.binding.python_executable,
            str(self.backend_path),
            "--config", str(self.config_path),
            "--state-root", str(self.state_root),
        ]
        if self.registry_path is not None:
            cmd += ["--profile-registry", str(self.registry_path)]
        cmd.append("reconcile")
        return cmd
