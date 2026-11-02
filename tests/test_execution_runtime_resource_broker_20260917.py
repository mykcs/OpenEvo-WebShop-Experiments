from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.openevo_runtime.resource_broker import RayResourceBroker, ResourceBrokerError
from scripts.openevo_runtime.resource_records import build_resource_request
from scripts.openevo_runtime.result_codes import ReasonCode, RetryClass
from scripts.openevo_webshop import bounded_effective_state_resource_broker_202609170757 as bounded
from scripts.openevo_webshop import ceiling1_ray_placement_backend_202609042025 as ray_backend

ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = ROOT / ".pytest-resource-broker-unused"
IDS = {
    "operation_id": "1" * 64,
    "execution_generation_id": "2" * 64,
    "resource_policy_identity": "3" * 64,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route(*, arm: str = "BOUNDED_OFF", kind: str = "rollout", reflector: str | None = None, count: int = 1):
    channel = bounded.channel_for(kind=kind, active_reflector_kind=reflector)
    workload, _, _, _ = bounded.workload_for(ROOT, experiment_arm_id=arm, channel=channel)
    topology_id = bounded.qualified_topology_identity_for(ROOT, workload=workload)
    preferred = [4, 5, 6, 7] if kind == "rollout" else [4]
    return bounded.build_broker_route(
        ROOT,
        state_root=STATE_ROOT,
        experiment_arm_id=arm,
        kind=kind,
        active_reflector_kind=reflector,
        gpu_count=count,
        preferred_gpu_indices=preferred,
        qualified_topology_identity=topology_id,
        **IDS,
    )


def _fake_success_runner(request, broker):
    def run(cmd, *, text, capture_output):
        assert text is True and capture_output is True
        assert "--exact-count" in cmd
        output = Path(cmd[cmd.index("--output") + 1])
        count = int(cmd[cmd.index("--count") + 1])
        selected = list(request["preferred_gpu_indices"])[:count]
        cfg = json.loads(broker.config_path.read_text())
        uuids = [cfg["managed_gpu_uuids"][str(index)] for index in selected]
        live = {str(index): {"uuid": uuid, "pass": True} for index, uuid in zip(selected, uuids)}
        receipt = {
            "schema": "openevo.ceiling1-ray-selection.202609042025",
            "workload": broker.binding.workload,
            "selected_gpus": selected,
            "selected_gpu_uuids": uuids,
            "placement_group_name": "pg-test",
            "reservation_id": "reservation-test",
            "config_sha256": _sha(broker.config_path),
            "vram_request_mib_per_gpu": request["envelope"]["gpu_scope"]["vram_mib_per_gpu"],
            "live_gate": live,
        }
        if broker.registry_path is not None:
            receipt["profile_registry_sha256"] = _sha(broker.registry_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        return subprocess.CompletedProcess(cmd, 0, json.dumps(receipt), "")

    return run


def test_all_bounded_gpu_channels_route_through_one_broker_api() -> None:
    cases = [
        ("rollout", None, "bounded_off_rollout"),
        ("transition", None, "bounded_off_sd_lora"),
        ("reflector", "text_memory", "bounded_off_text_memory"),
        ("reflector", "skill_bundle", "bounded_off_skill_bundle"),
        ("reflector", "agent_system", "bounded_off_agent_system"),
    ]
    for kind, reflector, expected in cases:
        route = _route(kind=kind, reflector=reflector)
        assert isinstance(route.broker, RayResourceBroker)
        assert route.workload == expected
        assert route.request["scientific_authority_granted"] is False
    on_route = _route(arm="BOUNDED_EFFECTIVE_STATE_ON", kind="reflector", reflector="agent_system")
    assert on_route.workload == "bounded_on_agent_system"


def test_broker_uses_exact_count_and_adopts_typed_ready_selection(tmp_path: Path) -> None:
    route = _route(kind="rollout", count=4)
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=tmp_path / "selection.json")
    assert decision.ready is True
    assert decision.outcome.side_effect_materialized is True
    assert decision.outcome.scientific_state_mutated is False
    assert decision.selection_receipt is not None
    assert len(decision.selection_receipt["selected_gpus"]) == 4
    assert decision.resource_blocked is None


def test_no_feasible_placement_becomes_typed_wait_without_text_branch_in_consumer(tmp_path: Path) -> None:
    route = _route(kind="transition")

    def no_capacity(cmd, *, text, capture_output):
        return subprocess.CompletedProcess(cmd, 1, "", "RAY_NO_FEASIBLE_PLACEMENT: zero candidates")
    broker = RayResourceBroker(route.broker.binding, command_runner=no_capacity)
    decision = broker.request_placement(route.request, selection_path=tmp_path / "selection.json")
    assert decision.ready is False
    assert decision.outcome.reason_code is ReasonCode.GPU_RESOURCE_UNAVAILABLE
    assert decision.outcome.retry_class is RetryClass.WAIT_RESOURCE
    assert decision.resource_blocked is not None
    assert decision.resource_blocked["block_class"] == "PLACEMENT_UNAVAILABLE"
    assert decision.resource_blocked["scientific_retry_authorized"] is False


def test_unknown_backend_failure_remains_fail_closed(tmp_path: Path) -> None:
    route = _route(kind="transition")

    def unknown(cmd, *, text, capture_output):
        return subprocess.CompletedProcess(cmd, 1, "", "mysterious backend failure")

    broker = RayResourceBroker(route.broker.binding, command_runner=unknown)
    decision = broker.request_placement(route.request, selection_path=tmp_path / "selection.json")
    assert decision.ready is False
    assert decision.outcome.reason_code is ReasonCode.UNCLASSIFIED_ENGINEERING_FAILURE
    assert decision.outcome.retry_class is RetryClass.INTEGRITY_UNKNOWN
    assert decision.resource_blocked is None


def test_existing_exact_selection_is_reused_without_backend_call(tmp_path: Path) -> None:
    route = _route(kind="reflector", reflector="text_memory")
    selection = tmp_path / "selection.json"
    seed_broker = RayResourceBroker(route.broker.binding)
    seed_broker.command_runner = _fake_success_runner(route.request, seed_broker)
    first = seed_broker.request_placement(route.request, selection_path=selection)
    assert first.ready

    def must_not_run(*args, **kwargs):
        raise AssertionError("existing immutable selection must be adopted")

    readback = RayResourceBroker(route.broker.binding, command_runner=must_not_run)
    second = readback.request_placement(route.request, selection_path=selection)
    assert second.ready
    assert second.selection_receipt == first.selection_receipt



def _write_launch_receipt(route, selection: dict, path: Path, *, uuid_override: str | None = None) -> None:
    selected = list(selection["selected_gpus"])
    live_gate = {}
    for index, uuid in zip(selected, selection["selected_gpu_uuids"]):
        observed = uuid_override if uuid_override is not None and index == selected[0] else uuid
        live_gate[str(index)] = {"uuid": observed, "expected_uuid": uuid, "pass": True}
    payload = {
        "schema": "openevo.ceiling1-ray-launch.202609042025",
        "status": "RUNNING",
        "selection": selection,
        "live_gate": live_gate,
        "lease": {
            "lease_id": "backend-lease-test",
            "workload": route.workload,
            "container": "test-container",
            "gpus": selected,
            "placement_group_name": selection["placement_group_name"],
            "reservation_id": selection["reservation_id"],
            "acquired": 1.0,
            "vram_request_mib_per_gpu": route.request["envelope"]["gpu_scope"]["vram_mib_per_gpu"],
            "recovered_from_launching_reservation": False,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")



def test_gang_rollout_requires_immutable_placement_group_identity(tmp_path: Path) -> None:
    route = _route(kind="rollout", count=4)
    def missing_pg(cmd, *, text, capture_output):
        output = Path(cmd[cmd.index("--output") + 1])
        count = int(cmd[cmd.index("--count") + 1])
        selected = list(route.request["preferred_gpu_indices"])[:count]
        cfg = json.loads(route.broker.config_path.read_text())
        uuids = [cfg["managed_gpu_uuids"][str(index)] for index in selected]
        output.write_text(json.dumps({
            "schema": "openevo.ceiling1-ray-selection.202609042025",
            "workload": route.workload,
            "selected_gpus": selected,
            "selected_gpu_uuids": uuids,
            "placement_group_name": "",
            "reservation_id": "reservation-test",
            "config_sha256": _sha(route.broker.config_path),
            "profile_registry_sha256": _sha(route.broker.registry_path),
            "vram_request_mib_per_gpu": 7000,
            "live_gate": {str(index): {"uuid": uuid, "pass": True} for index, uuid in zip(selected, uuids)},
        }) + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    broker = RayResourceBroker(route.broker.binding, command_runner=missing_pg)
    with pytest.raises(ResourceBrokerError, match="placement-group identity"):
        broker.request_placement(route.request, selection_path=tmp_path / "selection.json")


def test_gang_rollout_selection_is_exact_four_gpu_placement_group(tmp_path: Path) -> None:
    route = _route(kind="rollout", count=4)
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=tmp_path / "selection.json")
    assert decision.ready
    assert decision.selection_receipt is not None
    assert decision.selection_receipt["selected_gpus"] == [4, 5, 6, 7]
    assert decision.selection_receipt["placement_group_name"] == "pg-test"
    assert decision.selection_receipt["reservation_id"] == "reservation-test"
    assert broker._config["rollout_strategy"] == "STRICT_SPREAD"

def test_launch_receipt_binds_physical_uuid_readback_into_typed_lease(tmp_path: Path) -> None:
    route = _route(kind="transition")
    selection_path = tmp_path / "selection.json"
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=selection_path)
    assert decision.selection_receipt is not None
    launch_path = tmp_path / "lease.launch.json"
    _write_launch_receipt(route, dict(decision.selection_receipt), launch_path)
    lease = broker.bind_launch_receipt_to_lease(
        route.request,
        selection_path=selection_path,
        launch_receipt_path=launch_path,
    )
    expected_uuid = json.loads(broker.config_path.read_text())["managed_gpu_uuids"]["4"]
    assert lease["selected_gpus"] == [{"index": 4, "uuid": expected_uuid}]
    assert lease["lease_receipt_identity"] == _sha(launch_path)
    assert lease["qualified_topology_identity"] == route.request["qualified_topology_identity"]
    assert lease["scientific_authority_granted"] is False
    assert lease["topology_authority_granted"] is False


def test_launch_receipt_uuid_drift_fails_closed_before_typed_lease(tmp_path: Path) -> None:
    route = _route(kind="transition")
    selection_path = tmp_path / "selection.json"
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=selection_path)
    assert decision.selection_receipt is not None
    launch_path = tmp_path / "lease.launch.json"
    _write_launch_receipt(route, dict(decision.selection_receipt), launch_path, uuid_override="GPU-uuid-drift")
    with pytest.raises(ResourceBrokerError, match="live UUID readback drift"):
        broker.bind_launch_receipt_to_lease(
            route.request,
            selection_path=selection_path,
            launch_receipt_path=launch_path,
        )


def test_launch_receipt_cannot_substitute_different_selection(tmp_path: Path) -> None:
    route = _route(kind="transition")
    selection_path = tmp_path / "selection.json"
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=selection_path)
    assert decision.selection_receipt is not None
    launch_path = tmp_path / "lease.launch.json"
    selection = dict(decision.selection_receipt)
    _write_launch_receipt(route, selection, launch_path)
    payload = json.loads(launch_path.read_text())
    payload["selection"]["reservation_id"] = "different-reservation"
    launch_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ResourceBrokerError, match="selection/readback drift"):
        broker.bind_launch_receipt_to_lease(
            route.request,
            selection_path=selection_path,
            launch_receipt_path=launch_path,
        )

def test_host_hard_demand_fails_closed_until_backend_can_account_it() -> None:
    route = bounded.build_broker_route(
        ROOT,
        state_root=STATE_ROOT,
        experiment_arm_id="BOUNDED_OFF",
        kind="transition",
        gpu_count=1,
        preferred_gpu_indices=[4],
        host_ram_mib=1024,
        qualified_topology_identity=bounded.qualified_topology_identity_for(ROOT, workload="bounded_off_sd_lora"),
        **IDS,
    )
    with pytest.raises(ResourceBrokerError, match="host-global hard admission"):
        route.broker.request_placement(route.request, selection_path=ROOT / ".never-written-selection.json")


def test_backend_exact_count_turns_rollout_into_all_or_nothing_count() -> None:
    _, profile, _, _ = bounded.workload_for(ROOT, experiment_arm_id="BOUNDED_OFF", channel="rollout")
    assert ray_backend.selection_count_bounds(profile, 4, True) == (4, 4)
    assert ray_backend.selection_count_bounds(profile, 4, False) == (4, 1)
    with pytest.raises(RuntimeError, match="requires --count"):
        ray_backend.selection_count_bounds(profile, None, True)



def test_broker_rejects_unregistered_topology_identity(tmp_path: Path) -> None:
    route = _route(kind="rollout", count=4)
    request = build_resource_request(
        operation_id=route.request["operation_id"],
        execution_generation_id=route.request["execution_generation_id"],
        resource_policy_identity=route.request["resource_policy_identity"],
        qualified_topology_identity="f" * 64,
        envelope=route.request["envelope"],
        placement_mode=route.request["placement_mode"],
        preferred_gpu_indices=route.request["preferred_gpu_indices"],
    )
    with pytest.raises(ResourceBrokerError, match="unregistered qualified topology identity"):
        route.broker.request_placement(request, selection_path=tmp_path / "selection.json")


def test_broker_rejects_unauthorized_fallback_topology(tmp_path: Path) -> None:
    route = bounded.build_broker_route(
        ROOT,
        state_root=STATE_ROOT,
        experiment_arm_id="BOUNDED_OFF",
        kind="rollout",
        gpu_count=4,
        preferred_gpu_indices=[4, 5, 6, 7],
        qualified_topology_identity=bounded.qualified_topology_identity_for(ROOT, workload="bounded_off_rollout"),
        fallback_topology_identity="e" * 64,
        **IDS,
    )
    with pytest.raises(ResourceBrokerError, match="fallback"):
        route.broker.request_placement(route.request, selection_path=tmp_path / "selection.json")


def test_broker_rejects_preferred_set_change_even_inside_managed_pool(tmp_path: Path) -> None:
    route = bounded.build_broker_route(
        ROOT,
        state_root=STATE_ROOT,
        experiment_arm_id="BOUNDED_OFF",
        kind="transition",
        gpu_count=1,
        preferred_gpu_indices=[5],
        qualified_topology_identity=bounded.qualified_topology_identity_for(ROOT, workload="bounded_off_sd_lora"),
        **IDS,
    )
    with pytest.raises(ResourceBrokerError, match="preferred GPU set changes qualified topology"):
        route.broker.request_placement(route.request, selection_path=tmp_path / "selection.json")


def test_broker_rejects_backend_selection_outside_exact_topology(tmp_path: Path) -> None:
    route = _route(kind="transition")
    def changed_topology(cmd, *, text, capture_output):
        output = Path(cmd[cmd.index("--output") + 1])
        cfg = json.loads(route.broker.config_path.read_text())
        index = 5
        uuid = cfg["managed_gpu_uuids"][str(index)]
        output.write_text(json.dumps({
            "schema": "openevo.ceiling1-ray-selection.202609042025",
            "workload": route.workload,
            "selected_gpus": [index],
            "selected_gpu_uuids": [uuid],
            "placement_group_name": "pg-topology-drift",
            "reservation_id": "topology-drift",
            "config_sha256": _sha(route.broker.config_path),
            "profile_registry_sha256": _sha(route.broker.registry_path),
            "vram_request_mib_per_gpu": 8000,
            "live_gate": {str(index): {"uuid": uuid, "pass": True}},
        }) + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    broker = RayResourceBroker(route.broker.binding, command_runner=changed_topology)
    with pytest.raises(ResourceBrokerError, match="preferred GPU set|qualified topology|escaped frozen preferred"):
        broker.request_placement(route.request, selection_path=tmp_path / "selection.json")

def test_broker_rejects_silent_gpu_count_shrink(tmp_path: Path) -> None:
    route = _route(kind="rollout", count=4)

    def shrink(cmd, *, text, capture_output):
        output = Path(cmd[cmd.index("--output") + 1])
        cfg = json.loads(route.broker.config_path.read_text())
        index = 7
        uuid = cfg["managed_gpu_uuids"][str(index)]
        output.write_text(json.dumps({
            "schema": "openevo.ceiling1-ray-selection.202609042025",
            "workload": route.workload,
            "selected_gpus": [index],
            "selected_gpu_uuids": [uuid],
            "placement_group_name": "pg-shrunk",
            "reservation_id": "shrunk",
            "config_sha256": _sha(route.broker.config_path),
            "profile_registry_sha256": _sha(route.broker.registry_path),
            "vram_request_mib_per_gpu": 7000,
            "live_gate": {str(index): {"uuid": uuid, "pass": True}},
        }) + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    broker = RayResourceBroker(route.broker.binding, command_runner=shrink)
    with pytest.raises(ResourceBrokerError, match="silently changed requested GPU count"):
        broker.request_placement(route.request, selection_path=tmp_path / "selection.json")


def _write_recovery_receipt(route, selection: dict, path: Path, *, reservation_id: str | None = None) -> None:
    payload = {
        "schema": "openevo.ceiling1-ray-launch-recovery.202609042102",
        "status": "RECOVERED_RUNNING_LEASE",
        "lease": {
            "lease_id": "backend-recovered-lease",
            "workload": route.workload,
            "container": "already-running-container",
            "gpus": list(selection["selected_gpus"]),
            "placement_group_name": selection["placement_group_name"],
            "reservation_id": reservation_id or selection["reservation_id"],
            "acquired": 1.0,
            "vram_request_mib_per_gpu": route.request["envelope"]["gpu_scope"]["vram_mib_per_gpu"],
            "recovered_from_launching_reservation": True,
        },
        "recovered_at_epoch": 2.0,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_selection_ack_loss_adopts_immutable_selection_without_second_backend_call(tmp_path: Path) -> None:
    route = _route(kind="transition")
    selection_path = tmp_path / "selection.json"
    first_broker = RayResourceBroker(route.broker.binding)
    first_broker.command_runner = _fake_success_runner(route.request, first_broker)
    first = first_broker.request_placement(route.request, selection_path=selection_path)
    assert first.ready and first.selection_receipt is not None

    def must_not_reselect(*args, **kwargs):
        raise AssertionError("selection acknowledgement loss must adopt exact immutable selection")

    recovered_broker = RayResourceBroker(route.broker.binding, command_runner=must_not_reselect)
    second = recovered_broker.request_placement(route.request, selection_path=selection_path)
    assert second.ready
    assert second.selection_receipt == first.selection_receipt


def test_launch_ack_loss_adopts_recovered_running_lease_without_relaunch(tmp_path: Path) -> None:
    route = _route(kind="transition")
    selection_path = tmp_path / "selection.json"
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=selection_path)
    assert decision.selection_receipt is not None
    recovery_path = tmp_path / "lease.recovered-launch.json"
    _write_recovery_receipt(route, dict(decision.selection_receipt), recovery_path)
    typed_lease = broker.bind_recovery_receipt_to_lease(
        route.request,
        selection_path=selection_path,
        recovery_receipt_path=recovery_path,
    )
    assert typed_lease["lease_receipt_identity"] == _sha(recovery_path)
    assert typed_lease["request_id"] == route.request["request_id"]
    assert typed_lease["scientific_authority_granted"] is False
    assert typed_lease["topology_authority_granted"] is False


def test_recovery_receipt_identity_drift_fails_closed(tmp_path: Path) -> None:
    route = _route(kind="transition")
    selection_path = tmp_path / "selection.json"
    broker = RayResourceBroker(route.broker.binding)
    broker.command_runner = _fake_success_runner(route.request, broker)
    decision = broker.request_placement(route.request, selection_path=selection_path)
    assert decision.selection_receipt is not None
    recovery_path = tmp_path / "lease.recovered-launch.json"
    _write_recovery_receipt(
        route,
        dict(decision.selection_receipt),
        recovery_path,
        reservation_id="different-reservation",
    )
    with pytest.raises(ResourceBrokerError, match="reservation drift"):
        broker.bind_recovery_receipt_to_lease(
            route.request,
            selection_path=selection_path,
            recovery_receipt_path=recovery_path,
        )
