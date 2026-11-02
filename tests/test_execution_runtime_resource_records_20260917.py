from __future__ import annotations

import copy

import pytest

from scripts.openevo_runtime.resource_records import (
    ResourceRecordError,
    build_lease_release,
    build_resource_blocked,
    build_resource_envelope,
    build_resource_lease,
    build_resource_request,
    validate_resource_envelope,
    validate_resource_lease,
)


def _envelope(**overrides):
    values = {
        "gpu_count": 1,
        "vram_mib_per_gpu": 8000,
        "process_slots_per_gpu": 1,
        "compute_share_advisory": 0.5,
        "host_ram_mib": 4096,
        "cpu_slots": 2,
        "disk_io_class": "NORMAL",
    }
    values.update(overrides)
    return build_resource_envelope(**values)


def _request(envelope=None, **overrides):
    values = {
        "operation_id": "1" * 64,
        "execution_generation_id": "2" * 64,
        "resource_policy_identity": "3" * 64,
        "qualified_topology_identity": "4" * 64,
        "envelope": envelope or _envelope(),
        "placement_mode": "QUALIFIED_SUBSET",
        "preferred_gpu_indices": [7, 6, 5, 4],
    }
    values.update(overrides)
    return build_resource_request(**values)


def test_resource_envelope_separates_per_gpu_and_host_global_accounting() -> None:
    envelope = _envelope(gpu_count=4, host_ram_mib=16384, cpu_slots=8)
    assert envelope["scope_model"] == "PER_GPU_PLUS_HOST_GLOBAL"
    assert envelope["gpu_scope"]["gpu_count"] == 4
    assert envelope["gpu_scope"]["vram_mib_per_gpu"] == 8000
    assert envelope["host_scope"]["ram_mib"] == 16384
    assert envelope["accounting"]["gpu_capacity_is_per_physical_gpu"] is True
    assert envelope["accounting"]["host_capacity_counted_once_per_host"] is True
    validate_resource_envelope(envelope)


def test_host_budget_is_identity_bearing_without_being_multiplied_by_gpu_count() -> None:
    first = _envelope(gpu_count=4, host_ram_mib=8192)
    second = _envelope(gpu_count=4, host_ram_mib=16384)
    assert first["envelope_id"] != second["envelope_id"]
    assert first["host_scope"]["ram_mib"] == 8192
    assert first["accounting"]["host_capacity_counted_once_per_host"] is True


def test_resource_request_binds_generation_policy_topology_and_zero_science_authority() -> None:
    request = _request()
    assert request["execution_generation_id"] == "2" * 64
    assert request["resource_policy_identity"] == "3" * 64
    assert request["qualified_topology_identity"] == "4" * 64
    assert request["scientific_authority_granted"] is False


def test_resource_request_rejects_preferred_set_that_cannot_satisfy_gpu_count() -> None:
    with pytest.raises(ResourceRecordError, match="cannot satisfy"):
        _request(envelope=_envelope(gpu_count=2), preferred_gpu_indices=[7])


def test_resource_lease_binds_physical_gpu_uuid_and_never_grants_topology_authority() -> None:
    request = _request()
    lease = build_resource_lease(
        request,
        selected_gpus=[{"index": 7, "uuid": "GPU-00000000-0000-0000-0000-000000000000"}],
        lease_receipt_identity="5" * 64,
        placement_group_name="ce1-example",
    )
    assert lease["selected_gpus"][0]["uuid"].startswith("GPU-")
    assert lease["scientific_authority_granted"] is False
    assert lease["topology_authority_granted"] is False
    validate_resource_lease(lease)


def test_resource_lease_rejects_gpu_outside_frozen_preference() -> None:
    request = _request(preferred_gpu_indices=[7])
    with pytest.raises(ResourceRecordError, match="outside frozen preferred"):
        build_resource_lease(
            request,
            selected_gpus=[{"index": 6, "uuid": "GPU-00000000-0000-0000-0000-000000000000"}],
            lease_receipt_identity="5" * 64,
            placement_group_name=None,
        )


def test_release_is_separate_from_lease_and_carries_zero_scientific_authority() -> None:
    request = _request()
    lease = build_resource_lease(
        request,
        selected_gpus=[{"index": 7, "uuid": "GPU-00000000-0000-0000-0000-000000000000"}],
        lease_receipt_identity="5" * 64,
        placement_group_name="ce1-example",
    )
    release = build_lease_release(
        lease,
        release_receipt_identity="6" * 64,
        released_gpu_indices=[7],
        placement_group_released=True,
        all_resources_released=True,
    )
    assert release["lease_id"] == lease["lease_id"]
    assert release["all_resources_released"] is True
    assert release["scientific_authority_granted"] is False


def test_resource_block_distinguishes_waitable_capacity_from_authority_or_integrity_block() -> None:
    request = _request()
    capacity = build_resource_blocked(request, block_class="CAPACITY_UNAVAILABLE")
    topology = build_resource_blocked(request, block_class="UNQUALIFIED_TOPOLOGY")
    ambiguous = build_resource_blocked(request, block_class="LEASE_STATE_AMBIGUOUS")
    assert capacity["retryable_resource_wait"] is True
    assert topology["retryable_resource_wait"] is False
    assert ambiguous["retryable_resource_wait"] is False
    assert capacity["scientific_retry_authorized"] is False


def test_envelope_identity_tampering_fails_closed() -> None:
    envelope = _envelope()
    tampered = copy.deepcopy(envelope)
    tampered["host_scope"]["cpu_slots"] += 1
    with pytest.raises(ResourceRecordError, match="envelope_id"):
        validate_resource_envelope(tampered)
