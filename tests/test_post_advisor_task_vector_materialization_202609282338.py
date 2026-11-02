import hashlib
import json

import pytest

from scripts.analysis.post_advisor_materialize_task_vector_leave_one_out_202609282338 import (
    SOURCE_STATES,
    materialize_module_factors as materialize_loo_module_factors,
)
from scripts.analysis.post_advisor_materialize_task_vector_interventions_202609282338 import (
    INTERVENTIONS,
    materialize_module_factors as materialize_direction_module_factors,
    validate_loo_root,
)


class Matrix:
    """Small dependency-free matrix stand-in for factor-concatenation contracts."""

    def __init__(self, rows, *, device="cpu", dtype="float64"):
        self.rows = [list(row) for row in rows]
        self.ndim = 2
        self.shape = (len(self.rows), len(self.rows[0]) if self.rows else 0)
        self.device = type("Device", (), {"type": device})()
        self.dtype = dtype

    def contiguous(self):
        return self

    def __neg__(self):
        return Matrix([[-value for value in row] for row in self.rows], device=self.device.type, dtype=self.dtype)

    def __mul__(self, scalar):
        return Matrix([[scalar * value for value in row] for row in self.rows], device=self.device.type, dtype=self.dtype)

    __rmul__ = __mul__


class MatrixOps:
    @staticmethod
    def cat(values, dim):
        if dim == 0:
            rows = [row for value in values for row in value.rows]
        else:
            rows = [
                [cell for value in values for cell in value.rows[row_index]]
                for row_index in range(values[0].shape[0])
            ]
        return Matrix(rows, device=values[0].device.type, dtype=values[0].dtype)


def matmul(left, right):
    return [
        [sum(left_row[k] * right[k][column] for k in range(len(right))) for column in range(len(right[0]))]
        for left_row in left
    ]


def matrix_add(left, right, scale=1.0):
    return [
        [left[i][j] + scale * right[i][j] for j in range(len(left[i]))]
        for i in range(len(left))
    ]


def assert_matrix_close(actual, expected, tolerance=1e-12):
    assert len(actual) == len(expected)
    assert all(len(a) == len(b) for a, b in zip(actual, expected))
    assert max((abs(a - b) for row_a, row_b in zip(actual, expected) for a, b in zip(row_a, row_b)), default=0) <= tolerance


def test_rank_concatenation_matches_leave_one_update_out_matrix():
    a_final = Matrix([[1, 2], [0, 1]])
    b_final = Matrix([[1, 0], [0, 1]])
    a_t = Matrix([[2, 1]])
    b_t = Matrix([[1], [2]])
    a_prev = Matrix([[1, -1]])
    b_prev = Matrix([[2], [1]])

    a_out, b_out = materialize_loo_module_factors(
        MatrixOps, a_final, b_final, a_t, b_t, a_prev, b_prev
    )

    expected = matrix_add(matrix_add(matmul(b_final.rows, a_final.rows), matmul(b_t.rows, a_t.rows), -1), matmul(b_prev.rows, a_prev.rows))
    assert_matrix_close(matmul(b_out.rows, a_out.rows), expected)
    assert a_out.shape == (4, 2)
    assert b_out.shape == (2, 4)
    assert a_out.dtype == a_final.dtype
    assert b_out.dtype == b_final.dtype


def test_materialization_rejects_incompatible_module_shapes():
    a_final = Matrix([[1, 1, 1], [1, 1, 1]])
    b_final = Matrix([[1, 1], [1, 1]])
    a_t = Matrix([[1, 1], [1, 1]])
    b_t = Matrix([[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="module dimensions drift"):
        materialize_loo_module_factors(
            MatrixOps, a_final, b_final, a_t, b_t, a_final, b_final
        )


def test_materialization_refuses_non_cpu_tensors():
    a_final = Matrix([[1, 2]], device="meta")
    b_final = Matrix([[1], [2]], device="meta")

    with pytest.raises(ValueError, match="CPU tensors only"):
        materialize_loo_module_factors(
            MatrixOps, a_final, b_final, a_final, b_final, a_final, b_final
        )


def test_reverse_direction_is_equal_magnitude_opposite_delta():
    a_final = Matrix([[1, 2], [0, 1]])
    b_final = Matrix([[1, 0], [0, 1]])
    a_t = Matrix([[2, 1]])
    b_t = Matrix([[1], [2]])
    a_prev = Matrix([[1, -1]])
    b_prev = Matrix([[2], [1]])

    loo_a, loo_b = materialize_direction_module_factors(
        MatrixOps, a_final, b_final, a_t, b_t, a_prev, b_prev, (1, -1, 1)
    )
    reverse_a, reverse_b = materialize_direction_module_factors(
        MatrixOps, a_final, b_final, a_t, b_t, a_prev, b_prev,
        INTERVENTIONS["reverse-direction"]["coefficients"],
    )
    loo_state = matmul(loo_b.rows, loo_a.rows)
    reverse_state = matmul(reverse_b.rows, reverse_a.rows)
    delta = matrix_add(matmul(b_t.rows, a_t.rows), matmul(b_prev.rows, a_prev.rows), -1)
    treatment_difference = matrix_add(reverse_state, loo_state, -1)

    expected_reverse = matrix_add(matmul(b_final.rows, a_final.rows), delta, -2)
    assert_matrix_close(reverse_state, expected_reverse)
    assert_matrix_close(treatment_difference, [[-value for value in row] for row in delta])
    assert reverse_a.shape == (4, 2)
    assert reverse_b.shape == (2, 4)


def test_direction_treatment_uses_distinct_ids_for_each_round():
    loo_receipt = {"states": {str(round_index): {} for round_index in range(152, 160)}}
    reverse_ids = {
        INTERVENTIONS["reverse-direction"]["output_id"].format(round_index=round_index)
        for round_index in range(152, 160)
    }
    reverse_receipt = {"states": {output_id: {} for output_id in reverse_ids}}
    loo_ids = {f"leave-one-update-out-r{round_index}" for round_index in loo_receipt["states"]}
    ids = loo_ids | set(reverse_receipt["states"])
    assert len(ids) == 16
    assert "leave-one-update-out-r152" in ids
    assert "reverse-direction-r152" in ids


def test_reverse_treatment_validates_and_references_leave_one_out_hashes(tmp_path):
    expected_sources = {
        round_index: {"state_id": values[0], "adapter_sha256": values[1]}
        for round_index, values in SOURCE_STATES.items()
    }
    states = {}
    for round_index in range(152, 160):
        output_id = f"leave-one-update-out-r{round_index}"
        adapter_dir = tmp_path / output_id
        adapter_dir.mkdir()
        adapter_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        adapter_path.write_bytes(f"synthetic-adapter-{round_index}".encode())
        config_path.write_text(json.dumps({"r": 384, "lora_alpha": 384}))
        states[str(round_index)] = {
            "intervention": f"W159 - (W{round_index} - W{round_index - 1})",
            "output_rank": 384,
            "module_count": 112,
            "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    receipt = {
        "schema": "openevo.post-advisor.task-vector-leave-one-out-materialization.202609282338",
        "status": "PASS",
        "source_rounds": {str(k): v for k, v in expected_sources.items()},
        "output_rank": 384,
        "modules_per_state": 112,
        "task_panel_accessed": False,
        "formal_rollouts_consumed": 0,
        "final_panel_access": 0,
        "states": states,
    }
    (tmp_path / "materialization-receipt.json").write_text(json.dumps(receipt))

    reference = validate_loo_root(tmp_path, expected_sources)

    assert len(reference["states"]) == 8
    assert reference["states"]["152"]["output_id"] == "leave-one-update-out-r152"
    assert reference["states"]["159"]["adapter_sha256"] == states["159"]["adapter_sha256"]
    assert len(reference["receipt_sha256"]) == 64

    (tmp_path / "leave-one-update-out-r152" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_loo_root(tmp_path, expected_sources)
