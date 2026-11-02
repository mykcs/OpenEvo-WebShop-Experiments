"""CPU-only regression for candidate-backend handoff evidence scoring."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_runtime import shadow_gpu_handoff as subject


def observation(index: int, *, gap_ms: int = 3, synthetic: bool = False) -> dict:
    end = 20_000_000_000 * (index + 1)
    prefix = "synthetic:" if synthetic else "receipt:"
    return {
        "schema": subject.SCHEMA,
        "backend": "hyperqueue",
        "workload_class": "webshop_eval_same_model",
        "clock_domain": "lyg2171-boot-id-20261023",
        "gpu_uuid": "GPU-12345678123412341234123456789abc",
        "previous_execution": f"sealed-arm-{index}",
        "next_execution": f"next-arm-{index}",
        "warm_model": True,
        "synthetic": synthetic,
        "previous_last_useful_kernel_end_ns": end,
        "next_science_ready_ns": end - 100_000_000,
        "dispatch_detected_ns": end + 500_000,
        "spawned_ns": end + 1_000_000,
        "next_first_useful_kernel_ns": end + gap_ms * 1_000_000,
        "next_first_scientific_output_ns": end + (gap_ms + 1) * 1_000_000,
        "previous_kernel_evidence_ref": prefix + f"old-{index}",
        "next_kernel_evidence_ref": prefix + f"new-{index}",
        "next_scientific_output_receipt_ref": prefix + f"science-{index}",
    }


class ShadowScoreTest(unittest.TestCase):
    def test_warm_ready_30_independent_real_observations_are_timing_candidate_only(self):
        result = subject.analyze([observation(i) for i in range(30)])
        group = result["backend_groups"][0]
        self.assertEqual(group["real_n"], 30)
        self.assertEqual(group["p50_useful_gap_ms"], 3)
        self.assertEqual(group["p95_useful_gap_ms"], 3)
        self.assertEqual(group["p95_first_scientific_output_gap_ms"], 4)
        self.assertEqual(group["count_useful_gap_gt_5s"], 0)
        self.assertEqual(group["count_useful_gap_gt_30s"], 0)
        self.assertEqual(group["status"], "TIMING_TARGET_MET_SHADOW_ONLY")
        self.assertFalse(result["production_cutover_authorized"])
        self.assertTrue(result["independent_proof_validation_required"])

    def test_two_slow_handoffs_exceed_conservative_nearest_rank_p95(self):
        rows = [observation(i, gap_ms=7000 if i >= 28 else 300) for i in range(30)]
        group = subject.analyze(rows)["backend_groups"][0]
        self.assertEqual(group["p95_useful_gap_ms"], 7000)
        self.assertEqual(group["count_useful_gap_gt_5s"], 2)
        self.assertEqual(group["count_useful_gap_gt_30s"], 0)
        self.assertEqual(group["status"], "TIMING_TARGET_MISSED")

    def test_exceeding_30_seconds_is_counted_separately(self):
        rows = [observation(i, gap_ms=35000 if i == 29 else 100) for i in range(30)]
        group = subject.analyze(rows)["backend_groups"][0]
        self.assertEqual(group["count_useful_gap_gt_5s"], 1)
        self.assertEqual(group["count_useful_gap_gt_30s"], 1)

    def test_nanosecond_precision_does_not_turn_over_target_into_pass(self):
        rows = [observation(i, gap_ms=5000) for i in range(30)]
        rows[28]["next_first_useful_kernel_ns"] += 1
        rows[28]["next_first_scientific_output_ns"] += 1
        rows[29]["next_first_useful_kernel_ns"] += 1
        rows[29]["next_first_scientific_output_ns"] += 1
        group = subject.analyze(rows)["backend_groups"][0]
        self.assertEqual(group["status"], "TIMING_TARGET_MISSED")

    def test_small_sample_is_not_validated_as_five_seconds(self):
        group = subject.analyze([observation(0)])["backend_groups"][0]
        self.assertEqual(group["status"], "INSUFFICIENT_REAL_OBSERVATIONS")

    def test_synthetic_requires_opt_in_and_never_meets_threshold(self):
        rows = [observation(i, synthetic=True) for i in range(31)]
        with self.assertRaisesRegex(subject.ObservationError, "synthetic"):
            subject.analyze(rows)
        group = subject.analyze(rows, allow_synthetic=True)["backend_groups"][0]
        self.assertEqual(group["synthetic_n"], 31)
        self.assertEqual(group["real_n"], 0)
        self.assertEqual(group["status"], "INSUFFICIENT_REAL_OBSERVATIONS")

    def test_unready_or_cold_are_kept_separate_from_warm_ready(self):
        rows = [observation(0), observation(1), observation(2)]
        rows[1]["next_science_ready_ns"] = rows[1]["previous_last_useful_kernel_end_ns"] + 1
        rows[2]["warm_model"] = False
        result = subject.analyze(rows)
        self.assertEqual(len(result["backend_groups"]), 3)
        self.assertEqual(
            sum(g["status"] == "DIAGNOSTIC_ONLY_NOT_READY_OR_COLD" for g in result["backend_groups"]),
            2,
        )

    def test_never_aggregate_distinct_backends_or_workload_classes(self):
        rows = [observation(0), observation(1)]
        rows[1]["backend"] = "persistent_worker"
        self.assertEqual(len(subject.analyze(rows)["backend_groups"]), 2)

    def test_duplicate_or_overlapping_execution_is_not_a_second_handoff(self):
        row = observation(0)
        with self.assertRaisesRegex(subject.ObservationError, "duplicate"):
            subject.analyze([row, dict(row)])
        row["next_first_useful_kernel_ns"] = row["previous_last_useful_kernel_end_ns"] - 1
        with self.assertRaisesRegex(subject.ObservationError, "event order"):
            subject.analyze([row])

    def test_bad_gpu_uuid_boolean_clock_or_science_receipt_fail_closed(self):
        for field, invalid in (
            ("gpu_uuid", "GPU-2"),
            ("synthetic", 0),
            ("clock_domain", ""),
            ("next_scientific_output_receipt_ref", ""),
        ):
            with self.subTest(field=field):
                row = observation(0)
                row[field] = invalid
                with self.assertRaises(subject.ObservationError):
                    subject.analyze([row])

    def test_synthetic_real_identity_mislabel_is_rejected(self):
        row = observation(0, synthetic=True)
        row["synthetic"] = False
        with self.assertRaisesRegex(subject.ObservationError, "synthetic evidence"):
            subject.analyze([row])

    def test_cli_rejects_blank_physical_record_and_accepts_fixture_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            row = observation(0, synthetic=True)
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = subject.main(["--input", str(path), "--allow-synthetic"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["observations_total"], 1)
            path.write_text(json.dumps(row) + "\n\n" + json.dumps(row), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(subject.main(["--input", str(path), "--allow-synthetic"]), 2)


if __name__ == "__main__":
    unittest.main()
