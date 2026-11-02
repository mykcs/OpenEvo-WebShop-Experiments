import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.openevo_webshop.validate_h138b_preregistration import (
    _validate_launch_boundary,
    validate,
)


ROOT = Path(__file__).parents[1]


def _copy_validation_fixture(destination: Path) -> None:
    for relative in ("configs", "docs/evidence/remote-runs/2026-08-20/h1.38b-qualification", "docs/evidence/remote-runs/2026-08-19/h1.38a"):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    for relative in (
        "scripts/openevo_webshop/ordinary_lora_trainer.py",
        "scripts/openevo_webshop/ordinary_lora_contract.py",
        "scripts/openevo_webshop/launch_h138b_method_control.sh",
        "scripts/openevo_webshop/freeze_h138b_sample_plans.py",
        "scripts/openevo_webshop/verify_h138b_qualification.py",
        "scripts/openevo_webshop/verify_adapter_reload.py",
        "scripts/openevo_webshop/reconcile_h138b_method_control.py",
        "scripts/openevo_webshop/h1_closed_loop_tracked.py",
        "docs/evidence/remote-runs/2026-08-20/h1.38b-qualification/implementation/inference.py",
    ):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


class H138BPreregistrationTests(unittest.TestCase):
    def test_published_preregistration_is_integral_and_completed_campaign_is_closed(self):
        result = validate(ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["formal_task_consumption_allowed"])
        self.assertEqual(result["validated_receipts"], 16)
        self.assertEqual(result["blockers"], ["preregistered-campaign-closed"])
        self.assertEqual(len(result["authorized_gpu_uuids"]), 6)

    def test_campaign_authorization_is_standing_not_a_session_human_gate(self):
        prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        self.assertEqual(prereg["gpu_authorization_status"], "campaign-contract-authorized")
        self.assertFalse(prereg["gpu_authorization"]["session_reconfirmation_required"])

    def test_unknown_uuid_cannot_be_paired_with_an_open_formal_gate(self):
        prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
        prereg["authorized_gpu_uuids"][0] = "GPU-unauthorized"
        with self.assertRaisesRegex(ValueError, "authorization set mismatch"):
            _validate_launch_boundary(prereg, current)

    def test_any_authorization_sequence_drift_fails_closed(self):
        source_prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        source_current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
        mutations = {
            "missing": lambda values: values[:-1],
            "extra": lambda values: values + ["GPU-d41486cf-unauthorized"],
            "duplicate": lambda values: values[:-1] + [values[0]],
            "reordered": lambda values: list(reversed(values)),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                prereg = json.loads(json.dumps(source_prereg))
                prereg["authorized_gpu_uuids"] = mutate(prereg["authorized_gpu_uuids"])
                with self.assertRaisesRegex(ValueError, "authorization set mismatch"):
                    _validate_launch_boundary(prereg, source_current)

    def test_preregistration_and_current_formal_gates_must_match(self):
        prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
        current["status"] = "partial-window-closed"
        current["protocol_status"] = "partial-unqualified-safe-boundary-closed"
        current["formal_task_consumption_allowed"] = True
        with self.assertRaisesRegex(ValueError, "closed campaign reopened formal consumption"):
            _validate_launch_boundary(prereg, current)

    def test_current_independent_successor_passes_without_hardcoded_id(self):
        prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
        self.assertNotEqual(current["campaign_id"], "20260820-0129-h138b-method-control")
        self.assertEqual(_validate_launch_boundary(prereg, current), ["preregistered-campaign-closed"])

    def test_synthetic_future_successor_passes_without_validator_edit(self):
        prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
        current.update(
            {
                "campaign_id": "20260820-0141-h141-future-successor",
                "phase": "H1.41",
                "preregistration": "configs/experiment/h1.41-v1.json",
            }
        )
        self.assertEqual(_validate_launch_boundary(prereg, current), ["preregistered-campaign-closed"])

    def test_router_reopening_h138b_formal_contract_fails(self):
        prereg = json.loads((ROOT / "configs/experiment/h1.38b-method-control-v1.json").read_text())
        current = json.loads((ROOT / "configs/experiment/current-campaign.json").read_text())
        current["campaign_id"] = "20260820-0129-h138b-method-control"
        with self.assertRaisesRegex(ValueError, "H1.38B current router preregistration mismatch"):
            _validate_launch_boundary(prereg, current)

    def test_immutable_receipt_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _copy_validation_fixture(root)
            receipt = root / "docs/evidence/remote-runs/2026-08-20/h1.38b-qualification/formal-attempt-3aac79c.json"
            value = json.loads(receipt.read_text())
            value["status"] = "tampered"
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "first attempt receipt SHA mismatch"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
