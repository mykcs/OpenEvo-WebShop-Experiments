import argparse
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts.openevo_webshop.gpu_resource_wrapper import account_receipt, classify_receipt, run
from scripts.openevo_webshop.path_contract import PathContractError, validate_contract


def wrapper_args(receipt: Path, command: list[str], gpu_uuids: str = "GPU-a,GPU-b", path_contract: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        receipt=str(receipt),
        attempt_id="attempt-test-001",
        campaign_id="20260820-0140r2-h140-g2-multigeneration",
        stage="qualification",
        command_identity="test-payload-v1",
        gpu_uuids=gpu_uuids,
        path_contract=path_contract,
        container_root=None,
        command=command,
    )


class H140ResourceWrapperTests(unittest.TestCase):
    def test_payload_failure_still_writes_resource_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            self.assertEqual(run(wrapper_args(receipt, [sys.executable, "-c", "raise SystemExit(3)"])), 3)
            value = json.loads(receipt.read_text())
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["return_code"], 3)
            self.assertIsNotNone(value["end_utc"])

    def test_shell_or_container_failure_writes_end_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            self.assertEqual(run(wrapper_args(receipt, ["sh", "-c", "exit 127"])), 127)
            value = json.loads(receipt.read_text())
            self.assertEqual(value["return_code"], 127)
            self.assertIsNotNone(value["start_utc"])
            self.assertIsNotNone(value["end_utc"])

    def test_nonzero_return_is_accounted(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            run(wrapper_args(receipt, [sys.executable, "-c", "raise SystemExit(7)"], gpu_uuids="GPU-a"))
            value = json.loads(receipt.read_text())
            self.assertEqual(value["return_code"], 7)
            self.assertEqual(account_receipt(value)["status"], "exact")

    def test_multi_gpu_hours_equal_wall_time_times_count(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            run(wrapper_args(receipt, [sys.executable, "-c", "import time; time.sleep(0.02)"], gpu_uuids="GPU-a,GPU-b,GPU-c"))
            value = json.loads(receipt.read_text())
            self.assertAlmostEqual(value["gpu_hours"], value["wall_seconds"] * 3 / 3600.0, places=9)

    def test_missing_payload_receipt_does_not_erase_wrapper_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "wrapper.json"
            payload = root / "payload.json"
            run(wrapper_args(receipt, [sys.executable, "-c", "import pathlib; pathlib.Path('never-created.json')"]),)
            self.assertTrue(receipt.exists())
            self.assertFalse(payload.exists())
            self.assertEqual(classify_receipt(receipt)["status"], "exact")

    def test_host_only_output_path_is_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            contract = {"paths": {"output": {"host_path": str(output), "container_path": str(output), "expected_type": "dir", "access": "rw", "writable": True}}}
            with self.assertRaisesRegex(PathContractError, "host-only"):
                validate_contract(contract)

    def test_invalid_mount_mapping_is_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            contract = {"paths": {"output": {"host_path": str(output), "container_path": "runs/output", "expected_type": "dir", "access": "rw", "writable": True}}}
            with self.assertRaisesRegex(PathContractError, "absolute"):
                validate_contract(contract)

    def test_container_visible_path_must_be_exact_host_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host-output"
            visible = root / "container" / "workspace" / "runs"
            host.mkdir()
            visible.mkdir(parents=True)
            contract = {"paths": {"output": {"host_path": str(host), "container_path": "/workspace/runs", "expected_type": "dir", "access": "rw", "writable": True}}}
            with self.assertRaisesRegex(PathContractError, "exact host mapping"):
                validate_contract(contract, container_root=root / "container")

    def test_exact_accounting_accepts_failure_receipt(self):
        value = {"schema": "openevo.gpu-resource-receipt.v1", "status": "failed", "wall_seconds": 2.0, "gpu_hours": 4 / 3600, "gpu_count": 2, "return_code": 9}
        self.assertEqual(account_receipt(value)["status"], "exact")
        self.assertEqual(account_receipt(value)["return_code"], 9)

    def test_unknown_only_when_wrapper_absent_or_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(classify_receipt(root / "missing.json")["status"], "unknown")
            corrupt = root / "corrupt.json"
            corrupt.write_text("not-json")
            self.assertEqual(classify_receipt(corrupt)["status"], "unknown")
            valid = root / "valid.json"
            valid.write_text(json.dumps({"schema": "openevo.gpu-resource-receipt.v1", "status": "failed", "wall_seconds": 1.0, "gpu_hours": 1 / 3600, "gpu_count": 1, "return_code": 1}))
            self.assertEqual(classify_receipt(valid)["status"], "exact")


if __name__ == "__main__":
    unittest.main()
