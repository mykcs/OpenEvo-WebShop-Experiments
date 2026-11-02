from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.openevo_webshop.verify_adapter_reload import loaded_adapter_snapshot


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts/openevo_webshop/h1_closed_loop_tracked.py"
INFERENCE = ROOT / "vendor/openevo-webshop/src/openevo_webshop/inference.py"


class _FakeParameter:
    shape = (1,)
    dtype = "torch.bfloat16"

    def __init__(self, payload: bytes = b"x", *, requires_grad: bool = False):
        self.payload = payload
        self.requires_grad = requires_grad

    def detach(self):
        return self

    def to(self, *, device):
        self.device = device
        return self

    def contiguous(self):
        return self

    def view(self, _dtype):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.payload

    def numel(self):
        return 1


class _FakeModel:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return list(self.parameters)


class H138BAdapterLoadContractTests(unittest.TestCase):
    def test_runner_binds_adapter_receipt_into_protocol_attempt_and_summary(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("adapter_load_receipt = backend.load_receipt", text)
        self.assertIn('"adapter_tensor_digest": adapter_load_receipt["loaded_tensor_digest"]', text)
        self.assertGreaterEqual(text.count('"adapter_load_receipt": adapter_load_receipt'), 2)
        self.assertIn('"ordinary-lora"', text)
        self.assertIn('"sd-lora"', text)
        self.assertIn('"evaluation_stage": args.evaluation_stage', text)

    def test_transformers_backend_fails_closed_and_hashes_loaded_tensors(self):
        text = INFERENCE.read_text(encoding="utf-8")
        self.assertIn("requested PEFT adapter is incomplete", text)
        self.assertIn("requested PEFT adapter loaded no tensors", text)
        self.assertIn('"loaded_tensor_digest": self._tensor_digest(self._model)', text)
        self.assertIn('"adapter_loaded": False', text)

    def test_reload_verifier_accepts_sd_or_ordinary_state(self):
        text = (ROOT / "scripts/openevo_webshop/verify_adapter_reload.py").read_text(encoding="utf-8")
        self.assertIn("openevo_ordinary_lora_state.json", text)
        self.assertIn('"ordinary-lora"', text)
        self.assertIn("requested adapter reload exposed no active LoRA tensors", text)
        self.assertIn('"base_trainable_parameter_count"', text)

    def test_reload_snapshot_rejects_empty_or_trainable_base(self):
        torch = SimpleNamespace(uint8="uint8")
        with self.assertRaisesRegex(RuntimeError, "no active LoRA tensors"):
            loaded_adapter_snapshot(_FakeModel([("model.weight", _FakeParameter())]), torch)
        with self.assertRaisesRegex(RuntimeError, "base parameters trainable"):
            loaded_adapter_snapshot(
                _FakeModel(
                    [
                        ("base.weight", _FakeParameter(requires_grad=True)),
                        ("adapter.lora_A.default.weight", _FakeParameter()),
                    ]
                ),
                torch,
            )

    def test_reload_snapshot_records_active_lora_capacity(self):
        snapshot = loaded_adapter_snapshot(
            _FakeModel(
                [
                    ("base.weight", _FakeParameter()),
                    ("adapter.lora_A.default.weight", _FakeParameter(b"a")),
                    ("adapter.lora_B.default.weight", _FakeParameter(b"b")),
                ]
            ),
            SimpleNamespace(uint8="uint8"),
        )
        self.assertEqual(snapshot["loaded_lora_tensor_count"], 2)
        self.assertEqual(snapshot["loaded_lora_parameter_count"], 2)
        self.assertEqual(snapshot["base_trainable_parameter_count"], 0)


if __name__ == "__main__":
    unittest.main()
