import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_teacher_provider_contract import build_receipt


HASHES = {
    "teacher_contract_sha256": "1" * 64,
    "teacher_prompt_sha256": "2" * 64,
    "teacher_payload_sha256": "3" * 64,
    "teacher_evidence_sha256": "4" * 64,
}


class TeacherProviderContractTests(unittest.TestCase):
    def manifest(self, root: Path) -> Path:
        path = root / "tasks.json"
        path.write_text(json.dumps({"tasks": [{"goal_idx": 11}, {"goal_idx": 22}]}), encoding="utf-8")
        return path

    def kwargs(self, manifest: Path) -> dict:
        return dict(
            task_manifest=manifest,
            base_url="https://provider.example/v1",
            teacher_identity="provider/model@revision",
            teacher_model="provider-model",
            **HASHES,
            train_goal_idx=[11],
            evaluation_goal_idx=[22],
            prompt_contract="seed3090.webshop-compact-action-prompt.v2",
            action_contract="seed3090.webshop-action-parser.v2-missing-close",
            temperature=0.0,
            max_steps=30,
            max_new_tokens=512,
            rollout_seeds=[1, 2],
        )

    def test_passes_without_network_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = build_receipt(**self.kwargs(self.manifest(Path(directory))))
        self.assertEqual(receipt["status"], "pass")
        self.assertFalse(receipt["network_probe_performed"])
        self.assertFalse(receipt["credential_value_recorded"])
        self.assertTrue(receipt["task_split"]["disjoint"])

    def test_missing_endpoint_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(self.manifest(Path(directory)))
            kwargs["base_url"] = None
            receipt = build_receipt(**kwargs)
        self.assertEqual(receipt["status"], "blocked")
        self.assertIn("provider endpoint is absent or not an http(s) URL", receipt["blocking_reasons"])

    def test_overlap_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(self.manifest(Path(directory)))
            kwargs["evaluation_goal_idx"] = [11]
            receipt = build_receipt(**kwargs)
        self.assertEqual(receipt["status"], "blocked")
        self.assertIn("train/evaluation goal sets overlap", receipt["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
