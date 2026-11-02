import json
import tempfile
import unittest
from pathlib import Path

from scripts.openevo_webshop.package_h138ac_publication import (
    build_bundle,
    classify_path,
    sha256_file,
)


class H138ACPublicationTests(unittest.TestCase):
    def test_classification_keeps_raw_and_tracking_payloads_server_only(self):
        self.assertEqual(
            classify_path("evaluation/diversity-8/task1/attempts.jsonl"),
            "raw-attempt-jsonl",
        )
        self.assertEqual(
            classify_path("evaluation/diversity-8/task1/episodes/a.json"),
            "raw-episode",
        )
        self.assertEqual(
            classify_path("evaluation/diversity-8/task1/wandb/run-a.wandb"),
            "wandb-state",
        )
        self.assertEqual(
            classify_path("evaluation/diversity-8/task1/summary.json"),
            "compact-summary",
        )

    def test_bundle_embeds_only_summary_and_gpu_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "selected" / "evaluation" / "diversity-8" / "task7"
            receipt = cell / "wandb" / "wandb-fallback-receipt.json"
            receipt.parent.mkdir(parents=True)
            payloads = {
                cell / "summary.json": {
                    "arm": "diversity-8",
                    "attempt_count": 4,
                    "valid_count": 4,
                    "tracking": {"mode": "offline"},
                },
                cell / "gpu-before.json": {"expected_uuid": "GPU-test", "compute_apps": []},
                cell / "gpu-after.json": {"expected_uuid": "GPU-test", "compute_apps": ["python"]},
                receipt: {"effective_mode": "offline", "spool_sha256": "sha256:" + "a" * 64},
            }
            for path, payload in payloads.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            selected = root / "selected"
            ledger = root / "ledger.tsv"
            rows = []
            for path in sorted(payloads):
                relative = path.relative_to(selected).as_posix()
                rows.append(f"{sha256_file(path)}\t{path.stat().st_size}\t{relative}")
            ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")

            bundle, coverage = build_bundle(
                selected,
                ledger,
                observed_at_utc="2026-08-19T00:00:00Z",
                server_root="/server/run",
                scientific_contract_sha="a" * 40,
                execution_sha="b" * 40,
                expected_arms={"diversity-8": 1},
            )

            self.assertEqual(bundle["cell_count"], 1)
            self.assertEqual(bundle["published_payload_file_count"], 3)
            self.assertNotIn("tracking_receipt", bundle["cells"][0])
            self.assertEqual(
                bundle["cells"][0]["tracking_receipt_publication"],
                "hash-only; payload remains server-only",
            )
            self.assertEqual(
                coverage["groups"]["tracking-receipt-hash-only"]["file_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
