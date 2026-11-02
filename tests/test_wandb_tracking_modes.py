import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.openevo_webshop.wandb_tracking import (
    WandbTracker,
    campaign_tracking_config,
    load_current_campaign,
    normalize_wandb_mode,
    start_wandb_run,
    short_run_name,
    standard_tags,
    trainer_wandb_environment,
)


class WandbModeTest(unittest.TestCase):
    def test_mode_normalization(self):
        self.assertEqual(normalize_wandb_mode(None), "online")
        self.assertEqual(normalize_wandb_mode("online"), "online")
        self.assertEqual(normalize_wandb_mode("no"), "disabled")
        with self.assertRaises(ValueError):
            normalize_wandb_mode("sometimes")

    def test_short_run_name_keeps_campaign_timestamp_out_of_legend(self):
        self.assertEqual(short_run_name(arm="adapter1x", unit="trainer", seed=1993), "adapter1x-trainer-s1993")

    def test_campaign_metadata_drives_group_config_and_namespaced_tags(self):
        payload = {
            "standard_core": "v1",
            "campaign_id": "20260814-1431-phaseg-base-vs-adapters",
            "status": "preflight",
            "title": "Phase G comparison",
            "scientific_question": "Do adapters change WebShop outcomes?",
            "benchmark": "webshop",
            "phase": "g",
            "split": "promotion-dev",
            "tracking": {
                "group": "20260814-1431-phaseg-base-vs-adapters",
                "entity": "example-entity",
                "project": "openevo-webshop-experiments",
                "default_mode": "online"
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            campaign = load_current_campaign(path)
        self.assertIsNotNone(campaign)
        self.assertEqual(campaign_tracking_config(campaign)["campaign_id"], payload["campaign_id"])
        self.assertEqual(
            standard_tags(campaign, method="sd-lora", arm="adapter1x"),
            ("benchmark:webshop", "phase:g", "split:promotion-dev", "method:sd-lora", "arm:adapter1x"),
        )

    def test_long_tags_are_bounded_deterministically(self):
        campaign = {"benchmark": "webshop", "phase": "H1.29", "split": "train-only-self-generated-success versus fresh-unused-evaluation"}
        first = standard_tags(campaign, extra=("custom:" + "x" * 100,))
        second = standard_tags(campaign, extra=("custom:" + "x" * 100,))
        self.assertEqual(first, second)
        self.assertTrue(all(1 <= len(tag) <= 64 for tag in first))
        self.assertEqual(len(first[2]), 64)
        self.assertTrue(first[2].startswith("split:"))
        self.assertRegex(first[2][-9:], r"-[0-9a-f]{8}")

    def test_trainer_environment_uses_auto_run_id_and_explicit_campaign_metadata(self):
        tracker = WandbTracker(mode="online", project="openevo-webshop-experiments", run=SimpleNamespace(id="wrapper123"))
        keys = {
            "OPENEVO_WANDB_ENABLE", "OPENEVO_WANDB_RUN_ID", "OPENEVO_WANDB_PARENT_RUN_ID",
            "OPENEVO_WANDB_JOB_TYPE", "OPENEVO_CAMPAIGN_ID", "WANDB_RUN_ID", "WANDB_RUN_GROUP",
            "WANDB_NAME", "WANDB_NOTES", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"
        }
        with patch.dict(os.environ, {}, clear=False):
            previous = {key: os.environ.get(key) for key in keys}
            with trainer_wandb_environment(
                tracker,
                mode="online",
                project="openevo-webshop-experiments",
                entity="example-entity",
                group="20260814-1431-phaseg-base-vs-adapters",
                run_name="adapter1x-trainer-s1993",
                campaign_id="20260814-1431-phaseg-base-vs-adapters",
                notes="bounded smoke",
            ):
                self.assertEqual(os.environ["OPENEVO_WANDB_ENABLE"], "1")
                self.assertNotIn("OPENEVO_WANDB_RUN_ID", os.environ)
                self.assertNotIn("WANDB_RUN_ID", os.environ)
                self.assertEqual(os.environ["OPENEVO_WANDB_PARENT_RUN_ID"], "wrapper123")
                self.assertEqual(os.environ["OPENEVO_WANDB_JOB_TYPE"], "trainer")
                self.assertEqual(os.environ["OPENEVO_CAMPAIGN_ID"], "20260814-1431-phaseg-base-vs-adapters")
                self.assertEqual(os.environ["WANDB_RUN_GROUP"], "20260814-1431-phaseg-base-vs-adapters")
                self.assertEqual(os.environ["WANDB_NAME"], "adapter1x-trainer-s1993")
                self.assertEqual(os.environ["WANDB_NOTES"], "bounded smoke")
                self.assertEqual(os.environ["WANDB_MODE"], "online")
                self.assertEqual(os.environ["WANDB_PROJECT"], "openevo-webshop-experiments")
                self.assertEqual(os.environ["WANDB_ENTITY"], "example-entity")
            self.assertEqual({key: os.environ.get(key) for key in keys}, previous)

    def test_disabled_trainer_environment_does_not_enable_wandb(self):
        tracker = WandbTracker(mode="disabled", project="test")
        with trainer_wandb_environment(tracker, mode="disabled", project="test", entity=None, group="disabled-test", run_name="disabled-trainer"):
            self.assertEqual(os.environ["OPENEVO_WANDB_ENABLE"], "0")
            self.assertEqual(os.environ["WANDB_MODE"], "disabled")

    def test_offline_fallback_propagates_to_trainer_child(self):
        tracker = WandbTracker(
            mode="offline",
            project="test",
            run=SimpleNamespace(id="offline123"),
            fallback_reason="missing API key",
        )
        with trainer_wandb_environment(
            tracker,
            mode="online",
            project="test",
            entity=None,
            group="offline-fallback",
            run_name="offline-trainer",
        ):
            self.assertEqual(os.environ["OPENEVO_WANDB_ENABLE"], "1")
            self.assertEqual(os.environ["WANDB_MODE"], "offline")

    def test_online_initialization_retries_offline_with_hashed_receipt(self):
        calls = []

        class FakeRun:
            id = "offline123"
            entity = "entity"
            path = "entity/project/offline123"
            url = None
            name = "test"
            group = "group"
            job_type = "smoke"
            dir = None
            offline = True
            settings = None
            summary = {}

            def finish(self):
                return None

        class FakeWandb:
            @staticmethod
            def init(**kwargs):
                calls.append(kwargs["mode"])
                if kwargs["mode"] == "online":
                    raise RuntimeError("network unavailable")
                return FakeRun()

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"wandb": FakeWandb}):
            tracker = start_wandb_run(
                mode="online",
                project="project",
                entity="entity",
                name="test",
                job_type="smoke",
                config={},
                directory=Path(tmp),
            )
            self.assertEqual(tracker.mode, "offline")
            self.assertEqual(calls, ["online", "offline"])
            self.assertIn("network unavailable", tracker.metadata()["fallback_reason"])
            tracker.finish()
            receipt = json.loads((Path(tmp) / "wandb-fallback-receipt.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["spool_sha256"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
