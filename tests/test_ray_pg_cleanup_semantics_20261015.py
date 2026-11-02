"""CPU-only regressions for real Ray 2.57.0 placement-group cleanup semantics.

Do not infer "already absent" from an unavailable Ray/GCS connection.  This
file does not run Ray, Docker, nvidia-smi, SSH, or any scientific workload.
The separate opt-in real-Ray qualification runs only in a disposable CPU pool.
"""
from __future__ import annotations

import unittest

from scripts.openevo_webshop import ceiling1_ray_placement_backend_202609042025 as backend


class RayPlacementCleanupContractTests(unittest.TestCase):
    def test_ray_named_group_absent_value_error_is_idempotently_clean(self):
        removed = []
        def missing(name):
            raise ValueError("Failed to look up placement group with name: " + name)
        self.assertTrue(backend.remove_pg_quiet(missing, removed.append, "absent"))
        self.assertEqual(removed, [])

    def test_ray_gcs_disconnection_is_not_reported_as_successful_release(self):
        removed = []
        def disconnected(_name):
            raise ConnectionError("GCS unavailable")
        self.assertFalse(backend.remove_pg_quiet(disconnected, removed.append, "possibly-live"))
        self.assertEqual(removed, [])

    def test_provider_lookup_runtime_error_is_not_an_absence_receipt(self):
        def broken(_name):
            raise RuntimeError("Ray server temporarily unavailable")
        self.assertFalse(backend.remove_pg_quiet(broken, lambda _: None, "possibly-live"))

    def test_removal_error_remains_pending(self):
        def failed_remove(_group):
            raise RuntimeError("cannot release a live PG")
        self.assertFalse(backend.remove_pg_quiet(lambda name: name, failed_remove, "live-pg"))

    def test_available_group_requests_a_removal(self):
        removed = []
        self.assertTrue(backend.remove_pg_quiet(lambda name: name, removed.append, "owned-pg"))
        self.assertEqual(removed, ["owned-pg"])


if __name__ == "__main__":
    unittest.main()
