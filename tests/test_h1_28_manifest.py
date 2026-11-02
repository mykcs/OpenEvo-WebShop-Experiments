import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.openevo_webshop.build_h1_28_manifest import (
    collect_prior_goal_indices,
    select_indices,
)


class H128ManifestTest(unittest.TestCase):
    def test_collect_prior_goal_indices_uses_only_task_identity_keys(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            prior = tmp_path / "prior.json"
            prior.write_text(json.dumps({
                "task_ids": [505, "webshop_000506"],
                "tasks": [{"goal_idx": 507}],
                "unrelated_count": 508,
            }))

            used, sources = collect_prior_goal_indices([tmp_path])

            self.assertEqual(used, {505, 506, 507})
            self.assertEqual(sources[505], [str(prior)])
            self.assertNotIn(508, used)

    def test_selection_is_deterministic_and_excludes_prior_tasks(self):
        first = select_indices(700, {500, 501, 502}, selection_seed=9)
        second = select_indices(700, {500, 501, 502}, selection_seed=9)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertFalse(set(first) & {500, 501, 502})


if __name__ == "__main__":
    unittest.main()
