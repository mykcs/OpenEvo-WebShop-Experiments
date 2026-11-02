"""Zero-GPU tests of 8-repeat metric definitions; fail closed on identity drift."""
import importlib.util
from pathlib import Path
import tempfile
import json
import hashlib
import unittest

PROJECT=Path(__file__).resolve().parents[1]
TARGET=PROJECT/"scripts/analysis/q17_r159_repeat8_cpu_reconcile_20261013.py"
spec=importlib.util.spec_from_file_location("q17_r159_repeat8",TARGET)
module=importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

def fake_episode(won:bool,score=None):
    return {"episode":{"won":won,"score":float(won) if score is None else score}}

class RepeatTests(unittest.TestCase):
    def test_per_task_product_not_naive_global_p_power(self):
        rows=[]
        for i in range(64):
            if i<32:
                bits=[True]*8
            else:
                bits=[False]*8
            rows.append(tuple(fake_episode(x) for x in bits))
        m=module.rate_metrics(rows)
        self.assertEqual(m["pass_all_k_by_exact_seed_prefix"]["8"],.5)
        self.assertEqual(m["naive_p_to_8_is_not_primary"],.5**8)
        self.assertEqual(m["success_at_least_once_k_diagnostic"]["8"],.5)

    def test_unstable_families_and_histogram(self):
        rows=[]
        for i in range(64):
            if i<2: bits=[False]*8
            elif i<5:bits=[True]*8
            else:bits=[True]*4+[False]*4
            rows.append(tuple(fake_episode(x) for x in bits))
        m=module.rate_metrics(rows)
        self.assertEqual(m["success_count_per_task_histogram"]["0"],2)
        self.assertEqual(m["success_count_per_task_histogram"]["8"],3)
        self.assertEqual(m["unreliable_some_but_not_all_success_tasks"],59)
        self.assertAlmostEqual(m["pass_all_k_by_exact_seed_prefix"]["8"],3/64)

    def test_bootstrap_seed_is_reproducible(self):
        rows=[tuple(fake_episode(j<((i%9))) for j in range(8)) for i in range(64)]
        a=module.boot_ci(rows,100)
        b=module.boot_ci(rows,100)
        self.assertEqual(a,b)
        self.assertEqual(a["bootstrap_repetitions"],100)

    def test_missing_one_repetition_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(FileNotFoundError):
                module.load(root/"missing-summary.json",root)

    def test_sha_protection(self):
        self.assertEqual(module.sha(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

if __name__=="__main__":
    unittest.main()
