import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("ceiling1_stage2_round_driver", ROOT/"scripts/openevo_webshop/ceiling1_stage2_round_driver.py")
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

class RoundIdentityTest(unittest.TestCase):
    def test_rollout_suffix_is_not_task_identity(self):
        rows=[]
        for task in range(16):
            for rollout in range(8):
                rows.append({"worker_seed":0,"source_session_index":task,"instruction_sha256":f"h{task}","task_id":f"group{task}-r{rollout}"})
        self.assertEqual(len({MOD.task_identity(r) for r in rows}),16)
        self.assertEqual(len({r["task_id"] for r in rows}),128)

if __name__=="__main__": unittest.main()
