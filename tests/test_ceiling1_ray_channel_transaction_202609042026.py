from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[1]
class T(unittest.TestCase):
 def test_shared_lock(self):
  s=(ROOT/'scripts/openevo_webshop/run_ceiling1_stage2_ray_202609042025.py').read_text()
  for token in ('fcntl.LOCK_EX','materialized_success','CEILING1_STAGE2_SD_RECEIPT','ray-launch.lock','rc==0 or materialized_success()'):
   self.assertIn(token,s)
  self.assertIn("execution_source_lock_path(locks,self.execution_git_sha,resource_migration_sha)",s)
  self.assertIn("resource_migration_sha[:16]",s)
 def test_authority(self):
  x=json.loads((ROOT/'configs/experiment/openevo-ceiling1-stage2-runtime-authority-202609042025.json').read_text())['control_plane_guarantees']
  self.assertTrue(x['per_channel_launch_transaction_lock']);self.assertTrue(x['materialized_success_prevents_duplicate_gpu_recall'])
if __name__=='__main__':unittest.main()
