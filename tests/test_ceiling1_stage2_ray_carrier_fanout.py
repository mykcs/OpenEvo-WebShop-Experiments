from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class RayCarrierFanoutStaticTest(unittest.TestCase):
 def test_four_independent_candidate_channels_fan_out_then_barrier(self):
  s=(ROOT/'scripts/openevo_webshop/run_ceiling1_stage2_vnext_arm.py').read_text()
  self.assertIn('ThreadPoolExecutor(max_workers=4',s)
  for expr in ("pool.submit(self.reflect,r,rr,'text_memory'","pool.submit(self.run_sd,r,rr)","pool.submit(self.reflect,r,rr,'skill_bundle'","pool.submit(self.reflect,r,rr,'agent_system'"):
   self.assertIn(expr,s)
  self.assertIn('new_mem=f_mem.result();new_sd,new_sd_id,new_count,sd_updated=f_sd.result();new_skill=f_skill.result();new_agent=f_agent.result()',s)
 def test_candidate_probe_remains_after_fanout_barrier(self):
  s=(ROOT/'scripts/openevo_webshop/run_ceiling1_stage2_vnext_arm.py').read_text()
  self.assertLess(s.index('new_mem=f_mem.result()'),s.index('candidate_probe=self.candidate_probe'))
if __name__=='__main__': unittest.main()
