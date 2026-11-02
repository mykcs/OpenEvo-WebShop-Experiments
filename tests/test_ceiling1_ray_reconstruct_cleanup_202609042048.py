import importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'
spec=importlib.util.spec_from_file_location('backend_cleanup',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Group:
 def ready(self):return 'ready'
class Ray:
 def wait(self,refs,timeout=None):return (refs,[])
class T(unittest.TestCase):
 def fixture(self):
  td=tempfile.TemporaryDirectory();root=Path(td.name);cp=root/'cfg.json';cfg={'managed_gpus':[4],'reservation_ttl_seconds':180,'rollout_strategy':'STRICT_SPREAD','single_gpu_strategy':'PACK','placement_ready_timeout_seconds':2.0};cp.write_text(json.dumps(cfg));sha=m.sha(cp);m.dump(root/'state.json',{'schema':'openevo.ceiling1-ray-placement-state.202609042025','config_sha256':sha,'gpus':{'4':{'leases':[]}},'reservations':{},'admission':{'paused':False}});sel=root/'sel.json';sel.write_text('{}');a=SimpleNamespace(config=cp,state_root=root,workload='w',selection_receipt=sel);x={'reservation_id':'rid','placement_group_name':'pg','selected_gpus':[4]};return td,root,cfg,a,x
 def test_live_gate_failure_removes_reconstructed_pg(self):
  td,root,cfg,a,sel=self.fixture();calls=[0];removed=[];old=m.live_gate
  def getpg(_):
   calls[0]+=1
   if calls[0]==1:raise RuntimeError('absent')
   return 'pg'
  m.live_gate=lambda *_:(_ for _ in ()).throw(RuntimeError('physical gate'))
  try:
   with self.assertRaisesRegex(RuntimeError,'physical gate'):m.ensure_selection_reservation(a,cfg,{'arm':'3b','vram_request_mib':1},sel,Ray(),lambda *_,**__:Group(),getpg,lambda pg:removed.append(pg))
  finally:m.live_gate=old
  self.assertEqual(removed,['pg']);self.assertEqual(m.load(root/'state.json')['reservations'],{});td.cleanup()
 def test_failed_pg_cleanup_stays_visible(self):
  td,root,cfg,a,sel=self.fixture();calls=[0];old=m.live_gate
  def getpg(_):
   calls[0]+=1
   if calls[0]==1:raise RuntimeError('absent')
   return 'pg'
  m.live_gate=lambda *_:(_ for _ in ()).throw(RuntimeError('physical gate'))
  try:
   with self.assertRaisesRegex(RuntimeError,'physical gate'):m.ensure_selection_reservation(a,cfg,{'arm':'3b','vram_request_mib':1},sel,Ray(),lambda *_,**__:Group(),getpg,lambda _:(_ for _ in ()).throw(RuntimeError('remove failed')))
   state=m.load(root/'state.json');self.assertTrue(state['reservations']['rid']['cleanup_pending'])
  finally:m.live_gate=old;td.cleanup()
if __name__=='__main__':unittest.main()
