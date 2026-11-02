import importlib.util,json,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT/'scripts/openevo_webshop/ceiling1_ray_gpu_pool_202609042025.py'
BACK=ROOT/'scripts/openevo_webshop/ceiling1_ray_placement_backend_202609042025.py'
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
pool=loadmod('ce1_pool_tx',POOL);back=loadmod('ce1_back_tx',BACK)
class FakeRay:
 def shutdown(self): pass
class T(unittest.TestCase):
 def test_host_pool_lock_is_exclusive(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'pool.lock';a=pool.acquire_host_pool_lock(p)
   try:
    with self.assertRaisesRegex(RuntimeError,'host lock already held'):pool.acquire_host_pool_lock(p)
   finally:a.close()
 def test_overlap_scan_catches_legacy_pool(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cfg=root/'legacy.json';cfg.write_text(json.dumps({'managed_gpus':[4,5,6,7]}))
   d=root/'1234';d.mkdir();(d/'cmdline').write_bytes(b'python3\0/x/ceiling1_ray_gpu_pool_legacy.py\0--config\0'+str(cfg).encode()+b'\0')
   got=pool.overlapping_pool_processes([4,5,6,7],root,self_pid=9999)
   self.assertEqual(got[0]['pid'],1234);self.assertEqual(got[0]['overlap'],[4,5,6,7])
 def test_recorded_launching_container_counts_live(self):
  old=pool.container_running;pool.container_running=lambda n:n=='launching-c'
  try:self.assertEqual(pool.recorded_live_containers({'gpus':{'4':{'leases':[]}},'reservations':{'r':{'launching':True,'container':'launching-c'}}}),['launching-c'])
  finally:pool.container_running=old
 def test_reconcile_promotes_live_launching_reservation(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cfgp=root/'cfg.json';cfg={'managed_gpus':[4],'launch_transaction_grace_seconds':10};cfgp.write_text(json.dumps(cfg));h=back.sha(cfgp)
   state={'schema':'openevo.ceiling1-ray-placement-state.202609042025','config_sha256':h,'gpus':{'4':{'leases':[]}},'reservations':{'rid':{'placement_group_name':'pg','workload':'stage2_sd_lora_3b','gpus':[4],'launching':True,'launching_at_epoch':time.time()-2,'lease_id':'L','container':'C','vram_request_mib_per_gpu':11000}},'admission':{'paused':False}}
   back.dump(root/'state.json',state);old=(back.ray_api,back.running,back.spawn_recovered_watcher);seen=[]
   back.ray_api=lambda *_:(FakeRay(),{},None,lambda n:n,lambda pg:None,None);back.running=lambda n:n=='C';back.spawn_recovered_watcher=lambda c,r,l:seen.append(l['lease_id'])
   try:back.reconcile(cfg,root,cfgp)
   finally:back.ray_api,back.running,back.spawn_recovered_watcher=old
   x=back.load(root/'state.json');self.assertEqual(x['reservations'],{});lease=x['gpus']['4']['leases'][0];self.assertEqual(lease['lease_id'],'L');self.assertTrue(lease['recovered_from_launching_reservation']);self.assertEqual(seen,['L'])
 def test_reconcile_cleans_dead_launch_after_grace(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cfgp=root/'cfg.json';cfg={'managed_gpus':[4],'launch_transaction_grace_seconds':1};cfgp.write_text(json.dumps(cfg));h=back.sha(cfgp)
   state={'schema':'openevo.ceiling1-ray-placement-state.202609042025','config_sha256':h,'gpus':{'4':{'leases':[]}},'reservations':{'rid':{'placement_group_name':'pg','workload':'w','gpus':[4],'launching':True,'launching_at_epoch':time.time()-5,'lease_id':'L','container':'C','vram_request_mib_per_gpu':1}},'admission':{'paused':False}}
   back.dump(root/'state.json',state);removed=[];old=(back.ray_api,back.running)
   back.ray_api=lambda *_:(FakeRay(),{},None,lambda n:n,lambda pg:removed.append(pg),None);back.running=lambda n:False
   try:back.reconcile(cfg,root,cfgp)
   finally:back.ray_api,back.running=old
   self.assertEqual(back.load(root/'state.json')['reservations'],{});self.assertEqual(removed,['pg'])
if __name__=='__main__':unittest.main()
