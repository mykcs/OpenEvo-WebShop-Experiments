"""Offline fail-closed regressions for Q17 W2 recovered R126/R150 GPU6/7 runner."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"scripts/analysis/q17_w2_recovered_r126_r150_gpu67.py"
S=importlib.util.spec_from_file_location("q17_w2_recovered_runner",P)
assert S and S.loader
q=importlib.util.module_from_spec(S)
S.loader.exec_module(q)
PLAN=ROOT/"configs/experiment/metric-campaigns/q17-w2-r126-r150-gpu67-20261013.json"

class Tests(unittest.TestCase):
 def setUp(self):
  self.m=json.loads(PLAN.read_bytes())
  q.verify_plan(self.m)

 def test_exact_scientific_arm_ids_and_gpu6_7(self):
  self.assertEqual(set(self.m["arms"]),{"r126","r150"})
  self.assertEqual([self.m["arms"][a]["index"] for a in q.ARMS],[6,7])
  self.assertEqual(self.m["safety"]["forbidden_gpu_indices"],[0,1,2,3,4,5])

 def test_forged_gpu0_5_never_accepted(self):
  for bad in range(6):
   m=copy.deepcopy(self.m)
   m["arms"]["r126"]["index"]=bad
   with self.assertRaises(q.Hold):q.verify_plan(m)

 def test_bad_gpu_uuid_never_accepted(self):
  m=copy.deepcopy(self.m)
  m["safety"]["exact_gpu_uuid"][1]="GPU-bogus"
  with self.assertRaises(q.Hold):q.verify_plan(m)

 def test_extra_rollout_not_accepted(self):
  m=copy.deepcopy(self.m)
  m["safety"]["maximum_new_development_episodes"]=129
  with self.assertRaises(q.Hold):q.verify_plan(m)

 def test_unsealed_final_panel_not_accepted(self):
  m=copy.deepcopy(self.m)
  m["safety"]["final_panel_access"]=1
  with self.assertRaises(q.Hold):q.verify_plan(m)

 def test_wrong_reconstructed_adapter_digest_not_accepted(self):
  m=copy.deepcopy(self.m)
  m["arms"]["r150"]["adapter_sha256"]="0"*64
  with self.assertRaises(q.Hold):q.verify_plan(m)

 def test_docker_device_visibility_does_not_broaden(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for arm,uuid in [("r126",q.EXPECTED_UUID[6]),("r150",q.EXPECTED_UUID[7])]:
    for kind in ("qualify","dev64"):
     args=q.run_args(self.m,root,arm,kind)
     self.assertEqual(args[args.index("--gpus")+1],"device="+uuid)
     self.assertIn("CUDA_VISIBLE_DEVICES="+uuid,args)
     self.assertIn("NVIDIA_VISIBLE_DEVICES="+uuid,args)
     self.assertNotIn("--privileged",args)
     self.assertNotIn("device=all",args)
     if kind=="dev64":
      self.assertIn("--q06-pre-action-diagnostics",args)
      self.assertIn("--instrumentation",args)

 def test_new_manifest_has_original_panel_and_no_final(self):
  dummy={
   "checkpoints":{"after_160_updates":{
       "adapter_sha256":"18032c5cabd135b2464d5cbb2a3949aeb6130046dcc9f33acb33aee38c1baf46"}},
   "tasks":[],"panel_sha256":self.m["frozen_hashes"]["panel_id"],
   "final_panel_access_allowed":False,"formal_task_consumption_allowed":False
  }
  with patch.object(q,"read",return_value=dummy):
   res=q.derive_manifest(self.m,"r126")
  self.assertEqual(res["completed_stage2_rounds"],127)
  self.assertEqual(res["checkpoints"]["after_127_updates"]["adapter_sha256"],q.EXPECTED_ADAPTER["r126"])
  self.assertFalse(res["final_panel_access_allowed"])

 def test_no_implicit_adapter_load_against_bad_qualifier(self):
  with tempfile.TemporaryDirectory() as td:
   work=Path(td)
   p=work/"r150";p.mkdir()
   (p/"qualification.json").write_text(json.dumps({
      "status":"PASS","adapter_sha256":q.EXPECTED_ADAPTER["r126"],
      "generation_calls":0,"final_panel_access":0,
      "formal_rollouts_consumed":0,"inference_started":False}))
   with self.assertRaises(q.Hold):q.native_qualify(self.m,work,"r150")

if __name__=="__main__":
 unittest.main()
