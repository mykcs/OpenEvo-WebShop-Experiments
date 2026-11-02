"""Offline regressions for only-GPU6/7 Q17 metric launcher (no model inference)."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"scripts/analysis/q17_gpu67_metric_campaign.py"
PLAN=ROOT/"configs/experiment/metric-campaigns/q17-gpu67-20261012.json"
spec=importlib.util.spec_from_file_location("q17_gpu67_metric_campaign",SOURCE)
assert spec is not None and spec.loader is not None
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Q17GPU67Tests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads(PLAN.read_text())
        runner.verify_plan(self.p)

    def test_fixed_only_6_and_7(self):
        self.assertEqual([x["index"] for x in self.p["device_allowlist"]],[6,7])
        self.assertEqual(set(runner.ALLOWED),{6,7})

    def test_other_device_never_enters_plan(self):
        for device in range(6):
            obj=copy.deepcopy(self.p)
            obj["device_allowlist"][0]["index"]=device
            with self.assertRaises(runner.GateError):
                runner.verify_plan(obj)

    def test_swapped_uuids_block(self):
        obj=copy.deepcopy(self.p)
        obj["device_allowlist"][0]["uuid"]=runner.ALLOWED[7]
        with self.assertRaises(runner.GateError):
            runner.verify_plan(obj)

    def test_task_budget_drift_blocks(self):
        obj=copy.deepcopy(self.p)
        obj["budgets"]["maximum_new_development_rollouts"]=129
        with self.assertRaises(runner.GateError):
            runner.verify_plan(obj)

    def test_final_panel_flag_blocks(self):
        obj=copy.deepcopy(self.p)
        obj["science"]["no_new_formal_or_final_tasks"]=False
        with self.assertRaises(runner.GateError):
            runner.verify_plan(obj)

    def test_container_args_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for arm,index in (("r159",6),("r129",7)):
                for mode in ("qualify","dev64"):
                    argv=runner.container_command(self.p,root,root/"out",arm,kind=mode)
                    self.assertEqual(argv[argv.index("--gpus")+1],"device="+runner.ALLOWED[index])
                    self.assertIn("CUDA_VISIBLE_DEVICES="+runner.ALLOWED[index],argv)
                    self.assertIn("NVIDIA_VISIBLE_DEVICES="+runner.ALLOWED[index],argv)
                    self.assertNotIn("--privileged",argv)
                    self.assertNotIn("--gpus-all",argv)
                    if mode=="dev64":
                        self.assertIn("--q06-pre-action-diagnostics",argv)
                        self.assertIn("--instrumentation",argv)

    def test_occupied_7_blocks_without_launch(self):
        rows=[]
        for i in range(8):
            uid=runner.ALLOWED.get(i,"GPU-unused-%d"%i)
            rows.append(f"{i}, {uid}, NVIDIA GeForce RTX 5090, 4, 32607, 0")
        with patch.object(runner,"subprocess_checked",side_effect=[
            "\n".join(rows),runner.ALLOWED[7]+", 4122"]):
            with self.assertRaisesRegex(runner.GateError,"GPU7 has a running compute-app"):
                runner.live_gpu_gate(self.p)

    def test_model_qualification_exact_bind(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for arm in ("r159","r129"):
                dst=root/arm/"qualification.json";dst.parent.mkdir()
                dst.write_text(json.dumps({
                    "status":"PASS","adapter_sha256":self.p["arms"][arm]["adapter_sha256"],
                    "generation_calls":0,"final_panel_access":0,
                    "formal_rollouts_consumed":0,"inference_started":False}))
            runner.require_ready_qualifications(self.p,root)
            (root/"r129/qualification.json").write_text('{"status":"PASS"}')
            with self.assertRaises(runner.GateError):
                runner.require_ready_qualifications(self.p,root)

    def test_default_is_read_only_preflight(self):
        out=io.StringIO()
        with patch.object(runner,"cpu_source_check",return_value={"gpu_calls":0}),\
             patch.object(runner,"live_gpu_gate",return_value=[{"index":6},{"index":7}]),\
             patch.object(runner,"docker_image_gate",return_value=self.p["image_id"]),\
             redirect_stdout(out):
            rc=runner.main(["--repo-root",str(ROOT),"--plan",str(PLAN)])
        self.assertEqual(rc,0)
        self.assertEqual(json.loads(out.getvalue())["status"],"STATIC_PREFLIGHT_PASS_LARGE_SHA_PENDING")

    def test_dev64_requires_explicit_allowance(self):
        err=io.StringIO()
        with patch.object(runner,"cpu_source_check",return_value={"gpu_calls":0}),\
             patch.object(runner,"live_gpu_gate",return_value=[{"index":6},{"index":7}]),\
             patch.object(runner,"docker_image_gate",return_value=self.p["image_id"]),\
             redirect_stderr(err):
            rc=runner.main(["--repo-root",str(ROOT),"--plan",str(PLAN),
                            "--phase","dev64","--run-id","test123"])
        self.assertEqual(rc,2)
        self.assertIn("--approve-development-rollouts",err.getvalue())


if __name__=="__main__":
    unittest.main()
