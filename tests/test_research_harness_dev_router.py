"""Ensure the research Harness Dev entrance stays task-scoped; no GPU or server calls."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
DEV=(ROOT/"docs/dev/README.md")
class ResearchHarnessRoutingTests(unittest.TestCase):
    def test_dev_is_not_a_fake_skill_or_scientific_authority(self):
        text=DEV.read_text(encoding="utf-8")
        self.assertIn("OpenEVO Research Harness",text)
        self.assertIn("WORKFLOW_PROMPTS.md#openevo-research-harness",text)
        self.assertIn("不复制一份可编辑的 Prompt",text)
        self.assertIn("never scientific execution authority",text)
    def test_three_owner_cross_repo_dispatch(self):
        text=DEV.read_text(encoding="utf-8")
        for marker in ("Experiment 负责实验定义", "mykcs/OpenEVO", "mykcs/zju-server",
            "只读对应权威", "this list is navigation, not an instruction to preload every file"):
            self.assertIn(marker,text)
    def test_prompt_does_not_grant_gpu_or_delete(self):
        text=DEV.read_text(encoding="utf-8")
        self.assertIn("短 Prompt 不授予新 GPU/删除/恢复训练权限",text)
        self.assertIn("最新 GPU 授权与实际占用须从用户及主机现行规则核实",text)

if __name__=="__main__":
    unittest.main()
