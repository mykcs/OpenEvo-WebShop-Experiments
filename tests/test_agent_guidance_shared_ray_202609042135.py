from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TestSharedRayAgentGuidance(unittest.TestCase):
    def test_current_gpu_router_points_to_timestamped_shared_small_arm_authority(self):
        text=(ROOT/'docs/operations/execution/CEILING1_GPU_SCHEDULING_CURRENT.md').read_text()
        self.assertIn('openevo-ceiling1-stage2-runtime-authority-202609042025.json',text)
        self.assertIn('Qwen2.5-3B + Qwen3-1.7B',text)
        self.assertIn('A merge to `main` does not migrate a formal run',text)
        self.assertNotIn('7B has scheduling priority',text)

    def test_startup_guidance_promotes_shared_resource_truth(self):
        agents=(ROOT/'AGENTS.md').read_text()
        runbook=(ROOT/'docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md').read_text()
        self.assertIn('Schedule ready work, not phase names',agents)
        self.assertIn('One physical GPU workload has one recoverable resource truth',agents)
        self.assertIn('Producer/runtime validator parity is mandatory',runbook)
        self.assertIn('Repository-current is not live-current',runbook)

    def test_new_case_is_discoverable_through_agent_and_troubleshooting_routers(self):
        name='CASE-CEILING1-RAY-SHARED-ARM-AUTHORITY-RETROSPECTIVE-202609042135.md'
        self.assertTrue((ROOT/'docs/troubleshooting/experiment-ops'/name).is_file())
        agent_router=(ROOT/'docs/agents/README.md').read_text()
        troubleshooting_router=(ROOT/'docs/troubleshooting/README.md').read_text()
        self.assertIn('../troubleshooting/README.md',agent_router)
        self.assertIn(name,troubleshooting_router)

if __name__=='__main__': unittest.main()
