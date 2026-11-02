import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('ceiling1_stage2_reflector', ROOT/'scripts/openevo_webshop/ceiling1_stage2_reflector.py')
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

class ReflectorSaturationTest(unittest.TestCase):
    def test_hard_cap_is_mechanical_invalid(self):
        valid_agent = '# Agent System\n## Trigger\nobserve\n## Action\nact\n## Validation\nverify\n'
        self.assertTrue(MOD.mechanically_valid('agent_system',valid_agent,1023,1024))
        self.assertFalse(MOD.mechanically_valid('agent_system',valid_agent,1024,1024))

    def test_skill_structure_still_required(self):
        self.assertFalse(MOD.mechanically_valid('skill_bundle','# SKILL.md',100,1024))
        self.assertTrue(MOD.mechanically_valid('skill_bundle','Trigger Procedure Validation Evidence',100,1024))

if __name__=='__main__': unittest.main()
