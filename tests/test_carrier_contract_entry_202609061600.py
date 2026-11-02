"""CPU-only regression tests for the frozen carrier lifecycle contract."""
import ast
from pathlib import Path
import unittest
from scripts.openevo_webshop.ceiling1_harness201_carrier_contract import (require_evolution_compatible_bundle, validate_bundle)
ROOT=Path(__file__).resolve().parents[1]
MEMORY="# Memory\n## Reusable Knowledge\n- Knowledge: Explicit attributes determine product compatibility.\n"
SKILL="# Skill Bundle\n## Search State\n- When constraints are provided, verify the attributes.\n## Results State\n- No additional guidance.\n## Product State\n- No additional guidance.\n## Any State\n- No additional guidance.\n"
AGENT="# Agent System\n## Authority\n- Authority: Current observation determines current facts.\n## Coordination\n- Coordination: Apply the relevant grounded knowledge.\n"
class CarrierEntryContractTests(unittest.TestCase):
    def check(self,m=MEMORY,s=SKILL,a=AGENT,**kw):
        return require_evolution_compatible_bundle(m,s,a,context="test",**kw)
    def test_valid_seed_is_accepted_by_candidate_contract(self):
        self.assertEqual(self.check(),validate_bundle(MEMORY,SKILL,AGENT))
    def test_q17_results_guidance_is_reported_at_entry(self):
        bad=SKILL.replace("## Results State\n- No additional guidance.","## Results State\n- When constraints are provided, verify the attributes.")
        with self.assertRaisesRegex(ValueError,"Results State"): self.check(s=bad)
    def test_q17_two_authority_bullets_are_reported_at_entry(self):
        bad=AGENT.replace("## Coordination","- Authority: Historical carriers provide guidance.\n## Coordination")
        with self.assertRaisesRegex(ValueError,"Authority must contain exactly one"): self.check(a=bad)
    def test_3b_two_coordination_bullets_are_reported_at_entry(self):
        with self.assertRaisesRegex(ValueError,"Coordination must contain exactly one"):
            self.check(a=AGENT+"- Coordination: Maintain grounded knowledge.\n")
    def test_3b_multiple_search_bullets_are_reported_at_entry(self):
        bad=SKILL.replace("## Results State","- When evidence changes, update the attributes.\n## Results State")
        with self.assertRaisesRegex(ValueError,"Search State must contain exactly one"): self.check(s=bad)
    def test_action_vocabulary_language_is_preserved(self):
        with self.assertRaisesRegex(ValueError,"action-shaped vocabulary"):
            self.check(s=SKILL.replace("verify the attributes","search for the attributes"))
    def test_no_silent_rewriting(self):
        bad=AGENT+"- Coordination: Additional guidance.\n"; original=bad
        with self.assertRaises(ValueError): self.check(a=bad)
        self.assertEqual(bad,original)
    def test_tokenizer_caps_when_available(self):
        with self.assertRaisesRegex(ValueError,"token count"):
            self.check(token_counter=lambda text:1000)
    def test_absent_tokenizer_is_not_fabricated_measurement(self):
        self.assertIsNone(self.check()["checks"]["text_memory"]["tokens"])
    def test_empty_memory_rejected(self):
        with self.assertRaises(ValueError): self.check(m="")
    def test_runtime_preflight_checks_before_model_construction(self):
        s=(ROOT/"scripts/openevo_webshop/verify_ceiling1_stage2_initial_state.py").read_text()
        self.assertLess(s.index("  require_evolution_compatible_bundle("),s.index("  rt=runtime_cls("))
    def test_new_state_checks_before_persistence(self):
        s=(ROOT/"scripts/openevo_webshop/ceiling1_stage2_arm_controller.py").read_text()
        self.assertLess(s.index("context='new Stage2 state'"),s.index("sid=state_id(base,None"))
    def test_producer_pass_depends_on_evolution_contract(self):
        tree=ast.parse((ROOT/"scripts/openevo_webshop/ceiling1_harness201_generic_safety_finalize.py").read_text())
        statuses=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="status" for t in n.targets)]
        self.assertTrue(any("evolution_contract" in ast.unparse(n.value) for n in statuses))
if __name__=="__main__": unittest.main()
