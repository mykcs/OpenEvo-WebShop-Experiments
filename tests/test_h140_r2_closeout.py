import unittest
from pathlib import Path

from scripts.openevo_webshop.validate_h140_r2_preregistration import validate as validate_preregistration
from scripts.verify_webshop_program_ledger import verify as verify_ledger


ROOT = Path(__file__).parents[1]


class H140R2CloseoutTests(unittest.TestCase):
    def test_terminal_upstream_gate_failure_is_a_closed_router_state(self):
        preregistration = validate_preregistration(ROOT)
        self.assertFalse(preregistration["formal_task_consumption_allowed"])
        self.assertTrue(preregistration["g2_adapter_created"])
        self.assertFalse(preregistration["t2_panel_evaluated"])

        ledger = verify_ledger(ROOT)
        self.assertEqual(ledger["status"], "pass")
        self.assertFalse(ledger["h1_38b_launch_authorized"])


if __name__ == "__main__":
    unittest.main()
