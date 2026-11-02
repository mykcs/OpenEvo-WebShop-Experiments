"""CPU-only safety unit checks for Q17 native SD-LoRA adapter exact exporter."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"scripts/analysis/q17_native_state_to_peft_20261013.py"
s=importlib.util.spec_from_file_location("q17_native_adapter",SOURCE)
assert s and s.loader
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_allowlisted_exact_rounds_only(self):
        self.assertEqual(set(m.EXPECTED_ADAPTER),{126,127,150})
        self.assertEqual(set(m.EXPECTED_COMPONENTS),{126,127,150})
        for bad in (-1,0,125,128,149,151,159):
            with self.assertRaises(m.IntegrityError):
                m.original(bad)

    def test_production_round_reconstruction_does_not_target_original_run_root(self):
        self.assertFalse(m.ALLOWED_ROOT.resolve().is_relative_to(m.Q17.resolve()))
        self.assertIn("Q17-SD-LORA-RECONSTRUCT-20261013",str(m.ALLOWED_ROOT))

    def test_digest_function(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"t.bin";p.write_bytes(b"abc")
            self.assertEqual(m.digest(p),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_coefficients_applied_to_B_not_A(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch only installed in the pinned science Docker")
        class Native:
            data={
             "components.0.a.A":torch.tensor([[1.,2.]]),
             "components.1.a.A":torch.tensor([[3.,4.]]),
             "components.0.a.B":torch.tensor([[5.],[6.]]),
             "components.1.a.B":torch.tensor([[7.],[8.]])
            }
            def get_tensor(self,key):return self.data[key]
        a,b=m.factor_pair(Native(),"a",[2.,.5],2)
        self.assertTrue(torch.equal(a,torch.tensor([[1.,2.],[3.,4.]])))
        self.assertTrue(torch.equal(b,torch.tensor([[10.,3.5],[12.,4.]])))

if __name__=="__main__":
    unittest.main()
