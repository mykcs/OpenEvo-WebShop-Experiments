from __future__ import annotations
import subprocess,tempfile,unittest
from pathlib import Path
from scripts.ci import check_master_plan_authority

class MasterPlanAuthorityTests(unittest.TestCase):
    def _repo(self, parallel: bool=False):
        td=tempfile.TemporaryDirectory(); root=Path(td.name)
        canonical=root/check_master_plan_authority.CANONICAL
        canonical.parent.mkdir(parents=True)
        canonical.write_text('# Master\nENGINEERING_PLATFORM_CANONICAL_AUTHORITY\n')
        current=root/check_master_plan_authority.CURRENT
        current.write_text('not a second Master Plan\nOPEN_EVO_CEILING_1.0_END_TO_END_MASTER_PLAN.md\n')
        if parallel:
            p=root/'docs/other.md'; p.write_text('Status: **ACTIVE EXECUTION MASTERPLAN**\n')
        subprocess.run(['git','init','-q',root],check=True)
        subprocess.run(['git','-C',root,'add','.'],check=True)
        return td,root

    def test_minimal_single_authority_passes(self):
        td,root=self._repo()
        try:self.assertEqual(check_master_plan_authority.check(root),[])
        finally:td.cleanup()

    def test_parallel_active_masterplan_is_rejected(self):
        td,root=self._repo(parallel=True)
        try:
            errors=check_master_plan_authority.check(root)
            self.assertTrue(any('parallel active authority' in e for e in errors),errors)
        finally:td.cleanup()

    def test_missing_engineering_sentinel_is_rejected(self):
        td,root=self._repo()
        try:
            (root/check_master_plan_authority.CANONICAL).write_text('# Master\n')
            errors=check_master_plan_authority.check(root)
            self.assertTrue(any('missing sentinel' in e for e in errors),errors)
        finally:td.cleanup()

if __name__=='__main__':unittest.main()
