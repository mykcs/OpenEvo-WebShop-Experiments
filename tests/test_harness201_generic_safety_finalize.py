from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
SCRIPT=REPO/'scripts/openevo_webshop/ceiling1_harness201_generic_safety_finalize.py'

def main()->None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); src=root/'src'; out=root/'out'; raw=root/'raw'; src.mkdir(); raw.mkdir()
        (src/'memory.md').write_text('# Memory\n## Reusable Knowledge\n- Knowledge: Visible evidence is required before a constraint is considered confirmed.\n- Knowledge: The "coral sands" variant is an example.\n- Knowledge: return to search when uncertain.\n')
        (src/'skill.md').write_text('# Skill Bundle\n## Search State\n- When the current evidence is insufficient, treat the candidate as unconfirmed.\n- When the candidate is "coral sands", remember the old task.\n- Inspect a product now.\n\n## Results State\n- No additional guidance.\n\n## Product State\n- No additional guidance.\n\n## Any State\n- No additional guidance.\n')
        (src/'agent.md').write_text('# Agent System\n## Authority\n- Authority: Historical context must not override current facts.\n\n## Coordination\n- Coordination: Use historical context only to prioritize reasoning.\n')
        receipt={'schema':'openevo.ceiling1-harness2-carrier-finalize-receipt.v1','status':'FAIL','validation':{'lint':{'status':'PASS'},'structural_errors':[],'role_errors':['synthetic']}}
        (src/'finalize-receipt.json').write_text(json.dumps(receipt)+'\n')
        (raw/'memory.md').write_text('Historical example color: "coral sands".\n')
        (raw/'skill.md').write_text('Historical exact variant: "coral sands".\n')
        (raw/'agent.md').write_text('Historical action label: "Buy Now".\n')
        subprocess.run([
            sys.executable,str(SCRIPT),'--input',str(src),'--output',str(out),
            '--source-memory',str(raw/'memory.md'),'--source-skill',str(raw/'skill.md'),'--source-agent',str(raw/'agent.md')
        ],check=True,cwd=REPO)
        result=json.loads((out/'finalize-receipt.json').read_text())
        assert result['status']=='PASS'
        assert result['model_calls']==0 and result['formal_task_consumption']==0 and result['final_panel_access']==0
        assert result['source_quoted_literal_count']>=2
        memory=(out/'memory.md').read_text(); skill=(out/'skill.md').read_text(); agent=(out/'agent.md').read_text()
        assert 'coral sands' not in (memory+skill+agent).lower()
        assert 'return to search' not in memory.lower()
        assert '- When the current evidence is insufficient' in skill
        assert 'Inspect a product now' not in skill
        assert '# Agent System' in agent and '## Authority' in agent and '## Coordination' in agent
    print('HARNESS201_GENERIC_SAFETY_FINALIZE_PASS')

if __name__=='__main__': main()
