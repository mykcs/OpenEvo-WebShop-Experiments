#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'docs/evidence/launcher-owner-migration-20260930/INDEX.json'
MARKER = b'  # legacy-revision-compat'
IMAGE_DECLARATION = b'IMAGE="experiment-host/openevo-scientific:h144-runtime-v1"\n'  # legacy-revision-compat
STAGE2_DISABLED = b'echo "NO-GO Stage2 successor execution authority is not committed; historical body is evidence only" >&2; exit 2\n'
E1_DISABLED = b'echo "NO-GO E1 is sealed; historical transition body is evidence only" >&2; exit 2\n'
E3_PARENT_GUARD = b'PYTHONPATH="$WT" python3 -m scripts.openevo_webshop.post_advisor_e3_guard_202609300248 --repo "$WT" --authority "$AGGREGATE" --formal >/dev/null\n'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    payload = json.loads(INDEX.read_text())
    assert payload['evidence_only'] is True and payload['execution_authority'] is False
    assert len(payload['records']) == 6
    for row in payload['records']:
        original = subprocess.check_output(
            ['git', 'show', row['original_git_commit'] + ':' + row['original_path']],
            cwd=ROOT,
        )
        assert digest(original) == row['original_sha256']
        archived = (ROOT / row['evidence_only_path']).read_bytes()
        assert archived == original
        current = (ROOT / row['active_driver_path']).read_bytes()
        assert digest(current) == row['active_driver_sha256']
        assert current.count(MARKER) == len(row['compatibility_comment_lines'])
        normalized = current.replace(MARKER, b'')
        if Path(row['active_driver_path']).name in {
            'bounded_carrier_container_driver_20260916.sh',
            'ceiling1_stage2_reflector_container_driver.sh',
            'ceiling1_stage2_reflector_qwen3_container_driver.sh',
            'ceiling1_stage2_round_qwen3_container_driver.sh',
        }:
            assert normalized.count(STAGE2_DISABLED) == 1
            normalized = normalized.replace(STAGE2_DISABLED, b'', 1)
        if row['active_driver_path'].endswith('post_advisor_magnitude_transition_driver_20260930.sh'):
            assert normalized.count(E1_DISABLED) == 1
            normalized = normalized.replace(E1_DISABLED, b'', 1)
        if row['active_driver_path'].endswith('post_advisor_e3_rank_capacity_transition_driver_202609300248.sh'):
            assert normalized.count(E3_PARENT_GUARD) == 1
            normalized = normalized.replace(E3_PARENT_GUARD, b'', 1)
        if b'post_advisor_e3_rank_capacity_transition_driver_' in row['active_driver_path'].encode():
            assert IMAGE_DECLARATION in normalized
            normalized = normalized.replace(IMAGE_DECLARATION, b'', 1)
            normalized = normalized.replace(b'"$IMAGE"', b'experiment-host/openevo-scientific:h144-runtime-v1', 1)  # legacy-revision-compat
        assert normalized == original, row['active_driver_path']
    print('launcher historical body and active driver equivalence: PASS')


if __name__ == '__main__':
    main()
