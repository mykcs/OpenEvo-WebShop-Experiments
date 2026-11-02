#!/usr/bin/env python3
"""Write-once zero-task replay materialization for E3 rank16/64."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/workspace/control/post-advisor-e3-rank-capacity-20260930')
AUTHORITY = ROOT / 'configs/experiment/activations/post-advisor-e3-rank-capacity-authority-202609300248.json'
EXPECTED_REPLAY_SHA = '96229160341bb5527f1991ad7d510600e80cc03b041a2d48e4006ce7d47f9abb'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=(16, 64), required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError('E3 staging requires root for exact UID1001 ownership')
    authority = json.loads(AUTHORITY.read_text())
    if authority.get('status') != 'PREPARED_NOT_ACTIVE':
        raise RuntimeError('E3 staging is limited to preactivation')
    cell = authority['arms'][str(args.rank)]
    source = Path(cell['initial_adapter_path']) / 'openevo_sd_lora_replay.jsonl'
    output = Path(cell['output_root'])
    preflight = output / 'preflight'
    target = preflight / 'prior-replay.jsonl'
    receipt = preflight / 'prior-replay-materialization.json'
    if output.exists() or target.exists() or receipt.exists():
        raise RuntimeError('E3 staging target exists; preserve it and reconcile before retry')
    source_stat = source.stat()
    if (not source.is_file() or sha(source) != EXPECTED_REPLAY_SHA
            or cell['openevo_sd_lora_replay.jsonl_sha256'] != EXPECTED_REPLAY_SHA):
        raise RuntimeError('E3 source replay identity drift')
    if source_stat.st_uid != 0 or source_stat.st_nlink != 1:
        raise RuntimeError('E3 rank16/64 source owner/link identity drift')
    output.mkdir(mode=0o750)
    os.chown(output, 1001, 1001)
    preflight.mkdir(mode=0o750)
    os.chown(preflight, 1001, 1001)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    try:
        os.fchown(fd, 1001, 1001)
        with source.open('rb') as original:
            for chunk in iter(lambda: original.read(1 << 20), b''):
                view = memoryview(chunk)
                while view:
                    view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    target_stat = target.stat()
    if (sha(source) != EXPECTED_REPLAY_SHA or sha(target) != EXPECTED_REPLAY_SHA
            or target_stat.st_uid != 1001 or target_stat.st_nlink != 1
            or source.stat().st_ino != source_stat.st_ino
            or source.stat().st_nlink != source_stat.st_nlink):
        raise RuntimeError('E3 write-once replay staging verification failed')
    payload = {
        'schema': 'openevo.post-advisor.e3-prior-replay-materialization.202609300248',
        'status': 'PASS_ZERO_TASK', 'rank': args.rank,
        'source_path': str(source), 'source_sha256': EXPECTED_REPLAY_SHA,
        'source_uid': source_stat.st_uid, 'source_nlink': source_stat.st_nlink,
        'source_inode_preserved': True, 'historical_source_mutated': False,
        'target_path': str(target), 'target_sha256': EXPECTED_REPLAY_SHA,
        'target_uid': target_stat.st_uid, 'target_nlink': target_stat.st_nlink,
        'formal_rollouts_consumed': 0, 'final_panel_access': 0,
    }
    fd = os.open(receipt, flags, 0o600)
    try:
        os.fchown(fd, 1001, 1001)
        os.write(fd, (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({'status': 'PASS_ZERO_TASK', 'rank': args.rank,
                      'receipt_path': str(receipt), 'receipt_sha256': sha(receipt)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
