"""Shared non-scientific runtime handoff proof absorbed from Draft PR #524.

This module proves a UID1001 producer -> host -> UID1001 consumer bind-mount
boundary. It does not launch scientific work, allocate GPUs, or grant execution
authority.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _handoff_file_identity(path: Path) -> dict:
    st = path.stat()
    return {
        "uid": st.st_uid,
        "gid": st.st_gid,
        "mode": stat.S_IMODE(st.st_mode),
        "is_regular_file": stat.S_ISREG(st.st_mode),
        "sha256": sha256(path),
    }


def production_handoff_smoke(handoff_root: Path, image: str) -> dict:
    """Prove UID1001 producer -> host -> UID1001 consumer bind-mount semantics.

    The caller supplies a non-scientific maintenance scratch root on the same
    filesystem/mount route used by formal runtime handoffs.  The smoke writes no
    experiment task and uses no GPU/network.
    """
    root = handoff_root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"formal handoff smoke root must be an existing regular directory: {root}")
    token = f"formal-launch-handoff-{os.getpid()}-{time.time_ns()}"
    scratch = root / token
    scratch.mkdir(mode=0o777)
    os.chmod(scratch, 0o777)
    payload = b"openevo-formal-handoff-smoke\n"
    payload_sha = hashlib.sha256(payload).hexdigest()
    writer_code = (
        "from pathlib import Path; "
        "p=Path('/handoff/payload.bin'); "
        f"p.write_bytes(bytes.fromhex('{payload.hex()}')); "
        "p.chmod(0o600); print('FORMAL_HANDOFF_WRITE_PASS')"
    )
    reader_code = (
        "from pathlib import Path; import hashlib; "
        "b=Path('/handoff/payload.bin').read_bytes(); "
        f"assert hashlib.sha256(b).hexdigest()=='{payload_sha}'; "
        "print('FORMAL_HANDOFF_READ_PASS')"
    )
    writer = [
        "docker", "run", "--rm", "--network", "none", "--user", "1001:1001",
        "-v", f"{scratch}:/handoff:rw", "--entrypoint", "/usr/local/bin/python3",
        image, "-c", writer_code,
    ]
    reader = [
        "docker", "run", "--rm", "--network", "none", "--user", "1001:1001",
        "-v", f"{scratch}:/handoff:ro", "--entrypoint", "/usr/local/bin/python3",
        image, "-c", reader_code,
    ]
    try:
        q = subprocess.run(writer, text=True, capture_output=True)
        if q.returncode != 0 or "FORMAL_HANDOFF_WRITE_PASS" not in q.stdout:
            raise RuntimeError(f"formal UID1001 handoff producer smoke failed: {(q.stderr or q.stdout).strip()}")
        payload_path = scratch / "payload.bin"
        if not payload_path.is_file():
            raise RuntimeError("formal UID1001 handoff producer did not materialize payload")
        identity = _handoff_file_identity(payload_path)
        if identity["uid"] != 1001 or identity["gid"] != 1001:
            raise RuntimeError(
                f"formal handoff ownership mismatch expected=1001:1001 observed={identity['uid']}:{identity['gid']}"
            )
        if identity["mode"] != 0o600 or identity["is_regular_file"] is not True:
            raise RuntimeError(
                f"formal handoff mode/type mismatch expected=regular/0600 observed={identity}"
            )
        if identity["sha256"] != payload_sha:
            raise RuntimeError("formal handoff host readback payload hash mismatch")
        q = subprocess.run(reader, text=True, capture_output=True)
        if q.returncode != 0 or "FORMAL_HANDOFF_READ_PASS" not in q.stdout:
            raise RuntimeError(f"formal UID1001 handoff consumer smoke failed: {(q.stderr or q.stdout).strip()}")
        return {
            "status": "PASS",
            "host_root": str(root),
            "host_device": root.stat().st_dev,
            "producer_user": "1001:1001",
            "consumer_user": "1001:1001",
            "producer_mount": "rw",
            "consumer_mount": "ro",
            "network": "none",
            "payload_sha256": payload_sha,
            "host_readback": identity,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
