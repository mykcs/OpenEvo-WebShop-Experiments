"""CPU-only regression for exact-UUID project flock wait: no CUDA or paid work."""
import fcntl
import tempfile
import threading
import time
import unittest
from pathlib import Path

from scripts.oe_impl import LaunchError, gpu_locks


class ExactProjectGpuLockWait(unittest.TestCase):
    def test_default_busy_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock_path = root / "GPU-A.lock"
            with lock_path.open("a+") as blocker:
                fcntl.flock(blocker.fileno(), fcntl.LOCK_EX)
                with self.assertRaisesRegex(LaunchError, "project GPU lock busy"):
                    with gpu_locks(["GPU-A"], root):
                        self.fail("must not enter without exclusive lock")

    def test_wait_for_other_owner_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock_path = root / "GPU-A.lock"
            with lock_path.open("a+") as blocker:
                fcntl.flock(blocker.fileno(), fcntl.LOCK_EX)
                released = threading.Event()

                def release():
                    fcntl.flock(blocker.fileno(), fcntl.LOCK_UN)
                    released.set()

                timer = threading.Timer(0.15, release)
                timer.start()
                try:
                    started = time.monotonic()
                    with gpu_locks(["GPU-A"], root, wait_seconds=0.8):
                        self.assertTrue(released.is_set())
                    self.assertGreaterEqual(time.monotonic() - started, 0.10)
                finally:
                    timer.join(timeout=1)
                    fcntl.flock(blocker.fileno(), fcntl.LOCK_UN)

    def test_timeout_keeps_flock_exclusion_and_releases_partial_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blocker_path = root / "GPU-B.lock"
            with blocker_path.open("a+") as blocker:
                fcntl.flock(blocker.fileno(), fcntl.LOCK_EX)
                with self.assertRaisesRegex(LaunchError, "bounded wait exhausted"):
                    with gpu_locks(["GPU-A", "GPU-B"], root, wait_seconds=0.05):
                        self.fail("must not enter without both exclusive locks")
                # The first acquired lock must have been released on refusal.
                with (root / "GPU-A.lock").open("a+") as after:
                    fcntl.flock(after.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(after.fileno(), fcntl.LOCK_UN)

    def test_unbounded_lock_wait_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(LaunchError, "0..60"):
                with gpu_locks(["GPU-A"], Path(temp), wait_seconds=3600):
                    self.fail("invalid wait never enters")


if __name__ == "__main__":
    unittest.main()
