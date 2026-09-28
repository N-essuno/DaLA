import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts.handoff_icelandic_workers import capture, eligible, group_members, identity, stop, write


class HandoffTests(unittest.TestCase):
    def test_success_gate(self):
        self.assertFalse(eligible({'exit_code': 1, 'status': 'failed_checkpoints_preserved'}))
        self.assertFalse(eligible({'exit_code': 0}))
        self.assertTrue(eligible({'exit_code': 0, 'status': 'candidate_build_complete_requires_final_audit'}))

    def test_pid_reuse_refuses_signal(self):
        with patch('scripts.handoff_icelandic_workers.os.killpg') as kill:
            with self.assertRaises(AssertionError):
                stop({'pid': os.getpid(), 'starttime': 'wrong'})
            kill.assert_not_called()

    def test_owned_group_stop(self):
        command = ['-c', 'import time; time.sleep(60)']
        process = subprocess.Popen([sys.executable, *command], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'launch.json'
                write(path, {'pid': process.pid, 'command': command})
                receipt = capture(path)
                self.assertEqual(receipt['starttime'], identity(process.pid))
                stop(receipt)
                process.wait(timeout=5)
                self.assertEqual(group_members(process.pid), [])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
