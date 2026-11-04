"""Synthetic filesystem guards. No real data, models or replay subprocesses."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
from reviewer_replay import main, reject_existing_output, verify_bindings


class ReviewerReplayGuards(unittest.TestCase):
    def make_fixture(self, root):
        repo = root / 'repo'; repo.mkdir()
        payload = repo / 'synthetic_scalar.json'
        payload.write_text('{"invented_loss": 2.0}\n', encoding='utf-8')
        records = {'bindings': [{'path': payload.name, 'bytes': payload.stat().st_size,
                                'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}]}
        bindings = root / 'bindings.json'
        bindings.write_text(json.dumps(records), encoding='utf-8')
        return repo, payload, bindings, hashlib.sha256(bindings.read_bytes()).hexdigest()

    def test_exact_synthetic_file_accepted_then_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, payload, bindings, digest = self.make_fixture(Path(tmp))
            self.assertEqual(len(verify_bindings(repo, bindings, digest)['bindings']), 1)
            # Same byte count, changed value: size check alone is insufficient.
            payload.write_text('{"invented_loss": 9.0}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_bindings(repo, bindings, digest)

    def test_missing_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, payload, bindings, digest = self.make_fixture(Path(tmp))
            payload.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                verify_bindings(repo, bindings, digest)

    def test_existing_output_refused_without_touching_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing'; output.mkdir()
            marker = output / 'preserve.txt'; marker.write_text('original')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                reject_existing_output(output)
            self.assertEqual(marker.read_text(), 'original')

    def test_binding_tamper_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, _, bindings, digest = self.make_fixture(Path(tmp))
            bindings.write_text('{"bindings": []}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'document changed'):
                verify_bindings(repo, bindings, digest)

    def test_child_failure_retains_failure_not_success_receipt(self):
        # Explicit mock of a failed scalar subprocess; no actual child/model
        # is launched and no genuine study result is asserted by this fixture.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); repo = root/'repo'; repo.mkdir()
            output = root/'new_output'
            command = ['reviewer_replay.py','--repo-root',str(repo),'--output-dir',str(output)]
            fake = subprocess.CompletedProcess(args=['synthetic_child'], returncode=7,
                                               stdout='', stderr='synthetic expected failure')
            with patch('sys.argv', command), patch('reviewer_replay.verify_bindings', return_value={'bindings': []}), \
                    patch('reviewer_replay.subprocess.run', return_value=fake):
                self.assertEqual(main(), 2)
            self.assertFalse((output/'REVIEWER_REPLAY_RECEIPT.json').exists())
            failure = json.loads((output/'REVIEWER_REPLAY_FAILURE.json').read_text())
            self.assertEqual(failure['status'], 'FAILED_NO_SUCCESS_RECEIPT')
            child = json.loads((output/'DEVELOPMENT_REPLAY_TRANSCRIPT.json').read_text())
            self.assertEqual(child['exit_code'], 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
