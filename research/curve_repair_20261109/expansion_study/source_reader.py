"""Explicit future-expansion response reader; never reassigns an old role.

Source-only presence does not authorize access. The experiment CLI validates
root actual-UTC registration and phase/release hashes before instantiation.
This is trusted software isolation, not encryption: full selected UTF16 member
bytes enter memory, but only exact allowed cells are converted to numeric Y.
"""
from __future__ import annotations
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parent.parent
IO_SHA = '00510a21819d2cce4a67297ddad40894c7f6cc3659b030d186b2ef0343d4a3b8'
ANCHORS = [0, 1, 13, 14, 15, 16]
PANEL = list(range(17, 25))


def pure_io():
    path = ROOT / 'material_stress' / 'resin_io.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != IO_SHA:
        raise ValueError('Original pure IO/role source bytes changed')
    name = '_mlst_expansion_original_io_' + IO_SHA
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


class ExpansionReader:
    """Read own six anchors ONCE or separately released eight target cells.

The immutable original 3dev/3future_expansion/4final_confirm recipe allocation
is recomputed from input-only metadata. Calling code must explicitly request
authorized_role='future_expansion'; no dev alias or altered roles are accepted.
"""
    def __init__(self, source_zip, inputs, *, authorized_role, stage, target_release=None):
        if authorized_role != 'future_expansion':
            raise PermissionError('Only explicit future_expansion role is authorized')
        if stage not in ('anchors', 'targets') or (stage == 'targets' and not target_release):
            raise PermissionError('Anchor stage or separately released target stage required')
        io = pure_io()
        if io.sha(source_zip) != io.SOURCE_SHA256:
            raise ValueError('Original ZIP bytes changed')
        self.roles = io.assign_recipe_roles(inputs)
        self.specs = {row['member']: row for row in inputs['per_run']}
        allowed = [name for name, row in self.roles.items() if row['role'] == 'future_expansion']
        if len(allowed) != 3:
            raise ValueError('Exactly three original future-expansion recipes required')
        self.archive = zipfile.ZipFile(source_zip)
        self.authorized_role, self.stage, self.ledger = authorized_role, stage, []
        self.read_members = set()

    def close(self):
        self.archive.close()

    def read_log_eta(self, member, indices):
        spec = self.specs[member]
        if self.roles[spec['recipe']]['role'] != self.authorized_role:
            raise PermissionError('Dev and final_confirm response values are forbidden')
        expected = ANCHORS if self.stage == 'anchors' else PANEL
        if list(indices) != expected or any(type(i) is not int for i in indices):
            raise PermissionError('Only the exact fixed six anchors or eight targets may be read')
        if member in self.read_members:
            raise PermissionError('Each expansion curve is read once per reader stage; no repeated anchor calls')
        self.read_members.add(member)
        original = self.archive.read(member)
        if hashlib.sha256(original).hexdigest() != spec['source_member_sha256']:
            raise ValueError('Original run bytes changed')
        lines = original.decode('utf-16').splitlines()
        answer = []
        for index in expected:
            at = spec['source_line_by_rank_1based'][index]
            parts = lines[at - 1].split('\t')
            if (len(parts) != 7 or parts[5].strip() != spec['native_gamma_raw_tokens_sorted'][index] or
                    not 34.5 <= float(parts[2]) < 35.5 or parts[4] != 'Dy_auto'):
                raise ValueError('Original registered coordinate/temperature/status address changed')
            site = {'member': member, 'recipe': spec['recipe'], 'role': self.authorized_role,
                    'rank_1based': index + 1, 'line_1based': at, 'stage': self.stage, 'valid': False}
            self.ledger.append(site)
            try:
                cp = float(parts[3])
                if not math.isfinite(cp) or cp <= 0:
                    raise ValueError('invalid')
                value = math.log(cp / 1000.)
                if not math.isfinite(value):
                    raise ValueError('invalid')
            except (ValueError, OverflowError):
                raise ValueError('STOP_NO_REPLACEMENT invalid/nonpositive accessed expansion eta at '
                                 + member + ' rank ' + str(index + 1)) from None
            site['valid'] = True
            answer.append(value)
        return answer

    def receipt(self):
        return {'stage': self.stage, 'authorized_role': self.authorized_role,
                'numerical_eta_access_calls': len(self.ledger),
                'unique_eta_cells_accessed': len({(r['member'], r['rank_1based']) for r in self.ledger}),
                'curves_accessed': len(self.read_members), 'access_sites': self.ledger,
                'dev_or_final_response_accessed': False,
                'final_confirm_runs_protected': 20, 'replacements': 0, 'QC_deletions': 0}
