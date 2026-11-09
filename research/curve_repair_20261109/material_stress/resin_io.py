"""Restricted original-file reader for the preregistered resin pilot.

Presence/import grants no execution authorization. The CLI validates root's
stage authorization before constructing this reader. No optimizer is provided.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import zipfile

SOURCE_SHA256 = 'a2a7a84d92bf84a69e6f6c49e0ae976d916cc63fa46ff28bccd08a7ebdd4e218'
INPUT_SHA256 = 'dd1e7b41b8db7076a6b55a68f94d41cc0b2e2e323ca1fd9596c57b9c7b4a2900'
ROLE_PREFIX = 'mlst-resin-role-v1|'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key: ' + key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique, parse_constant=invalid)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_inputs(path):
    if sha(path) != INPUT_SHA256:
        raise ValueError('Registered input-mask bytes changed')
    data = read_json(path)
    if data['source_zip_sha256'] != SOURCE_SHA256 or len(data['per_run']) != 50:
        raise ValueError('Unexpected source or input population')
    return data


def assign_recipe_roles(inputs):
    """Input-only 3 dev / 3 expansion / 4 final recipe allocation."""
    recipe_runs = {}
    for run in inputs['per_run']:
        recipe_runs.setdefault(run['recipe'], []).append(run)
    strata = {'ascending_35_first': [], 'descending_35_last': []}
    for recipe, runs in recipe_runs.items():
        histories = {run['thermal_history'] for run in runs}
        if len(runs) != 5 or len(histories) != 1:
            raise ValueError('Five same-history runs required per recipe')
        history = histories.pop()
        if history not in strata:
            raise ValueError('Unknown thermal-order stratum')
        strata[history].append(recipe)
    if [len(strata[k]) for k in strata] != [7, 3]:
        raise ValueError('Expected seven ascending and three descending recipes')
    result = {}
    for history, recipes in strata.items():
        ordered = sorted(recipes, key=lambda name:
                         (hashlib.sha256((ROLE_PREFIX + name).encode('utf-8')).hexdigest(), name))
        labels = (['dev'] * 2 + ['future_expansion'] * 2 + ['final_confirm'] * 3
                  if history == 'ascending_35_first'
                  else ['dev', 'future_expansion', 'final_confirm'])
        for recipe, role in zip(ordered, labels):
            result[recipe] = {'role': role, 'thermal_history': history,
                              'role_hash': hashlib.sha256((ROLE_PREFIX + recipe).encode('utf-8')).hexdigest()}
    if [sum(v['role'] == role for v in result.values())
            for role in ['dev', 'future_expansion', 'final_confirm']] != [3, 3, 4]:
        raise ValueError('Recipe role capacity mismatch')
    return result


class RestrictedResinReader:
    """Only selected dev anchor cells, or released common dev target cells.

    Numeric eta tokens are selected by the original immutable line address.
    Whole ZIP/UTF16 text bytes necessarily enter memory; other eta tokens are
    never converted, returned, inspected for validity, or used for QC. An
    invalid/nonpositive ACCESSED value stops; no replacement/deletion exists.
    The reader is a narrow API boundary, not an adversarial OS sandbox.
    """
    def __init__(self, source_zip, inputs, *, stage, target_release=None):
        if stage not in ('anchors', 'targets'):
            raise ValueError('Explicit anchor or released-target stage required')
        if stage == 'targets' and not target_release:
            raise PermissionError('Independent prediction release required')
        if sha(source_zip) != SOURCE_SHA256:
            raise ValueError('Original ZIP SHA256 changed')
        self.archive = zipfile.ZipFile(source_zip)
        self.roles = assign_recipe_roles(inputs)
        self.specs = {run['member']: run for run in inputs['per_run']}
        self.stage, self.ledger = stage, []

    def close(self):
        self.archive.close()

    def read_log_eta(self, member, indices):
        """Return only one method's six anchors, or common eight targets."""
        spec = self.specs[member]
        if self.roles[spec['recipe']]['role'] != 'dev':
            raise PermissionError('Future/final recipe eta remains sealed')
        indices = list(indices)
        if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
            raise ValueError('Unique integer native-grid indices required')
        if self.stage == 'anchors':
            if len(indices) != 6 or not {0, 1}.issubset(indices) or not set(indices).issubset(range(17)):
                raise PermissionError('Exactly six own anchors required; target panel forbidden')
        elif indices != list(range(17, 25)):
            raise PermissionError('Only the released common eight targets may be read')
        original = self.archive.read(member)
        if hashlib.sha256(original).hexdigest() != spec['source_member_sha256']:
            raise ValueError('Original run bytes changed')
        lines = original.decode('utf-16').splitlines()
        answer = []
        for index in indices:
            at = spec['source_line_by_rank_1based'][index]
            parts = lines[at - 1].split('\t')
            if len(parts) != 7 or parts[5].strip() != spec['native_gamma_raw_tokens_sorted'][index]:
                raise ValueError('Registered native input address mismatch')
            if not 34.5 <= float(parts[2]) < 35.5 or parts[4] != 'Dy_auto':
                raise ValueError('Registered temperature/status inputs changed')
            # The only selection/conversion of a response token in this source.
            site = {'member': member, 'rank_1based': index + 1, 'line_1based': at,
                    'stage': self.stage, 'valid': False}
            self.ledger.append(site)
            try:
                cp = float(parts[3])
                if not math.isfinite(cp) or cp <= 0:
                    raise ValueError('invalid')
                value = math.log(cp / 1000.0)
                if not math.isfinite(value):
                    raise ValueError('invalid log conversion')
            except (ValueError, OverflowError):
                raise ValueError('STOP_NO_REPLACEMENT invalid/nonpositive accessed eta at '
                                 + member + ' rank ' + str(index + 1)) from None
            site['valid'] = True
            answer.append(value)
        return answer

    def receipt(self):
        unique = {(r['member'], r['rank_1based']) for r in self.ledger}
        return {'stage': self.stage, 'numerical_eta_access_calls': len(self.ledger),
                'unique_eta_cells_accessed': len(unique), 'access_sites': self.ledger,
                'future_or_final_eta_accessed': False,
                'target_eta_accessed': self.stage == 'targets' and bool(self.ledger),
                'replacements': 0, 'QC_deletions': 0}
