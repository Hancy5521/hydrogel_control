"""Explicit local inputs, information roles, immutable hashes and output I/O.

Only this adapter reads the closed TRAIN cache.  The original source tree is
never mutated; public code can use an explicitly supplied lawful asset root.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np

CODE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE / 'frozen'))
from coordinate_reference import CoordinateGP, Normalization

MASKS = {
    'spread': [0, 7, 15, 23, 31, 38],
    'middle_gap': [0, 4, 8, 30, 34, 38],
    'one_sided': [0, 3, 6, 9, 12, 15],
}
P_ROWS = (0, 1, 3, 4, 5, 6, 11, 15, 16, 20, 21, 24, 26, 27,
          28, 29, 30, 31, 33, 35, 36, 37, 40, 41, 44)
S_ROWS = tuple(i for i in range(45) if i not in P_ROWS)
OLD_SEED = 20261030
SEEDS = tuple(range(20261105, 20261113))
METHODS = tuple(f'random_{seed}' for seed in SEEDS) + ('prior_kl', 'curvature45')
INPUT_PINS = {
    'data/TRAIN.npz': 'cbd09c37e80548937a3b012e86e4173b3dc663cbe8e61f5ca7baa18c9002d974',
    'inputs/FROZEN_CANDIDATE_MODEL.json': 'd72fc262fbae0e6fecc36568f0638166fe95e991bd5f7df100ab1434097d82d1',
    'inputs/FROZEN_CANDIDATE_SCORES.json': '8800800920bad262fc06bdfc18ccdf319f080387eacbaf5808d1f28851ed0b43',
    'inputs/baseline/RESULT.json': '4ecf3b49cbf3499b983a0086170473f3f3ebbd81a4bfe6f4f5cd641bcb677a2f',
    'results/RANDOM_GAUSSIAN_RESULT.json': '5840b80b8c665776e18141c3c473d0bbb203e4b38cf3025877dde87b02f2cf18',
}
VENDOR_PINS = {
    'joint_code_gaussian.py': '0c8a57c1bdf96e1c880ffd1ba2258e1b379952b844e960a7c67d6cdcdc65e299',
    'coordinate_reference.py': 'bc93acc707cb26943cc065545254fc9fed27601e8d0f04c4a23dbb7407cf43d3',
    'local_shape.py': '652797e34e7299edd31a8b494287ad8ab36aca9cdcdcc7b455dc614201231d3b',
    'conditional_metrics.py': '3332508bfe887a13550333761a49e8d130a5b7611f166d9d1040b930806bc461',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_bindings():
    files = [CODE / name for name in
             ('assets.py', 'representations.py', 'experiment.py')]
    files.extend(CODE / 'frozen' / name for name in VENDOR_PINS)
    for name, digest in VENDOR_PINS.items():
        if sha(CODE / 'frozen' / name) != digest:
            raise PermissionError('Frozen core identity changed: ' + name)
    return [{'file': str(p.relative_to(CODE).as_posix()), 'sha256': sha(p)}
            for p in files]


def validate_reference(info, held):
    expected = [j for j in P_ROWS if j != held] + list(S_ROWS)
    a = info['admission']
    if (info['held_P'] != held or not a['admitted']
            or not a['held_row_entirely_excluded']
            or a['training_rows'] != expected or a['visible_cells'] != 1056):
        raise PermissionError('Held formulation reference roles changed')


def prepare_inputs(asset_root, run_root):
    """Byte/hash copies of an explicit allowlist; never extracts historical S."""
    asset_root, run_root = Path(asset_root).resolve(), Path(run_root).resolve()
    destination = run_root / 'experiment' / 'inputs'
    destination.mkdir(parents=True, exist_ok=False)
    bindings = []
    for relative, digest in INPUT_PINS.items():
        source = asset_root / relative
        if sha(source) != digest:
            raise PermissionError('Historical input pin changed: ' + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise PermissionError('Byte copy changed: ' + relative)
        bindings.append({'file': relative, 'sha256': digest})
    for held in P_ROWS:
        relative = f'inputs/baseline/P_{held}/REFERENCE.json'
        source = asset_root / relative
        info = read(source)
        validate_reference(info, held)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha(source)
        if sha(target) != digest:
            raise PermissionError('Reference copy changed')
        bindings.append({'file': relative, 'sha256': digest})
    return bindings


def verify(run_root, protocol):
    if (protocol['kind'] != 'bounded_representation_robustness_v1'
            or protocol['materials'] != list(P_ROWS)
            or protocol['methods'] != list(METHODS)
            or protocol['masks'] != MASKS):
        raise PermissionError('Exact registered experiment required')
    if source_bindings() != protocol['source_bindings']:
        raise PermissionError('Registered source changed')
    for rec in protocol['input_bindings']:
        if sha(Path(run_root) / 'experiment' / 'inputs' / rec['file']) != rec['sha256']:
            raise PermissionError('Registered input changed: ' + rec['file'])


def load_train(run_root):
    path = Path(run_root) / 'experiment/inputs/data/TRAIN.npz'
    if sha(path) != INPUT_PINS['data/TRAIN.npz']:
        raise PermissionError('Closed TRAIN identity changed')
    with np.load(path, allow_pickle=False) as z:
        y = z['y'].copy()
        gamma = z['gamma'].copy()
        roles = z['roles'].tolist()
        ids = z['group_ids'].tolist()
    hidden = sorted(set(range(39)) - set(MASKS['spread']))
    if (y.shape != (45, 39) or gamma.shape != (45, 39)
            or [i for i, role in enumerate(roles) if role == 'P'] != list(P_ROWS)
            or [i for i, role in enumerate(roles) if role == 'S'] != list(S_ROWS)
            or len(ids) != 45 or len(set(ids)) != 45
            or not np.isfinite(y[list(P_ROWS)]).all()
            or not np.isfinite(y[np.ix_(S_ROWS, MASKS['spread'])]).all()
            or not np.isnan(y[np.ix_(S_ROWS, hidden)]).all()
            or not np.isfinite(gamma).all() or (gamma <= 0).any()):
        raise PermissionError('Exact closed25P/20S information roles required')
    return y, np.log(gamma), ids


def restore_reference(run_root, held):
    info = read(Path(run_root) / 'experiment/inputs/inputs/baseline'
                / f'P_{held}/REFERENCE.json')
    validate_reference(info, held)
    model = CoordinateGP.fixed(Normalization(**info['normalization']),
                               np.log(info['hyper_values']),
                               info['beta_standardized'])
    return model, info


def imported_score_rows(run_root):
    """Previously consumed scores, never recomputed from targets here."""
    base = Path(run_root) / 'experiment/inputs'
    original = read(base / 'inputs/baseline/RESULT.json')['rows']
    candidate = read(base / 'inputs/FROZEN_CANDIDATE_SCORES.json')['new_rows']
    random = read(base / 'results/RANDOM_GAUSSIAN_RESULT.json')['new_rows']
    rows = [dict(row) for row in original + candidate + random]
    models = ('gp', 'generic', 'local', 'gaussian', 'joint_code_gaussian',
              'random_gaussian')
    for model in models:
        selected = [r for r in rows if r['model'] == model]
        if (len(selected) != 75 or {(r['P'], r['mask']) for r in selected}
                != {(held, mask) for held in P_ROWS for mask in MASKS}):
            raise PermissionError('Original score census changed')
    return rows
