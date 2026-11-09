"""SOURCE ONLY: fixed strongest marginals + selected frozen dependence holdout.

Four explicit phases, each once-only and root registered. prepare has no Y;
predict reads exactly own six anchors once/curve and freezes all105 laws;
reveal needs independent preY PASS + exact root release before120 target reads;
score reports every predeclared law. Only three ORIGINAL future_expansion
recipes are eligible; four final_confirm recipes remain protected. No kernel
fit, seed search, mixture, hyperparameter update or changed dev-role labels.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtr
from source_reader import ANCHORS, PANEL, ExpansionReader, pure_io


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = HERE / 'EXPANSION_PROTOCOL.json'
DEPENDENCIES = {
    'baselines/pure_student.py': '567212afb8a48cebcdbf2b69f051f3a1ecfbc7b3d10e5f3bf45181aebb59c892',
    'marginal_transport/pure.py': '0db9e373dbbe35e2efa805606b9edcb065cb2d920b3f3745ae2fcf6e9529acc4',
    'material_stress/frozen_grid_reference.py': 'd0c77cbc3a09b776c5a870d1a770b9afe2a4ea8b726dddfa3394e924fbc011bc',
    'material_stress/resin_io.py': '00510a21819d2cce4a67297ddad40894c7f6cc3659b030d186b2ef0343d4a3b8',
    'core/frozen_loader.py': 'd36469aae6222cd8470f7e4f3d3d9111a27c8bd69c6c48d82fc2346b883dbe53',
    'FROZEN_DEPENDENCIES.json': '76bcdac23b063909160e661911c02f1ccdcb578342e548d4355ac12dea553580',
    'marginal_transport/DECISION_GATE_BEFORE_DEV.json': '9a91170a8806783f9364f87f162e872d8b51b4a3ada41c44fc20f4bf481cf9f0'}
METHODS = ('physical_full_student', 'copula_independence', 'copula_physical_student_shape',
           'copula_frozen_gp_posterior', 'copula_frozen_curvature_posterior',
           'gaussian_frozen_gp', 'gaussian_frozen_curvature')


def io():
    return pure_io()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return io().read_json(path)


def write(path, value):
    io().write_new(path, value)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_pinned(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEPENDENCIES[relative]:
        raise ValueError('Fixed dependency bytes changed: ' + relative)
    name = '_mlst_expansion_' + path.stem + '_' + DEPENDENCIES[relative]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def authorize(registration, phase):
    reg = read(registration)
    if reg.get('schema') != 'mlst-resin-expansion-registration-v1' or phase not in reg.get('authorized_phases', []):
        raise PermissionError('Separate root actual-UTC expansion source registration required')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture timezone-aware root UTC required')
    protocol = read(PROTOCOL)
    if reg['protocol_id'] != protocol['protocol_id'] or reg['protocol_sha256'] != sha(PROTOCOL):
        raise PermissionError('Expansion protocol bytes changed')
    pins = {str(Path(p).resolve()).casefold(): value for p, value in reg['source_pins'].items()}
    for path in list(HERE.glob('*.py')) + [ROOT / rel for rel in DEPENDENCIES]:
        if pins.get(str(path.resolve()).casefold()) != sha(path):
            raise PermissionError('Missing/changed expansion source/dependency pin: ' + str(path))
    for relative, expected in DEPENDENCIES.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Existing immutable dependency changed: ' + relative)
    if sha(reg['development_selection_receipt']) != reg['development_selection_receipt_sha256']:
        raise PermissionError('Pre-expansion selected-candidate decision receipt changed')
    selection = read(reg['development_selection_receipt'])
    if (selection.get('status') != 'PASS' or not selection.get('expansion_gate_passed') or
            selection.get('selected_candidate') != 'frozen_curvature_posterior'):
        raise PermissionError('Completed development gate must fix curvature dependence before expansion')
    if sha(protocol['source_zip']) != io().SOURCE_SHA256:
        raise ValueError('Original resin ZIP bytes changed')
    inputs = io().load_inputs(protocol['input_masks'])
    if (protocol['anchor_indices'] != ANCHORS or protocol['target_indices'] != PANEL or
            protocol['method_ids'] != list(METHODS) or protocol['authorized_role'] != 'future_expansion'):
        raise ValueError('Expansion role/layout/law protocol differs from fixed design')
    return reg, protocol, inputs


@contextmanager
def phase_path(output, phase, registration):
    path = Path(output) / phase
    path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write(path / 'STARTED.json', {'actual_utc': utc(), 'phase': phase,
          'registration_sha256': sha(registration), 'execution_command': sys.argv,
          'authorized_role': 'future_expansion', 'final_confirm_runs_protected': 20})
    try:
        yield path, start
    except BaseException as error:
        write(path / 'FAILED.json', {'actual_utc': utc(), 'phase': phase,
              'error_type': type(error).__name__, 'message': str(error),
              'seconds': time.perf_counter() - start,
              'decision': 'STOP_KEEP_PARTIAL_NO_REPLACEMENT_NO_RETRY_NO_FLOOR'})
        raise


def complete(path, filename, start, **facts):
    write(path / 'COMPLETE.json', {'actual_utc': utc(), 'status': 'COMPLETE',
          'payload': filename, 'payload_sha256': sha(path / filename),
          'seconds': time.perf_counter() - start, **facts})


def prior(output, phase, filename, registration):
    base = Path(output) / phase
    receipt = read(base / 'COMPLETE.json')
    if receipt['status'] != 'COMPLETE' or receipt['payload'] != filename or receipt['payload_sha256'] != sha(base / filename):
        raise PermissionError('Previous completed phase and exact bytes required')
    data = read(base / filename)
    if data['registration_sha256'] != sha(registration):
        raise ValueError('Previous phase belongs to another root registration')
    return data


def prepare(output, registration, reg, protocol, inputs, *, checkpoint_root=None, pins_path=None):
    roles = io().assign_recipe_roles(inputs)
    specs = sorted([r for r in inputs['per_run'] if roles[r['recipe']]['role'] == 'future_expansion'],
                   key=lambda row: row['member'])
    if len(specs) != 15 or len({r['recipe'] for r in specs}) != 3:
        raise ValueError('Exactly original three expansion recipes/allfive runs required')
    adapter = load_pinned('material_stress/frozen_grid_reference.py')
    kwargs = {}
    if checkpoint_root is not None:
        kwargs['checkpoint_root'] = checkpoint_root
    if pins_path is not None:
        kwargs['pins_path'] = pins_path
    with phase_path(output, 'prepare', registration) as (path, start):
        curves = []
        for spec in specs:
            gamma = np.asarray(spec['native_gamma_sorted_per_s'])
            if gamma.shape != (25,) or (gamma <= 0).any() or (np.diff(gamma) <= 0).any():
                raise ValueError('Native25 ordered positive coordinate grid required')
            bundle = adapter.build_frozen_grid_reference(np.log(gamma), **kwargs)
            reference = bundle.reference
            laws = {}
            for name, theta in [('gp_theta0', bundle.theta_gp),
                                ('curvature_frozen_theta', bundle.theta_curvature_gaussian)]:
                mean, covariance = reference.joint(theta)
                laws[name] = {'theta': theta.tolist(), 'mean': mean.tolist(),
                              'covariance': covariance.tolist()}
            curve = {key: spec[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                                               'source_member_sha256')}
            curve.update(role='future_expansion', native_gamma=gamma.tolist(), log_gamma=np.log(gamma).tolist(),
                         reference={'mean': reference.mean.tolist(), 'covariance': reference.covariance.tolist(),
                                    'cross': reference.cross.tolist()}, laws=laws,
                         anchor_indices=ANCHORS, target_indices=PANEL, adapter_metadata=bundle.metadata)
            curves.append(curve)
        write(path / 'PREPARED.json', {'schema': 'mlst-resin-expansion-prepared-v1',
            'actual_utc': utc(), 'registration_sha256': sha(registration),
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha(PROTOCOL),
            'source_zip_sha256': io().SOURCE_SHA256, 'input_sha256': io().INPUT_SHA256,
            'authorized_role': 'future_expansion', 'roles': roles, 'curves': curves,
            'numerical_eta_accessed': 0, 'frozen_covariance_updates': 0,
            'final_confirm_runs_protected': 20,
            'independence_scope': 'same public source, held-out recipe groups; not new laboratory or certified new batches'})
        complete(path, 'PREPARED.json', start, curves=15, numerical_eta_accessed=0)


def predict(output, registration, reg, protocol, inputs, *, checkpoint_root=None):
    prepared = prior(output, 'prepare', 'PREPARED.json', registration)
    student = load_pinned('baselines/pure_student.py')
    copula = load_pinned('marginal_transport/pure.py')
    loader = load_pinned('core/frozen_loader.py')
    core_path = (None if checkpoint_root is None else
                 Path(checkpoint_root) / 'runtime' / 'core' / 'joint_code_gaussian.py')
    original = loader.load_frozen_core(core_path)
    with phase_path(output, 'predict', registration) as (path, start):
        reader = ExpansionReader(protocol['source_zip'], inputs,
                                 authorized_role='future_expansion', stage='anchors')
        forecasts = []
        try:
            for curve in prepared['curves']:
                values = reader.read_log_eta(curve['member'], ANCHORS)
                x = np.asarray(curve['log_gamma'])
                physical = student.fit_predict(x[ANCHORS], values, x[PANEL], 'power_law')
                ref = curve['reference']
                reference = original.CodeReference(np.asarray(ref['mean']), np.asarray(ref['covariance']),
                                                    np.asarray(ref['cross']))
                gaussian = {}
                for key in ('gp_theta0', 'curvature_frozen_theta'):
                    mean, covariance = reference.condition(curve['laws'][key]['theta'], ANCHORS, values, PANEL)
                    np.linalg.cholesky(covariance)
                    gaussian[key] = (mean, covariance)
                identity = {key: curve[key] for key in ('member', 'recipe', 'run_id', 'thermal_history', 'role')}
                identity.update(anchor_indices=ANCHORS, anchor_values=values, target_indices=PANEL)
                forecasts.append({**identity, 'method_id': METHODS[0], **physical.as_dict()})
                scale = np.sqrt(np.diag(physical.scale_matrix))
                correlations = [np.eye(8), copula.correlation_from_covariance(physical.scale_matrix),
                                copula.correlation_from_covariance(gaussian['gp_theta0'][1]),
                                copula.correlation_from_covariance(gaussian['curvature_frozen_theta'][1])]
                for method_id, correlation in zip(METHODS[1:5], correlations):
                    law = copula.GaussianCopulaStudent(physical.location, scale, correlation)
                    forecasts.append({**identity, 'method_id': method_id, **law.as_dict(),
                                      'quantiles05_50_95': law.quantile([.05, .5, .95]).tolist(),
                                      'marginal_parent_method_id': METHODS[0]})
                for method_id, key in zip(METHODS[5:], ('gp_theta0', 'curvature_frozen_theta')):
                    mean, covariance = gaussian[key]
                    forecasts.append({**identity, 'method_id': method_id, 'reference_id': key,
                                      'distribution': 'multivariate_gaussian',
                                      'mean': mean.tolist(), 'covariance': covariance.tolist()})
        finally:
            write(path / 'ACCESS.json', reader.receipt())
            reader.close()
        if (len(forecasts) != 105 or reader.receipt()['numerical_eta_access_calls'] != 90 or
                reader.receipt()['unique_eta_cells_accessed'] != 90):
            raise ValueError('Exactly105 laws from90 own-anchor cells, each read once, required')
        write(path / 'PREDICTIONS.json', {'schema': 'mlst-resin-expansion-predictions-v1',
            'actual_utc': utc(), 'registration_sha256': sha(registration),
            'prepare_sha256': sha(Path(output) / 'prepare' / 'PREPARED.json'),
            'protocol_id': protocol['protocol_id'], 'authorized_role': 'future_expansion',
            'forecasts': forecasts, 'OLS_fits': 15, 'covariance_updates': 0,
            'reader_access_sha256': sha(path / 'ACCESS.json'), 'target_labels_accessed': 0,
            'anchor_cells_accessed': 90, 'final_confirm_runs_protected': 20,
            'selected_candidate_fixed_before_anchor_read': 'copula_frozen_curvature_posterior',
            'selected_using_development_only': True,
            'marginal_identity': 'all four copulas have exactly physical_full_student univariate t4 marginals'})
        complete(path, 'PREDICTIONS.json', start, laws=105, OLS_fits=15,
                 unique_anchor_cells=90, target_labels_accessed=0)


def reveal(output, registration, reg, protocol, inputs, audit_path, release_path):
    prepared = prior(output, 'prepare', 'PREPARED.json', registration)
    predictions = prior(output, 'predict', 'PREDICTIONS.json', registration)
    prepared_sha = sha(Path(output) / 'prepare' / 'PREPARED.json')
    predicted_sha = sha(Path(output) / 'predict' / 'PREDICTIONS.json')
    if predictions['prepare_sha256'] != prepared_sha:
        raise ValueError('Expansion predictions and prepared law identity mismatch')
    audit, release = read(audit_path), read(release_path)
    if (audit.get('status') != 'PASS' or audit.get('audited_expansion_predictions_sha256') != predicted_sha or
            audit.get('audited_expansion_prepare_sha256') != prepared_sha):
        raise PermissionError('Exact-hash independent preY105law audit PASS required')
    if (release.get('schema') != 'mlst-resin-expansion-target-release-v1' or
            release.get('authorization') != 'REVEAL_ONLY_120_ORIGINAL_FUTURE_EXPANSION_TARGETS' or
            release.get('predictions_sha256') != predicted_sha or release.get('prepare_sha256') != prepared_sha or
            release.get('audit_sha256') != sha(audit_path)):
        raise PermissionError('Exact root expansion target release required')
    stamp = dt.datetime.fromisoformat(release['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture target release UTC required')
    with phase_path(output, 'reveal', registration) as (path, start):
        reader = ExpansionReader(protocol['source_zip'], inputs, authorized_role='future_expansion',
                                 stage='targets', target_release=release)
        curves = []
        try:
            for curve in prepared['curves']:
                row = {key: curve[key] for key in ('member', 'recipe', 'run_id', 'thermal_history', 'role')}
                row.update(target_indices=PANEL, values=reader.read_log_eta(curve['member'], PANEL))
                curves.append(row)
        finally:
            write(path / 'ACCESS.json', reader.receipt()); reader.close()
        if len(curves) != 15 or reader.receipt()['unique_eta_cells_accessed'] != 120:
            raise ValueError('Only original fifteen expansion curves and120 target cells required')
        write(path / 'TARGETS.json', {'schema': 'mlst-resin-expansion-targets-v1', 'actual_utc': utc(),
            'registration_sha256': sha(registration), 'prepare_sha256': prepared_sha,
            'predictions_sha256': predicted_sha, 'audit_sha256': sha(audit_path),
            'root_release_sha256': sha(release_path), 'curves': curves,
            'reader_access_sha256': sha(path / 'ACCESS.json'), 'final_confirm_runs_protected': 20})
        complete(path, 'TARGETS.json', start, unique_expansion_targets=120, final_confirm_runs_protected=20)


def gaussian_scores(mean, covariance, values):
    mean, covariance, y = np.asarray(mean), np.asarray(covariance), np.asarray(values)
    lower = np.linalg.cholesky(covariance)
    error = y - mean
    sigma = np.sqrt(np.diag(covariance))
    z = error / sigma
    crps = sigma*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
    q = 1.6448536269514722
    coverage, width = np.abs(error) <= q*sigma, 2*q*sigma
    standardized = np.linalg.solve(lower, error)
    nll = (len(y)*math.log(2*math.pi)+2*np.log(np.diag(lower)).sum()+standardized@standardized)/(2*len(y))
    metrics = {'crps': float(np.mean(crps)), 'log_rmse': float(np.sqrt(np.mean(error**2))),
               'coverage90': float(np.mean(coverage)), 'width90': float(np.mean(width)),
               'joint_nll_per_dimension': float(nll)}
    points = [{'truth': float(y[i]), 'location': float(mean[i]), 'sd': float(sigma[i]),
               'crps': float(crps[i]), 'squared_error': float(error[i]**2),
               'coverage90': bool(coverage[i]), 'width90': float(width[i])} for i in range(8)]
    return metrics, points


def score(output, registration, reg, protocol):
    prepared = prior(output, 'prepare', 'PREPARED.json', registration)
    predicted = prior(output, 'predict', 'PREDICTIONS.json', registration)
    targets = prior(output, 'reveal', 'TARGETS.json', registration)
    if (targets['predictions_sha256'] != sha(Path(output)/'predict'/'PREDICTIONS.json') or
            targets['prepare_sha256'] != sha(Path(output)/'prepare'/'PREPARED.json')):
        raise ValueError('Expansion labels are not bound to these frozen forecasts')
    student, copula = load_pinned('baselines/pure_student.py'), load_pinned('marginal_transport/pure.py')
    lookup = {row['member']: row for row in targets['curves']}
    physical = {row['member']: row for row in predicted['forecasts'] if row['method_id'] == METHODS[0]}
    with phase_path(output, 'score', registration) as (path, start):
        baseline = {}
        for member, forecast in physical.items():
            baseline[member] = student.score_saved_law(forecast['location'], forecast['scale_matrix'],
                                                       4, lookup[member]['values'])
        rows, point_terms = [], []
        for forecast in predicted['forecasts']:
            member, method = forecast['member'], forecast['method_id']
            truth = lookup[member]
            if truth['target_indices'] != PANEL or truth['recipe'] != forecast['recipe']:
                raise ValueError('The same fixed target panel and original grouping are required')
            base_metrics, base_points = baseline[member]
            if method == METHODS[0]:
                metrics, points = dict(base_metrics), [dict(p) for p in base_points]
            elif method in METHODS[1:5]:
                source = physical[member]
                if (forecast['location'] != source['location'] or
                        forecast['scale'] != np.sqrt(np.diag(np.asarray(source['scale_matrix']))).tolist() or
                        forecast['df'] != 4):
                    raise ValueError('Copula marginals must be exactly the original physical Student4')
                law = copula.GaussianCopulaStudent(forecast['location'], forecast['scale'], forecast['correlation'])
                metrics = dict(base_metrics)
                metrics['joint_nll_per_dimension'] = -law.logpdf(truth['values'])/8
                points = [{**p, 'marginal_metrics_inherited_exactly': True} for p in base_points]
            else:
                metrics, points = gaussian_scores(forecast['mean'], forecast['covariance'], truth['values'])
            identity = {key: forecast[key] for key in ('member', 'recipe', 'run_id', 'thermal_history', 'role', 'method_id')}
            rows.append({**identity, **metrics,
                         'joint_nll_delta_vs_full_student': metrics['joint_nll_per_dimension']-base_metrics['joint_nll_per_dimension']})
            point_terms.extend({**identity, 'target_index': PANEL[i], **point} for i, point in enumerate(points))
        keys = ['crps', 'joint_nll_per_dimension', 'log_rmse', 'coverage90', 'width90', 'joint_nll_delta_vs_full_student']
        recipes = sorted({row['recipe'] for row in rows})
        recipe_scores, macro = [], []
        for method in METHODS:
            for recipe in recipes:
                selected = [r for r in rows if r['method_id']==method and r['recipe']==recipe]
                if len(selected) != 5:
                    raise ValueError('Five original run groups per recipe/method required')
                recipe_scores.append({'method_id': method, 'recipe': recipe, 'runs': 5,
                    **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
            selected = [r for r in recipe_scores if r['method_id']==method]
            macro.append({'method_id': method, 'recipes': 3,
                **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
        values = {r['method_id']:r for r in macro}
        per_recipe = []
        for recipe in recipes:
            find = lambda method: next(r['joint_nll_per_dimension'] for r in recipe_scores
                                       if r['method_id']==method and r['recipe']==recipe)
            per_recipe.append({'recipe':recipe, 'primary_delta':find(METHODS[4])-find(METHODS[0]),
                               'secondary_delta_curv_minus_gp':find(METHODS[4])-find(METHODS[3])})
        if len(rows)!=105 or len(point_terms)!=840 or len(recipes)!=3:
            raise ValueError('Bounded105laws/840terms/threeexpansionrecipe cardinality failed')
        write(path/'SCORES.json', {'schema':'mlst-resin-expansion-scores-v1','actual_utc':utc(),
            'registration_sha256':sha(registration), 'protocol_id':protocol['protocol_id'],
            'predictions_sha256':sha(Path(output)/'predict'/'PREDICTIONS.json'),
            'targets_sha256':sha(Path(output)/'reveal'/'TARGETS.json'),
            'curve_scores':rows,'point_terms':point_terms,'recipe_scores':recipe_scores,'macro_by_method':macro,
            'primary':{'contrast':'copula_frozen_curvature_posterior minus physical_full_student jointNLL/8',
                       'recipe_macro_joint_nll_delta':values[METHODS[4]]['joint_nll_per_dimension']-values[METHODS[0]]['joint_nll_per_dimension'],
                       'direction':'negative_improves','per_recipe':per_recipe},
            'secondary':{'contrast':'curvature copula minus GP copula jointNLL/8',
                         'recipe_macro_joint_nll_delta':values[METHODS[4]]['joint_nll_per_dimension']-values[METHODS[3]]['joint_nll_per_dimension']},
            'marginal_CRPS_identity':'allfourcopulas exactly equal original physical Student by construction, not empirical marginal improvement',
            'grouping':'same8targets/run ->five originalruns/recipe ->equalthree recipe macro; mean runRMSE',
            'independence_scope':'same public source held-out recipes, not independent laboratory or certified new batches',
            'OLS_fits':15,'covariance_updates':0,'unique_anchor_cells':90,'unique_target_cells':120,
            'final_confirm_runs_protected':20,'actual_new_measurements':0,'actual_experiment_savings_measured':False})
        complete(path,'SCORES.json',start,laws_scored=105,point_terms=840,final_confirm_runs_protected=20)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--checkpoint-root',type=Path)
    parser.add_argument('--pins-path',type=Path)
    parser.add_argument('phase',choices=['prepare','predict','reveal','score'])
    args=parser.parse_args(argv)
    reg,protocol,inputs=authorize(args.registration,args.phase)
    if args.phase=='prepare':
        prepare(args.output,args.registration,reg,protocol,inputs,
                checkpoint_root=args.checkpoint_root,pins_path=args.pins_path)
    elif args.phase=='predict':
        predict(args.output,args.registration,reg,protocol,inputs,checkpoint_root=args.checkpoint_root)
    elif args.phase=='reveal':
        if args.audit is None or args.release is None:
            raise PermissionError('Explicit independent preY audit and root target release paths required')
        reveal(args.output,args.registration,reg,protocol,inputs,args.audit,args.release)
    else:
        score(args.output,args.registration,reg,protocol)
    print('COMPLETE',args.phase,str(args.output.resolve()))


if __name__=='__main__':
    main()
