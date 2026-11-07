"""Run exactly six registered handcases, never material data or model fitting."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path

import finite_code_math as m


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(serial(value), stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def delta(value):
    return ((F(value), F(1)),)


def case_one(spec):
    components = tuple(delta(v) for v in spec['points'])
    p = tuple(map(F, spec['p']))
    geo = m.geometry(components, p)
    require(geo['Q'] == ((F(2), F(1)), (F(1), F(1))), 'Hand-integrated Q differs')
    rows = []
    for theta in spec['theta']:
        theta = tuple(map(F, theta)); beta = tuple(a-b for a, b in zip(theta[:-1], p[:-1]))
        prediction = m.mix(components, theta)
        for target in map(F, spec['targets']):
            direct = m.crps_energy(prediction, target)-m.crps_energy(geo['reference_law'], target)
            c = m.c_vector(geo, target)
            formula = m.quadratic(geo, beta, c)
            require(direct == formula, 'Energy-score/CDF-quadratic disagreement')
            rows.append({'theta': theta, 'target': target, 'beta': beta, 'c': c, 'gap': direct})
    return {'case': 1, 'Q': geo['Q'], 'identity_positions': len(rows), 'rows': rows}


def case_two(spec):
    components = tuple(delta(v) for v in spec['points']); p = tuple(map(F, spec['p']))
    geo = m.geometry(components, p)
    expected = ((F(-1), F(1)), (F(-1, 4), F(3, 4)))
    require(geo['c_bounds'] == expected, 'Hand-derived all-real bounds differ')
    means = tuple(sum((mass*m.c_vector(geo, target)[j]
                       for target, mass in geo['reference_law']), F())
                  for j in range(geo['dimension']))
    require(means == (F(), F()), 'E_reference c must vanish')
    for left, right in zip(geo['points'][:-1], geo['points'][1:]):
        middle = (left+right)/2
        require(m.c_vector(geo, middle) == tuple((a+b)/2 for a, b in zip(m.c_vector(geo, left), m.c_vector(geo, right))),
                'c is not affine between adjacent atoms')
    require(m.c_vector(geo, F(-10)) == m.c_vector(geo, F(0)), 'Left constant tail differs')
    require(m.c_vector(geo, F(10)) == m.c_vector(geo, F(2)), 'Right constant tail differs')
    zero = m.geometry((delta(0), delta(0), delta(0)), p)
    require(zero['c_bounds'] == ((F(), F()), (F(), F())) and m.rank_exact(zero['Q']) == 0,
            'Zero-width legal score statistic must be identically zero')
    return {'case': 2, 'all_real_bounds': expected, 'reference_mean_c': means,
            'tail_outside_support_checks': 2, 'zero_width_bounds': zero['c_bounds']}


def case_three(spec):
    components = tuple(delta(v) for v in spec['points']); p = tuple(map(F, spec['p']))
    geo = m.geometry(components, p); rows = []
    theta0, theta1 = tuple(map(F, spec['theta0'])), tuple(map(F, spec['theta1']))
    beta0 = tuple(a-b for a, b in zip(theta0[:-1], p[:-1]))
    beta1 = tuple(a-b for a, b in zip(theta1[:-1], p[:-1]))
    for g in map(F, spec['g']):
        lifted = m.lift_context(geo, g)
        theta = tuple((1-g)*a+g*b for a, b in zip(theta0, theta1))
        direct = m.crps_energy(m.mix(components, theta), F(3, 2))-m.crps_energy(geo['reference_law'], F(3, 2))
        formula = m.quadratic(lifted, beta0+beta1, m.c_vector(lifted, F(3, 2)))
        require(direct == formula, 'Context lifted coefficient identity differs')
        require(lifted['dimension'] == 6 and m.rank_exact(lifted['Q']) == 3,
                'Six coordinates must not be mistaken for rank six')
        direction = (F(1, 100), F(-2, 100), F(3, 100))
        vector = tuple(g*a for a in direction) + tuple(-(1-g)*a for a in direction)
        null_theta0 = m.theta_from_beta(p, vector[:3], F(1, 10))
        null_theta1 = m.theta_from_beta(p, vector[3:], F(1, 10))
        require(all(sum((a*b for a, b in zip(row['h'], vector)), F()) == 0 for row in lifted['intervals']),
                'Declared null direction changes CDF')
        for target in (F(-9), F(0), F(3, 2), F(3), F(9)):
            require(m.quadratic(lifted, vector, m.c_vector(lifted, target)) == 0,
                    'Null direction changes any risk')
        box_upper = m.robust_upper(lifted, vector, lifted['c_bounds'])
        require(box_upper == g*(1-g)/5,
                'Ignoring the known range(Q) constraint must retain this exact conservatism counterexample')
        rows.append({'g': g, 'theta_effective': theta, 'gap': direct,
                     'ambient_dimension': 6, 'rank': 3, 'null_direction': vector,
                     'null_feasible_theta0': null_theta0, 'null_feasible_theta1': null_theta1,
                     'null_true_risk_all_targets': F(), 'null_box_upper': box_upper,
                     'known_range_Q_intersection_upper': F()})
    duplicate = (delta(0), delta(0), delta(1)); dp = (F(1, 4), F(1, 4), F(1, 2))
    dg = m.geometry(duplicate, dp); beta = (F(1, 10), F(-1, 10))
    changed = m.theta_from_beta(dp, beta, F(1, 10))
    require(m.rank_exact(dg['Q']) == 1 and m.mix(duplicate, changed) == dg['reference_law'],
            'Nonunique coefficients should preserve the same distribution, no ridge needed')
    return {'case': 3, 'contexts': rows, 'duplicate_component_Q_rank': 1,
            'same_law_different_feasible_theta': changed}


def case_four(spec):
    geo = m.geometry((delta(0), delta(1)), (F(1, 2), F(1, 2)))
    box = ((F(spec['lower']), F(spec['upper'])),)
    rows = []
    for coefficient in map(F, spec['beta']):
        beta = (coefficient,); upper = m.robust_upper(geo, beta, box)
        corners = tuple(m.quadratic(geo, beta, corner)
                        for corner in itertools.product(*[(lower, upper) for lower, upper in box]))
        center = (box[0][0]+box[0][1])/2; radius = (box[0][1]-box[0][0])/2
        require(upper == max(corners) == coefficient**2+2*coefficient*center+2*radius*abs(coefficient),
                'Uniform box support-function upper differs')
        rows.append({'beta': coefficient, 'robust_upper': upper, 'corner_values': corners})
    beta = (F(3, 10),); theta = m.theta_from_beta((F(1, 2), F(1, 2)), beta, F(1, 10))
    target = F(3, 20); actual_c = m.c_vector(geo, target)
    actual = m.crps_energy(m.mix((delta(0), delta(1)), theta), target)-m.crps_energy(geo['reference_law'], target)
    require(actual_c == (F(-7, 20),) and actual == F(-3, 25)
            and m.robust_upper(geo, beta, box) == F(-9, 100) and actual < F(-9, 100),
            'Nontrivial robust repair handcase differs')
    no_label_box = geo['c_bounds']
    require(all(m.robust_upper(geo, (coefficient,), no_label_box) >= 0 for coefficient in map(F, spec['beta'])),
            'Full all-real no-label box spuriously certifies improvement')
    outside_actual = m.quadratic(geo, beta, m.c_vector(geo, F(1)))
    require(outside_actual == F(39, 100) > 0, 'An invalid box must not be called a guarantee')
    return {'case': 4, 'box_uniform_rows': rows, 'learned_feasible_theta': theta,
            'true_target': target, 'true_c_in_declared_box': actual_c,
            'actual_gap': actual, 'robust_gap_upper': F(-9, 100),
            'no_label_box': no_label_box, 'invalid_box_counterexample_actual_harm': outside_actual}


def case_five(spec):
    units = []
    for scale, target in zip(map(F, spec['scales']), map(F, spec['targets'])):
        geo = m.geometry((delta(0), delta(scale), delta(2*scale)), (F(1, 4), F(1, 2), F(1, 4)))
        widths = tuple(upper-lower for lower, upper in geo['c_bounds'])
        units.append({'c': m.c_vector(geo, target), 'bounds': geo['c_bounds'], 'a': max(widths),
                      'mid': tuple((lower+upper)/2 for lower, upper in geo['c_bounds'])})
    total = tuple(sum((u['c'][j] for u in units), F()) for j in range(2))
    mid = tuple(sum((u['mid'][j] for u in units), F()) for j in range(2))
    rows = []
    for length in range(3):
        for prefix in itertools.permutations(range(3), length):
            remaining = tuple(i for i in range(3) if i not in prefix)
            A = sum((units[i]['a'] for i in remaining), F())
            past = tuple(sum((units[i]['c'][j]-units[i]['mid'][j] for i in prefix), F()) for j in range(2))
            center = tuple(a+b for a, b in zip(mid, past)); expectation = [F(), F()]; branches = []
            for i in remaining:
                q = units[i]['a']/A
                z = tuple(center[j]+(units[i]['c'][j]-units[i]['mid'][j])/q for j in range(2))
                require(all(center[j]-A/2 <= z[j] <= center[j]+A/2 for j in range(2)), 'Predictable vector coordinate range differs')
                for j in range(2):
                    expectation[j] += q*z[j]
                branches.append({'index': i, 'q': q, 'z': z})
            require(tuple(expectation) == total, 'Vector conditional total unbiasedness differs')
            queried = tuple(sum((units[i]['c'][j] for i in prefix), F()) for j in range(2))
            translated = tuple(total[j]-queried[j] for j in range(2))
            require(translated == tuple(sum((units[i]['c'][j] for i in remaining), F()) for j in range(2)), 'Subtract actual queried vector, not count/mean')
            rows.append({'prefix': prefix, 'remaining': remaining, 'center': center, 'range_length': A,
                         'branches': branches, 'expected_z': expectation, 'actual_remaining_c': translated})
    require(len(rows) == 10, 'Exactly ten proper prefixes')
    # Three legal two-atom c values: c=(1/2,-1/2,-1/2), q=1/3.
    centered = {'range_center': F(), 'range_length': F(3), 'true_total': F(-1, 2), 'possible_z': F(3, 2)}
    require(abs(centered['possible_z']-centered['true_total']) == F(2) > centered['range_length']/2,
            'Counterexample: predictable interval need not be centered on the unknown total')
    return {'case': 5, 'units': units, 'fixed_total': total, 'proper_prefix_count': 10,
            'prefixes': rows, 'not_true_total_centered_counterexample': centered,
            'no_CS_coverage_or_material_efficacy_experiment': True}


def case_six(spec):
    geo = m.geometry((delta(0), delta(1)), (F(1, 2), F(1, 2)))
    errors = []
    actions = (
        ('float_input', lambda: m.exact(0.5)),
        ('bool_input', lambda: m.exact(True)),
        ('negative_mass', lambda: m.law(((0, '2'), (1, '-1')))),
        ('invalid_g', lambda: m.lift_context(geo, F(11, 10))),
        ('invalid_epsilon', lambda: m.theta_from_beta((F(1, 2), F(1, 2)), (F(),), F(-1, 10))),
        ('support_floor_violation', lambda: m.theta_from_beta((F(1, 2), F(1, 2)), (F(1, 2),), F(1, 10))),
    )
    for name, action in actions:
        try:
            action()
        except (ValueError, TypeError):
            errors.append(name)
        else:
            raise AssertionError('Expected gate rejection: '+name)
    theta0 = m.theta_from_beta((F(1, 2), F(1, 2)), (F(1, 2),), 0)
    require(theta0 == (F(1), F()) and m.mix((delta(0), delta(1)), theta0) == delta(0),
            'epsilon zero permits support deletion; must not advertise retention')
    require(m.theta_from_beta((F(1, 2), F(1, 2)), (F(),), 1) == (F(1, 2), F(1, 2)), 'epsilon one returns reference only')
    h = F(1, int(spec['nonsmooth_denominator']))
    left_slope = (m.robust_upper(geo, (F(),), geo['c_bounds'])-m.robust_upper(geo, (-h,), geo['c_bounds']))/h
    right_slope = (m.robust_upper(geo, (h,), geo['c_bounds'])-m.robust_upper(geo, (F(),), geo['c_bounds']))/h
    require(left_slope == -1-h and right_slope == 1+h, 'Absolute-value potential needs a subgradient/proximal method at zero')
    return {'case': 6, 'rejected_calls': errors, 'epsilon0_support_deleted': theta0,
            'finite_difference_left': left_slope, 'finite_difference_right': right_slope,
            'nondifferentiable_at_zero': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--approval', required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text('utf8'))
    if (plan.get('status') != 'REGISTERED_UNIFORM_SIX_PRINCIPLE_GATE'
            or args.approval != 'ROOT_UNIFORM_SIX_PRINCIPLE_GATE_ONCE'):
        raise ValueError('Actual root registration/approval required before execution')
    base = Path(__file__).resolve().parent
    for name, digest in plan['source_sha256'].items():
        require(sha(base/name) == digest, 'Registered source changed')
    handfile = Path(plan['handcases']['file'])
    require(sha(handfile) == plan['handcases']['sha256'], 'Registered handcases changed')
    spec = json.loads(handfile.read_text('utf8'))
    require(len(spec['cases']) == 6, 'Exactly six fixed cases required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write(args.output_dir/'START.json', {'actual_utc': started, 'plan_sha256': sha(args.plan),
          'root_approval': args.approval, 'new_material_reads_or_fits': 0})
    results = []
    try:
        for function, case in zip((case_one, case_two, case_three, case_four, case_five, case_six), spec['cases']):
            results.append(function(case))
        result = {'status': 'PASS_SIX_EXACT_PRINCIPLE_HANDCASES', 'cases': results}
    except Exception as error:
        result = {'status': 'FAIL', 'completed_cases': results, 'error_type': type(error).__name__, 'error': str(error)}
    result.update({'actual_start_utc': started, 'actual_end_utc': datetime.now(timezone.utc).isoformat(),
                   'plan_sha256': sha(args.plan), 'new_fits_or_materials': 0,
                   'current512_access_or_changes': 0, 'monte_carlo_coverage_trials': 0,
                   'limit': 'Exact finite handcases validate identities, not a new theorem, optimizer, material gain or CS power.'})
    write(args.output_dir/'RESULTS.json', result)
    print(json.dumps({'status': result['status'], 'completed_cases': len(results), 'result_sha256': sha(args.output_dir/'RESULTS.json')}))
    if result['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
