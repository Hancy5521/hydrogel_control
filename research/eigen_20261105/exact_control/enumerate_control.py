"""Exact finite-world complementarity control, not a new material algorithm.

All worlds and costs are declared algebraic fixtures. No material files, model
forecasts, wet measurements, DFT records, generated data or network are read.
Execution requires a source-pinned plan and root exact-plan approval.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ACTIONS = ('V1', 'V2', 'R', 'E1', 'E2')
STATIC_ORDER = ('R', 'E1', 'E2')
SCENARIOS = ('coupled', 'theta_known', 'fake_pair', 'no_relation')
OBJECTIVES = ('brier', 'verification_shortfall')
POLICIES = ('exact_budget_DP', 'myopic_cost_normalized', 'static_RE_cycle')
TICK = Fraction(1, 4)
COST_SETTINGS = {
    'cheap_repair_expansion': ({'V1': 4, 'V2': 4, 'R': 1, 'E1': 1, 'E2': 1}, 4),
    'equal_cost': ({'V1': 4, 'V2': 4, 'R': 4, 'E1': 4, 'E2': 4}, 8),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(value):
    """Keep rational identity and a convenience display; no float computation."""
    value = Fraction(value)
    return {'fraction': str(value), 'numerator': value.numerator,
            'denominator': value.denominator, 'float_display': float(value)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read(path):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid,
                      object_pairs_hook=unique)


@dataclass(frozen=True)
class World:
    latents: tuple
    targets: tuple
    observations: tuple

    def __post_init__(self):
        if len(self.targets) != 2 or any(type(v) is not int or v not in (-1, 1) for v in self.targets):
            raise ValueError('Two declared binary targets are required')
        if tuple(key for key, _ in self.observations) != ACTIONS:
            raise ValueError('Observation tuple must use the exact fixed action order')
        if any(type(v) is not int or v not in (-1, 1) for _, v in self.observations):
            raise ValueError('Every observation is a binary fixture')
        if len({key for key, _ in self.latents}) != len(self.latents):
            raise ValueError('Latent names must be unique')
        if any(type(v) is not int or v not in (-1, 1) for _, v in self.latents):
            raise ValueError('Latents are explicit independent fair bits')
        if self.observations[0][1] != self.targets[0] or self.observations[1][1] != self.targets[1]:
            raise ValueError('V observations must be the corresponding true targets')


def build_worlds(scenario):
    """Uniform prior over explicitly enumerated worlds, without random sampling."""
    if scenario not in SCENARIOS:
        raise ValueError('Unknown exact-control scenario')
    names = {'coupled': ('theta', 'u1', 'u2'),
             'theta_known': ('u1', 'u2'),
             'fake_pair': ('theta_target', 'u1', 'u2', 'theta_old'),
             'no_relation': ('theta', 'u1', 'u2', 'y1', 'y2')}[scenario]
    worlds = []
    for values in itertools.product((-1, 1), repeat=len(names)):
        v = dict(zip(names, values))
        if scenario == 'no_relation':
            r, y1, y2 = v['theta'], v['y1'], v['y2']
        else:
            theta = 1 if scenario == 'theta_known' else v.get('theta', v.get('theta_target'))
            r = v['theta_old'] if scenario == 'fake_pair' else theta
            y1, y2 = theta * v['u1'], theta * v['u2']
        worlds.append(World(tuple(zip(names, values)), (y1, y2),
                      (('V1', y1), ('V2', y2), ('R', r), ('E1', v['u1']), ('E2', v['u2']))))
    return tuple(worlds)


def validate_belief(worlds, ids):
    if (not isinstance(worlds, tuple) or not worlds or any(not isinstance(w, World) for w in worlds)
            or not isinstance(ids, tuple) or not ids
            or any(type(i) is not int or not 0 <= i < len(worlds) for i in ids)
            or tuple(sorted(set(ids))) != ids):
        raise ValueError('A nonempty sorted unique subset of uniform world IDs is required')
    return ids


def target_probabilities(worlds, ids):
    ids = validate_belief(worlds, ids)
    return tuple(Fraction(sum(worlds[i].targets[j] == 1 for i in ids), len(ids)) for j in range(2))


def bayes_risk(worlds, ids):
    """Mean of the two Bayes Brier risks for positive-target probabilities."""
    return sum((p * (1 - p) for p in target_probabilities(worlds, ids)), Fraction(0)) / 2


def transitions(worlds, ids, action):
    ids = validate_belief(worlds, ids)
    if action not in ACTIONS:
        raise ValueError('Unknown observation action')
    position = ACTIONS.index(action)
    groups = defaultdict(list)
    for i in ids:
        groups[worlds[i].observations[position][1]].append(i)
    return tuple((value, Fraction(len(group), len(ids)), tuple(group)) for value, group in sorted(groups.items()))


def after_actions(worlds, actions, ids=None):
    if not isinstance(actions, tuple) or len(set(actions)) != len(actions) or any(a not in ACTIONS for a in actions):
        raise ValueError('A tuple of distinct declared actions is required')
    ids = tuple(range(len(worlds))) if ids is None else validate_belief(worlds, ids)
    leaves = [(Fraction(1), ids)]
    for action in actions:
        leaves = [(probability * p, next_ids) for probability, current in leaves
                  for _, p, next_ids in transitions(worlds, current, action)]
    return tuple(leaves)


def expected_risk(worlds, actions):
    return sum((p * bayes_risk(worlds, ids) for p, ids in after_actions(worlds, actions)), Fraction(0))


def _uniform_image_entropy(values):
    """Exact bits for these linear-fair-bit fixtures; not a generic log routine."""
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    n = len(counts)
    if n == 0 or n & (n - 1) or len(set(counts.values())) != 1:
        raise ValueError('Exact entropy helper requires a uniform power-of-two image')
    return Fraction(n.bit_length() - 1)


def target_information(worlds, actions, target='joint'):
    if target not in ('Y1', 'Y2', 'joint'):
        raise ValueError('Known target information endpoint required')
    def values(ids):
        return [worlds[i].targets if target == 'joint' else worlds[i].targets[int(target[-1]) - 1]
                for i in ids]
    initial = _uniform_image_entropy(values(range(len(worlds))))
    remainder = sum((p * _uniform_image_entropy(values(ids))
                     for p, ids in after_actions(worlds, actions)), Fraction(0))
    return initial - remainder


def verify_costs(costs, budget):
    if (not isinstance(costs, dict) or set(costs) != set(ACTIONS)
            or any(type(v) is not int or v <= 0 for v in costs.values())
            or type(budget) is not int or budget < 0):
        raise ValueError('All five positive integer-tick costs and a nonnegative integer budget are required')


def mask_from_used(used):
    if not isinstance(used, tuple) or len(set(used)) != len(used) or any(a not in ACTIONS for a in used):
        raise ValueError('Used actions must be a tuple of distinct declared names')
    return sum(1 << ACTIONS.index(a) for a in used)


def _verified_count(mask):
    return int(bool(mask & 1)) + int(bool(mask & 2))


def objective_value(worlds, ids, mask, objective):
    if objective == 'brier':
        return bayes_risk(worlds, ids)
    if objective == 'verification_shortfall':
        return Fraction(2 - _verified_count(mask), 2)
    raise ValueError('Known objective required')


class Policy:
    """Exact DP or fixed classical control over beliefs and used actions."""
    def __init__(self, worlds, costs, budget, kind, objective='brier'):
        verify_costs(costs, budget)
        validate_belief(worlds, tuple(range(len(worlds))))
        if kind not in POLICIES or objective not in OBJECTIVES:
            raise ValueError('Known policy/objective required')
        self.worlds, self.costs, self.budget = worlds, dict(costs), budget
        self.kind, self.objective = kind, objective

        @lru_cache(None)
        def solve(ids, mask, remaining):
            # Stop is always available. Optimize risk first, then charged cost;
            # exact action ties retain the requested V1,V2,R,E1,E2 order.
            best = (objective_value(worlds, ids, mask, objective), Fraction(0), None)
            for action in ACTIONS:
                bit, cost = 1 << ACTIONS.index(action), self.costs[action]
                if mask & bit or cost > remaining:
                    continue
                outcomes = transitions(worlds, ids, action)
                future = [(p, solve(next_ids, mask | bit, remaining - cost)) for _, p, next_ids in outcomes]
                value = sum((p * result[0] for p, result in future), Fraction(0))
                spend = Fraction(cost) + sum((p * result[1] for p, result in future), Fraction(0))
                if (value, spend) < best[:2]:
                    best = (value, spend, action)
            return best
        self._solve = solve

    def choose(self, ids, used=(), remaining=None):
        ids = validate_belief(self.worlds, ids)
        mask = mask_from_used(used)
        remaining = self.budget if remaining is None else remaining
        if type(remaining) is not int or remaining < 0:
            raise ValueError('Nonnegative integer remaining ticks required')
        if self.kind == 'exact_budget_DP':
            return self._solve(ids, mask, remaining)[2]
        if self.kind == 'static_RE_cycle':
            for action in STATIC_ORDER:
                if action not in used and self.costs[action] <= remaining:
                    return action
            return None
        current = objective_value(self.worlds, ids, mask, self.objective)
        best_gain, choice = Fraction(0), None
        for action in ACTIONS:
            bit, cost = 1 << ACTIONS.index(action), self.costs[action]
            if mask & bit or cost > remaining:
                continue
            future = sum((p * objective_value(self.worlds, next_ids, mask | bit, self.objective)
                          for _, p, next_ids in transitions(self.worlds, ids, action)), Fraction(0))
            gain = (current - future) / cost
            if gain > best_gain:
                best_gain, choice = gain, action
        return choice  # None if no strictly positive immediate gain.

    def value(self, ids=None, used=(), remaining=None):
        """DP value tuple: exact objective, expected additional ticks, first action."""
        if self.kind != 'exact_budget_DP':
            raise TypeError('The cached Bellman value is only defined for exact_budget_DP')
        ids = tuple(range(len(self.worlds))) if ids is None else validate_belief(self.worlds, ids)
        mask = mask_from_used(used)
        remaining = self.budget if remaining is None else remaining
        if type(remaining) is not int or remaining < 0:
            raise ValueError('Nonnegative integer remaining ticks required')
        return self._solve(ids, mask, remaining)


def optimize(worlds, costs, budget, objective='brier'):
    return Policy(worlds, costs, budget, 'exact_budget_DP', objective)


def myopic(worlds, costs, budget, objective='brier'):
    return Policy(worlds, costs, budget, 'myopic_cost_normalized', objective)


def static_cycle(worlds, costs, budget, objective='brier'):
    return Policy(worlds, costs, budget, 'static_RE_cycle', objective)


def summarize(policy):
    """Exhaustive policy-tree rollout; all metrics remain exact rational values."""
    worlds = policy.worlds
    initial_ids = tuple(range(len(worlds)))
    leaves = []
    def walk(ids, used, remaining, probability, history):
        action = policy.choose(ids, used, remaining)
        if action is None:
            probabilities = target_probabilities(worlds, ids)
            direct = sum(name in used for name in ('V1', 'V2'))
            known_unverified = sum(p in (0, 1) and 'V' + str(j + 1) not in used
                                   for j, p in enumerate(probabilities))
            leaves.append({'probability': probability, 'belief_ids': ids, 'used': used,
                           'spent_ticks': policy.budget - remaining,
                           'posterior_positive_probabilities': probabilities,
                           'bayes_risk': bayes_risk(worlds, ids),
                           'objective_value': objective_value(worlds, ids, mask_from_used(used), policy.objective),
                           'directly_verified_targets': direct,
                           'inferred_but_unverified_targets': known_unverified,
                           'independent_information_bits_revealed':
                               _uniform_image_entropy([w.latents for w in worlds]) -
                               _uniform_image_entropy([worlds[i].latents for i in ids]),
                           'oracle_calls': len(used), 'history': history})
            return
        if action in used or policy.costs[action] > remaining:
            raise AssertionError('Policy selected an illegal/repeated/unaffordable action')
        for observation, p, next_ids in transitions(worlds, ids, action):
            walk(next_ids, used + (action,), remaining - policy.costs[action], probability * p,
                 history + ({'action': action, 'observation': observation,
                             'cost_ticks': policy.costs[action]},))
    walk(initial_ids, (), policy.budget, Fraction(1), ())
    if sum((leaf['probability'] for leaf in leaves), Fraction(0)) != 1:
        raise AssertionError('Policy leaves must exhaust the prior exactly')
    expectation = lambda field: sum((leaf['probability'] * leaf[field] for leaf in leaves), Fraction(0))
    stats = {field: expectation(field) for field in ('bayes_risk', 'objective_value', 'spent_ticks',
             'directly_verified_targets', 'inferred_but_unverified_targets', 'oracle_calls',
             'independent_information_bits_revealed')}
    stats['charged_oracle_cost'] = stats['spent_ticks'] * TICK
    stats['unmeasured_inference_counts_as_verified_label'] = False
    if policy.kind == 'exact_budget_DP':
        value, spend, _ = policy.value()
        if stats['objective_value'] != value or stats['spent_ticks'] != spend:
            raise AssertionError('Exhaustive rollout must agree with the Bellman values')
    return {'policy': policy.kind, 'objective': policy.objective, 'world_count': len(worlds),
            'budget_ticks': policy.budget, 'tick_cost': TICK, 'costs_ticks': policy.costs,
            'expected': stats, 'leaves': leaves}


def complementarity(scenario):
    worlds = build_worlds(scenario)
    risk = {label: expected_risk(worlds, actions) for label, actions in
            (('none', ()), ('R', ('R',)), ('E1', ('E1',)), ('R_and_E1', ('R', 'E1')))}
    information = {label: {target: target_information(worlds, actions, target) for target in ('Y1', 'Y2', 'joint')}
                   for label, actions in (('R', ('R',)), ('E1', ('E1',)), ('R_and_E1', ('R', 'E1')))}
    return {'scenario': scenario, 'mean_two_target_bayes_risk': risk,
            'mean_brier_complementarity_gain': risk['R'] + risk['E1'] - risk['R_and_E1'] - risk['none'],
            'target_mutual_information_bits': information,
            'Y1_information_complementarity_bits': information['R_and_E1']['Y1'] - information['R']['Y1'] - information['E1']['Y1'],
            'independently_measured_targets_from_R_E': 0}


def pseudo_overconfidence():
    """An independent guessed h is NOT a measured theta or target observation."""
    bad_brier, proper_brier, correct_both = Fraction(0), Fraction(0), Fraction(0)
    for theta, u1, u2, h in itertools.product((-1, 1), repeat=4):
        truth = (theta * u1, theta * u2)
        predicted = (h * u1, h * u2)
        wrong_probability = tuple(Fraction(int(y == 1)) for y in predicted)
        bad_brier += sum(((p - int(y == 1)) ** 2 for p, y in zip(wrong_probability, truth)), Fraction(0)) / 32
        proper_brier += Fraction(1, 4) / 16
        correct_both += Fraction(int(predicted == truth), 16)
    return {'scenario': 'independent_pseudo_theta_treated_as_true_calibration', 'world_count': 16,
            'true_positive_probabilities_given_h_E1_E2': (Fraction(1, 2), Fraction(1, 2)),
            'reported_wrong_posterior_bayes_risk': Fraction(0),
            'actual_wrong_posterior_mean_Brier': bad_brier,
            'proper_posterior_mean_Brier': proper_brier,
            'probability_both_confident_pseudo_targets_correct': correct_both,
            'generated_h_measurement_count': 0, 'directly_verified_target_count': 0,
            'charged_real_oracle_actions': ('E1', 'E2'), 'cheap_charged_oracle_cost': Fraction(1, 2),
            'h_role': 'independent_generated_guess_not_an_observation',
            'pseudo_predictions_are_not_measured_labels': True}


def json_ready(value):
    if isinstance(value, Fraction):
        return exact(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def plan_configuration():
    return {'version': 'finite_complementarity_control_v1',
            'scenarios': list(SCENARIOS), 'objectives': list(OBJECTIVES), 'policies': list(POLICIES),
            'action_order': list(ACTIONS), 'static_RE_order': list(STATIC_ORDER),
            'tick_cost': '1/4', 'cost_settings': {name: {'costs_integer_ticks': costs, 'budget_integer_ticks': budget}
                                                for name, (costs, budget) in COST_SETTINGS.items()},
            'DP_tie_break': 'minimum exact objective, then minimum expected charged ticks, then fixed action order',
            'myopic_rule': 'positive immediate exact objective reduction per charged integer tick; ties fixed action order; otherwise stop',
            'uniform_priors': True, 'no_material_file_reads': True,
            'mandatory_verification_counts_only': ['V1', 'V2']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    expected_keys = {'kind', 'status', 'actual_registered_utc', 'source_sha256', 'configuration',
                     'failure_policy', 'not_material_experiment_or_novel_algorithm'}
    if (not isinstance(plan, dict) or set(plan) != expected_keys
            or plan['kind'] != 'source_pinned_exact_logical_control'
            or plan['status'] != 'REGISTERED'
            or not isinstance(plan['actual_registered_utc'], str) or not plan['actual_registered_utc']
            or plan['source_sha256'] != {'enumerate_control.py': digest(__file__)}
            or plan['configuration'] != plan_configuration()
            or plan['not_material_experiment_or_novel_algorithm'] is not True):
        raise ValueError('An exact registered source-pinned control plan is required')
    if (approval.get('approved') is not True
            or approval.get('scope') != 'EXACT_HYPERCYCLE_LOGICAL_CONTROL'
            or approval.get('plan_sha256') != digest(args.plan)):
        raise ValueError('Root exact-plan approval required before enumeration')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'plan_sha256': digest(args.plan),
        'approval_sha256': digest(args.approval), 'source_sha256': digest(__file__),
        'data': 'Only declared algebraic binary worlds; no material file access'})
    try:
        policies = []
        for scenario in SCENARIOS:
            worlds = build_worlds(scenario)
            for setting, (costs, budget) in COST_SETTINGS.items():
                for objective in OBJECTIVES:
                    for make_policy in (optimize, myopic, static_cycle):
                        result = summarize(make_policy(worlds, costs, budget, objective))
                        policies.append(dict(result, scenario=scenario, cost_setting=setting))
        result = {'artifact_kind': 'exact_finite_world_logical_control_not_material_evidence',
            'actual_end_utc': utc(), 'plan_sha256': digest(args.plan), 'source_sha256': digest(__file__),
            'exact_arithmetic': 'fractions.Fraction throughout inference, costs, DP, rollout and scoring',
            'policy_configurations': len(policies), 'policies': policies,
            'complementarity': [complementarity(scenario) for scenario in SCENARIOS],
            'pseudo_overconfidence': pseudo_overconfidence(),
            'new_material_measurements_fits_predictions_or_target_scores': 0,
            'not_claimed': ['new Bayesian algorithm', 'general adaptive advantage theorem',
                            'learned catalytic relations', 'material efficacy', 'top-conference readiness'],
            'mandatory_verification_interpretation': 'Only direct V observations count as target measurements; exact derived target knowledge does not.'}
        write(output / 'EXACT_RESULTS.json', json_ready(result))
        print(json.dumps({'status': 'COMPLETE_EXACT_CONTROL', 'policy_configurations': len(policies),
                          'new_material_evidence': False}, ensure_ascii=False))
    except Exception as error:
        write(output / 'FAILED_CONTROL.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
                                               'message': str(error), 'retain_failure': True})
        raise


if __name__ == '__main__':
    main()
