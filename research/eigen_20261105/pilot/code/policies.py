"""Eight matched policies; hypercycle names denote a predictive heuristic only.

The catalytic matrix uses model-based gains, not independently measured causal
transfer. Replicator/exponential weighting is classical, not a novelty claim.
"""
from dataclasses import dataclass, field
import numpy as np

from benchmark_io import POLICIES


def catalytic_update(weights, matrix, *, remove_cross=False):
    w, c = np.asarray(weights, dtype=float), np.asarray(matrix, dtype=float)
    if (w.shape != (2,) or c.shape != (2, 2) or not np.isfinite(w).all()
            or not np.isfinite(c).all() or (w <= 0).any()
            or not np.isclose(w.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError('Positive normalized weights and finite2x2 gains required')
    used = np.diag(np.diag(c)) if remove_cross else c.copy()
    fitness = used @ w
    logits = np.log(w) + .5 * fitness
    soft = np.exp(logits - np.max(logits))
    updated = .9 * (soft / soft.sum()) + .05
    if not np.isfinite(updated).all() or (updated <= 0).any():
        raise ArithmeticError('Catalytic resource update failed')
    return updated, {'C_predictive': c.tolist(), 'C_used': used.tolist(),
                     'fitness': fitness.tolist(), 'weights_before': w.tolist(),
                     'weights_after': updated.tolist(), 'remove_cross': remove_cross,
                     'independently_measured_causal_gain': False}


@dataclass
class PolicyState:
    weights: np.ndarray = field(default_factory=lambda: np.asarray([.5, .5]))


def score_actions(model, actions, test_indices, frontier_target_indices, normalizers):
    return [dict(action, **model.acquisition_gains(action['indices'], test_indices,
                 frontier_target_indices, normalizers['initial_test_trace'],
                 normalizers['initial_frontier_sd'], normalizers['utility_sign'])) for action in actions]


def _best(scored, key):
    if not scored:
        raise ValueError('No eligible actions')
    # The incoming row/bundle ordering is fixed; exact ties preserve that order.
    return max(range(len(scored)), key=lambda i: scored[i][key])


def choose_action(policy, scored, step, uniform, state):
    if policy not in POLICIES or not scored or type(step) is not int or step < 0:
        raise ValueError('Known policy and nonempty scored actions required')
    if not np.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError('Fixed uniform variate must lie in[0,1)')
    roles = {kind: [i for i, a in enumerate(scored) if a['kind'] == kind] for kind in ('R', 'E')}
    if any(not np.isfinite(a[k]) for a in scored for k in ('IVR', 'KG', 'joint')):
        raise ValueError('Finite gains required')
    receipt = {'policy': policy, 'step': step, 'fixed_uniform': float(uniform),
               'known_classical_policy_or_heuristic': True}
    if policy == 'random':
        selected = int(uniform * len(scored))
    elif policy in ('repair_only_IVR', 'expansion_only_KG'):
        role, gain = ('R', 'IVR') if policy == 'repair_only_IVR' else ('E', 'KG')
        choices = roles[role] or list(range(len(scored)))
        selected = choices[_best([scored[i] for i in choices], gain)]
        receipt['role_fallback'] = not bool(roles[role])
    elif policy == 'joint_lookahead_all':
        selected = _best(scored, 'joint')
    else:
        proposals = {kind: (choices[_best([scored[i] for i in choices], 'joint')] if choices else None)
                     for kind, choices in roles.items()}
        receipt['proposals'] = {kind: None if i is None else {'row': scored[i]['row'],
                    'bundle': scored[i]['bundle'], 'IVR': scored[i]['IVR'], 'KG': scored[i]['KG']}
                               for kind, i in proposals.items()}
        if policy == 'fixed_cycle':
            kind = 'R' if step % 2 == 0 else 'E'
        elif policy == 'fixed_random_modules':
            kind = 'R' if uniform < .5 else 'E'
            receipt['fixed_module_weights'] = [.5, .5]
            receipt['adaptive_C_update'] = False
        else:
            c = np.zeros((2, 2))
            for j, kind in enumerate(('R', 'E')):
                if proposals[kind] is not None:
                    a = scored[proposals[kind]]
                    c[j, :] = a['IVR'], a['KG']
            state.weights, update = catalytic_update(state.weights, c,
                                            remove_cross=policy == 'hypercycle_no_cross')
            receipt.update(update)
            kind = 'R' if uniform < state.weights[0] else 'E'
        if proposals[kind] is None:
            kind = 'E' if kind == 'R' else 'R'
            receipt['role_fallback'] = True
        selected = proposals[kind]
    action = scored[selected]
    receipt['chosen'] = dict(action)
    return action, receipt
