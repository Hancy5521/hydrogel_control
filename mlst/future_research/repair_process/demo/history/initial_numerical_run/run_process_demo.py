"""Run one registered synthetic repair-process demonstration, with plots.

No training, data reconstruction, material prediction or target scoring occurs.
The exact Gaussian law is a correctness reference, not hidden material truth.
"""
import argparse, hashlib, json, platform
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from repair_graph import grid_fixture, free_problem, activation_depth
from local_repair import mean_relaxation, gibbs_samples, collapse_of_deterministic_ensemble


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def plot_process(folder, fixture, snapshots, result):
    size = fixture['size']
    shown = np.concatenate([r['values'][r['active']] for r in snapshots])
    norm = colors.Normalize(vmin=float(shown.min()), vmax=float(shown.max()))
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('#e0e3e7')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, state in zip(axes.ravel(), snapshots):
        field = state['values'].copy()
        field[~state['active']] = np.nan
        im = ax.imshow(field.reshape(size, size), cmap=cmap, norm=norm, origin='lower')
        ax.set_title(f"Sweep {state['sweep']} | active missing: {state['activated_missing_nodes']}/49", fontsize=11)
        # The fixed boundary of the erased patch has no data value of its own.
        ax.add_patch(plt.Rectangle((2.5, 2.5), 7, 7, fill=False, ec='#222222', lw=1.1, ls='--'))
        ax.set_xticks([0, 6, 12]); ax.set_yticks([0, 6, 12])
        ax.set_xlabel('Synthetic grid column'); ax.set_ylabel('Synthetic grid row')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.85, pad=.02)
    cb.set_label('Synthetic field value (arbitrary units)')
    fig.suptitle('Local repair process | gray = not yet activated | observations fixed outside dashed patch', fontsize=12)
    fig.savefig(folder/'REPAIR_PROCESS.png', dpi=170)
    plt.close(fig)


def plot_probability(folder, fixture, problem, det, samples):
    free = problem['free']
    selected = np.flatnonzero(free // fixture['size'] == fixture['size']//2)
    cols = free[selected] % fixture['size']
    center = fixture['mean'][free] + problem['mean_centered']
    exact_sd = np.sqrt(np.diag(problem['covariance']))
    empirical = fixture['mean'][free] + samples.mean(axis=0)
    lower, upper = np.quantile(samples, [.05, .95], axis=0) + fixture['mean'][free]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.fill_between(cols, (center-1.6448536269514722*exact_sd)[selected],
                    (center+1.6448536269514722*exact_sd)[selected], color='#d4e5f5', label='Exact marginal 90% interval')
    ax.plot(cols, center[selected], '-o', color='#1c4d73', label='Exact conditional mean')
    ax.plot(cols, (fixture['mean'][free]+det)[selected], 'x', color='#d37827', markersize=8, label='Deterministic repair (point)')
    ax.plot(cols, empirical[selected], '--', color='#247955', label='Gibbs empirical mean')
    ax.plot(cols, lower[selected], ':', color='#247955', label='Gibbs empirical 5/95% quantiles')
    ax.plot(cols, upper[selected], ':', color='#247955')
    ax.set_xlabel('Synthetic grid column, middle row'); ax.set_ylabel('Synthetic field value (arbitrary units)')
    ax.set_title('Stable structure is not the same as a degenerate probability distribution')
    ax.legend(fontsize=9, loc='best'); ax.grid(alpha=.2)
    fig.savefig(folder/'REPAIR_PROBABILITY.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('PROTOCOL.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('output'))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text('utf-8-sig'))
    here = Path(__file__).resolve().parent
    for name, expected in protocol['source_sha256'].items():
        if digest(here/name) != expected:
            raise ValueError('Source hash differs from registered prototype: '+name)
    if args.output_dir.exists():
        raise FileExistsError('Refuse to overwrite a prior demonstration')
    args.output_dir.mkdir(parents=True)
    config = protocol['config']
    started = datetime.now(timezone.utc)
    fixture = grid_fixture(**config['graph'])
    problem = free_problem(fixture)
    depth = activation_depth(fixture)
    observed_copy = fixture['values'][fixture['observed']].copy()
    mask_copy = fixture['observed'].copy()
    det, snapshots, max_errors, energies = mean_relaxation(
        fixture, problem, depth, config['deterministic_sweeps'], config['snapshot_sweeps'])
    samples = gibbs_samples(problem, det, **config['gibbs'])
    collapse = collapse_of_deterministic_ensemble(problem, **config['collapse'])
    exact_var = np.diag(problem['covariance'])
    empirical_var = samples.var(axis=0, ddof=1)
    mean_error = float(np.max(np.abs(det-problem['mean_centered'])))
    standardized_mc_mean_error = float(np.max(np.abs(samples.mean(axis=0)-problem['mean_centered'])/np.sqrt(exact_var)))
    relative_mc_var_error = float(np.max(np.abs(empirical_var/exact_var-1)))
    covariance_error = float(np.linalg.norm(np.cov(samples,rowvar=False)-problem['covariance'])/np.linalg.norm(problem['covariance']))
    collapsed_variance = float(np.max(collapse.var(axis=0,ddof=1)))
    obs_unchanged = np.array_equal(observed_copy, fixture['values'][fixture['observed']])
    mask_unchanged = np.array_equal(mask_copy, fixture['observed'])
    checks = {
        'observations_exactly_preserved':bool(obs_unchanged),
        'observed_mask_never_promotes_generated_values':bool(mask_unchanged),
        'fixed_precision_positive_definite':bool(np.linalg.eigvalsh(fixture['precision']).min()>0),
        'all_missing_nodes_activate':bool(snapshots[-1]['activated_missing_nodes']==len(problem['free'])),
        'deterministic_mean_matches_exact':mean_error<=protocol['diagnostic_gates']['deterministic_max_abs_mean_error'],
        'Gibbs_mean_diagnostic':standardized_mc_mean_error<=protocol['diagnostic_gates']['max_standardized_mean_error'],
        'Gibbs_variance_diagnostic':relative_mc_var_error<=protocol['diagnostic_gates']['max_relative_variance_error'],
        'deterministic_ensemble_collapses':collapsed_variance<=protocol['diagnostic_gates']['collapsed_variance_upper_bound'],
    }
    ended = datetime.now(timezone.utc)
    result={
        'classification':'Synthetic process demonstrator, not a trained model, new algorithm or material benchmark',
        'synthetic_only':True,'started_utc':started.isoformat(),'completed_utc':ended.isoformat(),
        'completed_asia_shanghai':ended.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'protocol_sha256':digest(args.protocol),'source_sha256':protocol['source_sha256'],
        'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
        'status':'PASS_FIXED_SYNTHETIC_DIAGNOSTICS' if all(checks.values()) else 'DIAGNOSTIC_FAILURE_RETAINED',
        'checks':checks,'config':config,
        'nodes':len(fixture['mean']),'genuine_synthetic_observed_nodes':int(fixture['observed'].sum()),
        'missing_nodes':len(problem['free']),'activation_depth_max':int(depth.max()),
        'snapshot_missing_activation_counts':[r['activated_missing_nodes'] for r in snapshots],
        'metrics':{'deterministic_max_abs_mean_error':mean_error,
                   'max_standardized_Gibbs_mean_error':standardized_mc_mean_error,
                   'max_relative_Gibbs_variance_error':relative_mc_var_error,
                   'relative_covariance_Frobenius_error':covariance_error,
                   'max_deterministic_ensemble_variance':collapsed_variance},
        'material_fit_forecast_targetscore_calls':0,'new_measurements':0,
        'limits':['Fixed Gaussian precision and local updates are established mathematics',
                  'A surrounded grid hole is not evidence for one-sided curve continuation',
                  'Activation states before convergence are computational transients, not calibrated posteriors',
                  'Monte Carlo gates are declared software diagnostics, not material calibration guarantees',
                  'Random Gibbs trajectories remain variable after reaching stationarity; they should not freeze to one curve',
                  'Generated states never become new observed data']}
    (args.output_dir/'RESULTS.json').write_text(json.dumps(jsonable(result),ensure_ascii=False,indent=2),'utf-8')
    fixture_out={'synthetic_only':True,'coordinates':fixture['coordinates'],'reference_mean':fixture['mean'],
                 'observed_mask':fixture['observed'],'observed_values':fixture['values'],
                 'edges':fixture['edges'],'ridge':config['graph']['ridge'],
                 'free_indices':problem['free'],'exact_conditional_mean':fixture['mean'][problem['free']]+problem['mean_centered'],
                 'exact_conditional_covariance':problem['covariance'],'activation_depth':depth}
    (args.output_dir/'SYNTHETIC_FIXTURE.json').write_text(json.dumps(jsonable(fixture_out),indent=2),'utf-8')
    trajectory={'synthetic_only':True,'snapshots':snapshots,'deterministic_max_error_by_sweep':max_errors,
                'conditional_quadratic_energy_by_sweep':energies}
    (args.output_dir/'TRAJECTORY.json').write_text(json.dumps(jsonable(trajectory),indent=2),'utf-8')
    plot_process(args.output_dir,fixture,snapshots,result)
    plot_probability(args.output_dir,fixture,problem,det,samples)
    receipt={'actual_utc':datetime.now(timezone.utc).isoformat(),'synthetic_only':True,
             'files':[{'file':p.name,'sha256':digest(p),'bytes':p.stat().st_size}for p in sorted(args.output_dir.iterdir())if p.is_file()],
             'all_plots_use_computed_trajectory':'No interpolation or invented repair frames',
             'independent_audit_pending':True}
    (args.output_dir/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2),'utf-8')
    print(json.dumps({'status':result['status'],'synthetic_only':True,'checks':checks,'metrics':result['metrics']}))


if __name__=='__main__':main()
