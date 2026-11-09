"""Plot saved recipe aggregates only; never refit or rescore any material."""
from pathlib import Path
import datetime
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def main():
    original = read('resin_dev_once/score/SCORES.json')
    physical = read('student_dev_once/score/SCORES.json')
    transport = read('transport_dev_once/score/SCORES.json')
    expansion = read('expansion_once/score/SCORES.json')
    def one(table, name):
        return next(r for r in table['macro_by_method'] if r['method_id'] == name)
    labels = ['Frozen GP Gaussian', 'Frozen curvature Gaussian', 'Power-law Student',
              'GP copula + same marginals', 'Curvature copula + same marginals']
    dev = [one(original, 'GP_nearest_target6'), one(original, 'curv_nearest_target6'),
           one(physical, 'power_law__GP_nearest_target6'),
           one(transport, 'frozen_gp_posterior'), one(transport, 'frozen_curvature_posterior')]
    held = [one(expansion, n) for n in ['gaussian_frozen_gp', 'gaussian_frozen_curvature',
             'physical_full_student', 'copula_frozen_gp_posterior', 'copula_frozen_curvature_posterior']]
    colors = ['#2563a6', '#54a7b0', '#4b5563', '#e39b24', '#bb3c3e']
    markers = ['o', 'D', 's', '^', 'v']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    for ax, rows, title in zip(axes, [dev, held],
         ['Development: 3 recipes / 15 runs', 'Held-out recipes: 3 recipes / 15 runs']):
        for label, row, color, marker in zip(labels, rows, colors, markers):
            ax.scatter(row['crps'], row['joint_nll_per_dimension'], color=color,
                       marker=marker, s=85, edgecolor='white', linewidth=.6,
                       label=label, zorder=3)
        ax.axvline(rows[2]['crps'], color='#c8c8c8', linestyle=':', linewidth=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Marginal CRPS (lower is better)')
        ax.grid(alpha=.2)
        ax.set_xlim(.02, .085)
        ax.set_ylim(-3.05, -1.05)
    axes[0].set_ylabel('Joint NLL per dimension (lower is better)')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(.5, -.005), frameon=False)
    fig.suptitle('Resin flow curves: matched six-anchor probability scores', fontsize=13)
    fig.subplots_adjust(bottom=.23, top=.83, left=.09, right=.98, wspace=.13)
    output = ROOT / 'figures/PROBABILISTIC_SCORE_TRADEOFF.png'
    if output.exists():
        raise FileExistsError('Preserve the original figure')
    fig.savefig(output, dpi=170)
    plt.close(fig)
    receipt = {'actual_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'figure_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'matplotlib': matplotlib.__version__,
               'scope': 'Only two probability scores; RMSE tradeoffs remain in full tables.',
               'new_fits_or_scores_or_measurements': 0}
    with output.with_suffix('.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    print('Saved aggregate-only score figure; no new material calculation')


if __name__ == '__main__':
    main()
