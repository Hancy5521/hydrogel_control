"""Create a new actual-time source-pinned audit plan; no SUT/model imports.

This creator reads protocol and freeze metadata only. For scores it hashes the
producer result without parsing it. Audit execution still requires a separate
root approval binding this plan SHA and phase.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for blob in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(blob)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('forecasts', 'scores'))
    parser.add_argument('--scientific-plan', type=Path, required=True)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--source-directory', type=Path, default=HERE.parent / 'code')
    parser.add_argument('--forecast-audit-result', type=Path)
    parser.add_argument('--score-result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scientific = read(args.scientific_plan)
    freeze_file = args.campaign_dir / 'ALL_FORECASTS_FROZEN.json'
    freeze = read(freeze_file)
    assert freeze['plan_sha256'] == sha(args.scientific_plan)
    for name, digest in scientific['source_sha256'].items():
        assert sha(args.source_directory / name) == digest
    now = datetime.now(timezone.utc)
    plan = {
        'artifact_kind': 'Independent postfit offline numerical audit actual-time registration',
        'actual_registered_utc': now.isoformat(),
        'actual_registered_asia_shanghai': now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'phase': args.phase,
        'audit_script_file': str(HERE / 'independent_numerical_audit.py'),
        'audit_script_sha256': sha(HERE / 'independent_numerical_audit.py'),
        'plan_creator_sha256': sha(__file__),
        'scientific_plan_file': str(args.scientific_plan.resolve()),
        'scientific_plan_sha256': sha(args.scientific_plan),
        'source_directory': str(args.source_directory.resolve()),
        'source_sha256': scientific['source_sha256'],
        'campaign_directory': str(args.campaign_dir.resolve()),
        'forecast_freeze_sha256': sha(freeze_file),
        'phase_requires_root_approval_before_execution': True,
        'prohibited_activity': 'No SUT imports, oracle/policy runs, refits, active learning retuning, new material labels or new-data claims.',
        'tolerances': {
            'metadata': {'atol': 2e-10, 'rtol': 2e-10},
            'mean': {'atol': 2e-8, 'rtol': 2e-8},
            'covariance': {'atol': 5e-9, 'rtol': 2e-8},
            'scores': {'atol': 5e-11, 'rtol': 2e-10},
        },
        'fixed_expected_scope': {
            'actual_campaigns': 40, 'logical_episode_aliases_including24reuses': 64,
            'forecast_laws': 160, 'bound_original_forecast_payload_files': 281,
            'campaign_bundle_budget_each': 24, 'equal_test_metadata_groups_each_dataset': 8,
            'datasets': ['ucsb_thermoelectrics', 'jarvis_dft_3d'],
        },
        'independent_reconstruction': 'RMS metadata fractions/T; warm-only mean/sampleSD/taskcov.8samplecov+.2I, RBF scalar length1, response residual.05I; one direct dense conditioning on all legal observations at each checkpoint, not producer sequential update.',
        'scores_oracle': 'Only after score-specific approval and frozen producer evaluation: original source three-responses; math.erf GaussianCRPS, equal8group mean, independently interpolated16group decile frontier threshold, exact cost/cell/R-E/alias reconciliation.',
        'no_test_truth_before_score_approval': True,
        'prior_audit_execution': 'Draft source/AST static review only before this plan. Original bounded pilot sources/results are not rerun.',
        'failure_policy': 'First failure is written, execution stops. Preserve all receipts and register a new plan for a legitimate correction; no producer retuning on inspected outcomes.',
        'claim_limit': 'Numerical correctness for a fixed offline pilot, not independent material efficacy or learned causal catalysis/top-conference readiness.',
    }
    if args.phase == 'scores':
        if args.forecast_audit_result is None or args.score_result is None:
            raise ValueError('Score phase needs frozen producer score and passed forecast audit paths')
        forecast_pass = read(args.forecast_audit_result)
        assert forecast_pass['status'] == 'PASS' and forecast_pass['phase'] == 'forecasts'
        assert forecast_pass['scientific_plan_sha256'] == plan['scientific_plan_sha256']
        assert forecast_pass['forecast_freeze_sha256'] == plan['forecast_freeze_sha256']
        plan.update({'forecast_audit_result_file': str(args.forecast_audit_result.resolve()),
                     'forecast_audit_result_sha256': sha(args.forecast_audit_result),
                     'score_result_file': str(args.score_result.resolve()),
                     'score_result_sha256': sha(args.score_result)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PLAN_WRITTEN_NOT_EXECUTED', 'phase': args.phase,
                      'plan_file': str(args.output.resolve()), 'plan_sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
