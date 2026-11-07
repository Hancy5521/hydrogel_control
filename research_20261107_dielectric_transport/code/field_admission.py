"""One registered three-field finite/positive admission, then metadata-only split.

No numeric property values or distribution statistics leave this phase. The
finite/positive target indicator IS observed and declared, not called target
blind. One previously fixed representative is used; no sibling substitution.
"""
import argparse
import hashlib
from pathlib import Path
import task_io as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    plan = io.validate(args.plan, args.approval, 'field_admission')
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=False)
    io.write(folder / 'START.json', {'actual_utc': io.utc(), 'plan_sha256': io.sha(args.plan),
                                    'source_sha256': io.sources(), 'property_values_emitted': 0})
    try:
        metadata = io.read(plan['inputs']['identity_metadata']['file'])
        summary = io.read(plan['inputs']['identity_summary']['file'])
        if (len(metadata['groups']) != 964 or summary['candidate_components_not_target_filtered'] != 851
                or summary['consumed_components'] != 1408 or summary['unknown_identity_components'] != 0):
            raise ValueError('expected complete identity admission required')
        expected = set(summary['candidate_component_ids'])
        groups = [g for g in metadata['groups'] if g['component_id'] in expected]
        if len(groups) != 851 or any(g['status'] != 'CANDIDATE_IDENTITY_ONLY_NOT_TARGET_FILTERED' for g in groups):
            raise ValueError('exact851 identity candidate components required')
        payload, columns = io.table(plan['inputs']['raw_source']['file'])
        flags, eligible = [], []
        for group in groups:
            identity = dict(group['representative'], component_id=group['component_id'])
            # Representative was frozen before properties; never choose a sibling
            # based on labels, values, quality, finiteness or desired performance.
            values = io.values_at(payload, columns, identity)
            accept = io.admitted(values)
            flags.append({'component_id': identity['component_id'], 'representative': identity,
                          'registered_three_field_predicate_pass': accept,
                          'no_target_magnitude_rank_or_decomposition_filter': True})
            if accept: eligible.append(identity)
        ordered = sorted(eligible, key=lambda r: (
            hashlib.sha256(('dielectric_transport_v1|' + r['component_id']).encode()).hexdigest(), r['component_id']))
        if len(ordered) < 448:
            io.write(folder / 'INSUFFICIENT_ELIGIBLE_STOP.json', {'actual_utc': io.utc(),
                     'eligible_components': len(ordered), 'required': 448, 'no_smaller_posthoc_experiment': True})
            raise ValueError('fewer than448 predeclared eligible groups; stop without fitting')
        representatives = {}
        offset = 0
        for role, size in io.ROLE_SIZES.items():
            representatives[role] = ordered[offset:offset + size]
            offset += size
        io.write(folder / 'ELIGIBILITY_FLAGS.json', {'records': flags, 'target_positivity_finiteness_observed': True,
                 'numeric_property_values_emitted': 0, 'fit_calls': 0, 'target_summary_statistics': 0})
        io.write(folder / 'ROLES.json', {'actual_utc': io.utc(), 'role_sizes': io.ROLE_SIZES,
                 'representatives': representatives, 'all_selected_component_ids': [r['component_id'] for r in ordered[:448]],
                 'remaining_unselected_component_ids': [r['component_id'] for r in ordered[448:]],
                 'candidate_components': 851, 'eligible_components': len(ordered),
                 'excluded_old_overlap_components': 113, 'hash_prefix': 'dielectric_transport_v1|',
                 'representative_chosen_before_property_admission': True,
                 'source_task_retraining_not_frozen_old_model_zero_shot': True})
        pins = {p.name: io.sha(p) for p in folder.glob('*.json')}
        io.write(folder / 'COMPLETE.json', {'status': 'ADMISSION_COMPLETE_NO_FIT_NO_PROPERTY_VALUES_EMITTED',
                 'actual_utc': io.utc(), 'plan_sha256': io.sha(args.plan), 'source_sha256': io.sources(),
                 'files_sha256': pins, 'eligible_components': len(ordered), 'selected_components': 448,
                 'fit_calls': 0, 'full_JSON_in_trusted_reader': True,
                 'target_positive_finite_indicator_inspected': True, 'numeric_property_values_emitted': 0})
        print({'status': 'ADMISSION_COMPLETE', 'eligible': len(ordered), 'selected': 448,
               'remaining': len(ordered) - 448, 'numeric_property_values_emitted': 0})
    except Exception as problem:
        io.write(folder / 'FAILED_STOP.json', {'actual_utc': io.utc(), 'type': type(problem).__name__,
                 'message': str(problem), 'source_sha256': io.sources(), 'no_posthoc_rescue': True})
        raise


if __name__ == '__main__': main()
