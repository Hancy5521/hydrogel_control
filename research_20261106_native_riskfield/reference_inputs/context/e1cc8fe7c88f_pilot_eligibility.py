"""Requested metadata/finite-mask pilot feasibility; no label ranges or scores.

Group components use the entire source before eligible rows are selected, so
removing an ineligible record never severs a previously declared identity link.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from audit_schemas import clean_formula, finite

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def linked_groups(rows, keys, namespace):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in keys:
            value = row.get(key)
            if value is None or not str(value).strip() or str(value).lower() in ('na', 'none', 'nan'):
                continue
            token = key, value
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    identifiers = {}
    for indices in members.values():
        ident = namespace + hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()[:20]
        for i in indices:
            identifiers[i] = ident
    return identifiers


def main():
    plan = {'actual_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
            'dependency_sha256': sha(ROOT / 'audit_schemas.py'),
            'inputs_sha256': {name: sha(ROOT / (name + '.json.gz')) for name in ('ucsb_thermoelectrics', 'jarvis_dft_3d')},
            'requested_checks': 'UCSB rho/S/kappa finite+T finite+nonempty src; JARVIS e_form/gapopt/bulk finite+normalizedformula. Retain global-source groups. No response ranges, split or model.'}
    dump(ROOT / 'PILOT_ELIGIBILITY_PLAN.json', plan)
    summaries, metadata = {}, {}
    for name in ('ucsb_thermoelectrics', 'jarvis_dft_3d'):
        with gzip.open(ROOT / (name + '.json.gz'), 'rt', encoding='utf-8') as stream:
            raw = json.load(stream)
        rows = [dict(zip(raw['columns'], row), normalized_composition=clean_formula(row[raw['columns'].index('composition')])) for row in raw['data']]
        keys = ['src', 'normalized_composition'] if name == 'ucsb_thermoelectrics' else ['jid', 'mpid', 'normalized_composition']
        ids = linked_groups(rows, keys, name + '_component_')
        targets = ['rho (ohm.cm)', 'S [muV/K]', 'kappa [W/mK]'] if name == 'ucsb_thermoelectrics' else ['e_form', 'gap opt', 'bulk modulus']
        admitted = []
        for i, row in enumerate(rows):
            if not all(finite(row[key]) for key in targets) or row['normalized_composition'] is None:
                continue
            if name == 'ucsb_thermoelectrics' and (not finite(row['T [K]']) or not isinstance(row['src'], str) or not row['src'].strip()):
                continue
            base = {'raw_row_zero_based': i, 'artifact_native_index': raw['index'][i], 'component_id': ids[i],
                    'normalized_composition': row['normalized_composition']}
            if name == 'ucsb_thermoelectrics':
                base.update({'composition': row['composition'], 'source_citation': row['src'], 'temperature_K': row['T [K]'],
                             'crystallinity': row['crystallinity'], 'synthesis': row['synthesis']})
            else:
                base.update({'jid': row['jid'], 'mpid': row['mpid'],
                             'composition_amounts': {key: val for key, val in row['composition'].items() if not key.startswith('@')}})
            admitted.append(base)
        counts = Counter(x['component_id'] for x in admitted)
        summaries[name] = {
            'eligible_rows': len(admitted), 'eligible_components_from_full_source_graph': len(counts),
            'full_source_component_count': len(set(ids.values())), 'identity_links_recomputed_on_eligible_subset': False,
            'candidate_target_fields': targets, 'eligibility_rule': 'Finite original targets; normalized simple composition; UCSB additionally finiteT and nonempty source.',
            'largest_eligible_component_row_counts': sorted(counts.values(), reverse=True)[:10],
            'eligible_group_size_histogram': dict(sorted(Counter(counts.values()).items())),
            'rows_if_at_most_two_per_group': sum(min(n, 2) for n in counts.values()),
            'group_count_at_least40': len(counts) >= 40,
            'row_selection_and_role_split_performed': False,
            'values_exposed_in_metadata': 'Condition/identity features only; none of the candidate response values or derived output fields.',
        }
        metadata[name] = admitted
    dump(ROOT / 'ELIGIBLE_METADATA.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
             'source_role': 'Metadata candidate pool only, not a selected experimental split.', 'datasets': metadata})
    dump(ROOT / 'PILOT_ELIGIBILITY.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
             'plan_sha256': sha(ROOT / 'PILOT_ELIGIBILITY_PLAN.json'), 'metadata_sha256': sha(ROOT / 'ELIGIBLE_METADATA.json'),
             'datasets': summaries, 'model_fits': 0, 'target_prediction_scores': 0,
             'group_count_limit': 'Components are conservative declared metadata linkage; not certified independent preparations or units.'})
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    main()
