"""Model-free schema/identity/availability audit of two pinned public tables.

It loads original labels for finite/missing counts only, never fits, scores a
predictor, scans candidate algorithms or selects a split from performance.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
FILES = {
    'ucsb_thermoelectrics': '7d0c9c74fd52995b876bd251a7732b5498314702441fa463197fb492be68ecc0',
    'jarvis_dft_3d': '3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def clean_formula(value):
    """Exact ratio for simple element-count formulas only; refuse complex syntax."""
    if isinstance(value, dict):
        pairs = [(key, val) for key, val in value.items() if not key.startswith('@')]
    elif isinstance(value, str):
        text = value.replace(' ', '')
        pattern = re.compile(r'([A-Z][a-z]?)([0-9]+(?:\.[0-9]*)?|\.[0-9]+)?')
        tokens = list(pattern.finditer(text))
        if ''.join(tok.group(0) for tok in tokens) != text or not tokens:
            return None
        pairs = [(tok.group(1), tok.group(2) or 1) for tok in tokens]
    else:
        return None
    values = {}
    try:
        for key, val in pairs:
            amount = Fraction(str(val))
            if amount < 0:
                return None
            if amount:
                values[key] = values.get(key, Fraction(0)) + amount
        total = sum(values.values())
        if total <= 0:
            return None
        return '|'.join(key + ':' + str(values[key] / total) for key in sorted(values))
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def components(rows, keys):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for name in keys:
            value = row.get(name)
            if value is None or not str(value).strip() or str(value).lower() in ('na', 'nan', 'none'):
                continue
            token = name, value
            if token in seen:
                left, right = find(i), find(seen[token])
                parent[left] = right
            else:
                seen[token] = i
    sizes = Counter(find(i) for i in range(len(rows)))
    return {'n_components': len(sizes), 'largest_component_row_counts': sorted(sizes.values(), reverse=True)[:10],
            'link_tokens': keys, 'meaning': 'Conservative metadata linkage only, not certified independent physical batches.'}


def main():
    plan = {'actual_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': FILES,
            'script_sha256': digest(__file__), 'scope': __doc__, 'model_fits': 0, 'target_prediction_scores': 0}
    for name, expected in FILES.items():
        assert digest(ROOT / (name + '.json.gz')) == expected
    dump(ROOT / 'SCHEMA_AUDIT_PLAN.json', plan)
    results = []
    for name, expected in FILES.items():
        with gzip.open(ROOT / (name + '.json.gz'), 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        columns, rows, index = payload['columns'], payload['data'], payload['index']
        assert len(rows) == len(index) and len(set(index)) == len(index)
        assert all(len(row) == len(columns) for row in rows)
        mappings = [dict(zip(columns, row)) for row in rows]
        counts = {key: sum(finite(row.get(key)) for row in mappings) for key in columns}
        targets = (['rho (ohm.cm)', 'S [muV/K]', 'kappa [W/mK]'] if name == 'ucsb_thermoelectrics'
                   else ['e_form', 'gap opt', 'bulk modulus', 'shear modulus'])
        availability = Counter(sum(finite(row.get(key)) for key in targets) for row in mappings)
        canonical = [clean_formula(row['composition']) for row in mappings]
        metadata_rows = [{**row, 'normalized_composition': formula} for row, formula in zip(mappings, canonical)]
        result = {'dataset': name, 'raw_sha256': expected, 'rows': len(rows), 'columns': columns,
                  'all_row_lengths_and_index_unique': True, 'index_role': 'Artifact-local row handle; not physical-unit certification',
                  'native_finite_numeric_count_by_column': counts, 'primary_candidate_target_fields': targets,
                  'target_availability_row_count_histogram': {str(k): v for k, v in sorted(availability.items())},
                  'target_count_roles': 'Availability only; no response ranges, predictor metrics or selected split inspected.',
                  'simple_formula_normalization_count': sum(x is not None for x in canonical),
                  'simple_formula_normalization_unique': len({x for x in canonical if x is not None}),
                  'normalization_limit': 'Exact simple element/count ratios; complex formulas refused, no surrogate values invented.'}
        if name == 'ucsb_thermoelectrics':
            source = [row['src'] for row in mappings]
            result.update({'nonempty_source_citation_count': sum(isinstance(x, str) and bool(x.strip()) for x in source),
                           'distinct_nonempty_source_citations': len({x for x in source if isinstance(x, str) and x.strip()}),
                           'crystallinity_states': dict(Counter(row['crystallinity'] for row in mappings)),
                           'source_only_components': components(metadata_rows, ['src']),
                           'source_and_normalized_formula_components': components(metadata_rows, ['src', 'normalized_composition']),
                           'native_missingness_present': any(counts[key] != len(rows) for key in targets),
                           'algebraic_dependence_from_official_metadata': ['sigma[S/cm] = 1/rho[ohm.cm]', 'PF = sigma*S^2 with unit conversion', 'zT = PF*T/kappa'],
                           'independent_candidate_targets': targets,
                           'derived_targets_excluded_as_independent_oracle_cells': ['sigma [S/cm]', 'PF [W/mK^2]', 'zT'],
                           'temperature_condition': 'T[K] is an experimental condition, not a target to hallucinate; same source/material at multiple T cannot become independent materials.',
                           'source_identity_gap': 'Native src is literature citation; no complete preparation/sample/measurement UUID ledger. Missing source rows require conservative handling specified before fitting.'})
        else:
            for field in ('jid', 'mpid'):
                vals = [row[field] for row in mappings]
                result[field + '_unique'] = len(set(vals))
                result[field + '_missing_or_sentinel'] = sum(not x or str(x).lower() in ('na', 'nan', 'none') for x in vals)
            result.update({'composition_and_native_ID_components': components(metadata_rows, ['jid', 'mpid', 'normalized_composition']),
                           'native_missingness_present': any(counts[key] != len(rows) for key in targets),
                           'computed_not_measured': True,
                           'same_run_dependencies': ['bulk/shear share original elastic calculations', 'dielectric x/y/z share a response tensor', 'gap opt and gap tbmbj are different functionals, not independent wet validation'],
                           'structure_input_cost_note': 'Known relaxed structures are a benchmark input assumption; their original DFT generation cost is not free in a real discovery campaign.',
                           'property_cell_budget_note': 'Equal revealed-cell counts are an offline abstraction. Elastic tensor bundle cost and real DFT walltime require separate accounting.'})
        results.append(result)
    dump(ROOT / 'SCHEMA_AND_IDENTITY_AUDIT.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': digest(ROOT / 'SCHEMA_AUDIT_PLAN.json'), 'datasets': results,
            'model_fit_calls': 0, 'target_prediction_score_calls': 0, 'new_measurements': 0,
            'old_MLST_data_opened': False})
    print(json.dumps({'schema_audit_complete': True, 'row_counts': {x['dataset']: x['rows'] for x in results},
                      'no_fitting_or_prediction_scoring': True}))


if __name__ == '__main__':
    main()
