"""Metadata only: no raw gzip, response columns, SUT import, split selection or fit."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

OLD=Path(r'H:\polymer_completion_public\research\hypercycle_agents_20261105T093229687251Z')
OUT=Path(__file__).resolve().parent

def digest(b):
    return hashlib.sha256(b).hexdigest()

def chemical_system(row):
    if 'composition_amounts' in row:
        return tuple(sorted(row['composition_amounts']))
    return tuple(sorted(x.split(':',1)[0] for x in row['normalized_composition'].split('|')))

def count_groups(rows):
    return len({r['component_id'] for r in rows})

def summary(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['component_id']].append(r)
    systems={chemical_system(r) for r in rows}
    hist=collections.Counter(len(chemical_system(v[0])) for v in groups.values())
    d={
        'rows':len(rows),'full_source_component_groups':len(groups),
        'normalized_compositions':len({r['normalized_composition'] for r in rows}),
        'chemical_element_sets':len(systems),
        'element_count_per_group_histogram':dict(sorted(hist.items())),
        'largest_component_row_counts':sorted([len(v) for v in groups.values()],reverse=True)[:12],
        'groups_with_multiple_eligible_rows':sum(len(v)>1 for v in groups.values()),
        'groups_with_at_least_two_eligible_rows':sum(len(v)>=2 for v in groups.values()),
        'groups_with_at_least_three_eligible_rows':sum(len(v)>=3 for v in groups.values()),
        'original_component_id_set_sha256':digest(json.dumps(sorted(groups),separators=(',',':')).encode()),
    }
    if rows and 'source_citation' in rows[0]:
        temps={r['temperature_K'] for r in rows}
        d.update({
            'source_citation_count':len({r['source_citation'] for r in rows}),
            'distinct_temperature_count':len(temps),
            'temperature_K_min_max_metadata_only':[min(temps),max(temps)],
            'groups_with_multiple_temperatures':sum(len({r['temperature_K'] for r in v})>1 for v in groups.values()),
            'groups_with_multiple_compositions':sum(len({r['normalized_composition'] for r in v})>1 for v in groups.values()),
        })
    return d

def main():
    run=json.loads((OUT/'RUN.json').read_text('utf-8'))
    for rel,expected in run['authorized_inputs_sha256'].items():
        assert digest((OLD/rel).read_bytes())==expected
    meta=json.loads((OLD/'data/ELIGIBLE_METADATA.json').read_text('utf-8'))
    roles=json.loads((OLD/'pilot/DATASET_ROLE_PLAN.json').read_text('utf-8'))
    result={
        'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'actual_asia_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'artifact_kind':'UNUSED_IDENTITY_METADATA_FEASIBILITY_NOT_A_NEW_SPLIT',
        'script_sha256':digest(Path(__file__).read_bytes()),'input_sha256':run['authorized_inputs_sha256'],
        'read_scope':'Only original metadata, source summaries, manifest and prior role metadata; no raw gzip, queried response receipt or saved scores opened.',
        'datasets':{},'new_split_selected':False,'new_role_or_seed_selected':False,
        'model_or_policy_executed':False,'response_arrays_or_native_property_values_read':False,
        'query_oracle_budget_consumed':0,
    }
    for ds,rows in meta['datasets'].items():
        old=roles['datasets'][ds]
        selected_ids={r['component_id'] for r in old['rows']}
        selected_row_ids={r['raw_row_zero_based'] for r in old['rows']}
        unused_ids=set(old['unused_eligible_group_ids'])
        assert len(selected_ids)==40 and not selected_ids&unused_ids
        assert {r['component_id'] for r in rows}==selected_ids|unused_ids
        fresh=[r for r in rows if r['component_id'] in unused_ids]
        tainted=[r for r in rows if r['component_id'] in selected_ids]
        selected=[r for r in rows if r['raw_row_zero_based'] in selected_row_ids]
        not_selected_but_consumed_group=[r for r in tainted if r['raw_row_zero_based'] not in selected_row_ids]
        fresh_sys={chemical_system(r) for r in fresh}
        consumed_sys={chemical_system(r) for r in tainted}
        fresh_elements={e for s in fresh_sys for e in s}
        consumed_elements={e for s in consumed_sys for e in s}
        fresh_compositions={r['normalized_composition'] for r in fresh}
        consumed_compositions={r['normalized_composition'] for r in tainted}
        assert not fresh_compositions&consumed_compositions
        info={
            'originally_complete_3_property_eligible':summary(rows),
            'first_pilot_selected_40_full_source_groups_all_off_limits_for_new_confirmation':summary(tainted),
            'actual_selected_rows_in_prior_pilot':len(selected),
            'unselected_rows_in_consumed_groups_NOT_fresh':len(not_selected_but_consumed_group),
            'genuinely_unselected_identity_groups':summary(fresh),
            'metadata_breadth_not_label_efficacy':{
                'fresh_chemical_systems_also_in_consumed_groups':len(fresh_sys&consumed_sys),
                'fresh_chemical_systems_unseen_in_consumed_groups':len(fresh_sys-consumed_sys),
                'fresh_groups_in_previously_seen_chemical_systems':count_groups([r for r in fresh if chemical_system(r) in consumed_sys]),
                'fresh_groups_in_previously_unseen_chemical_systems':count_groups([r for r in fresh if chemical_system(r) not in consumed_sys]),
                'fresh_element_vocab_size':len(fresh_elements),
                'prior_consumed_group_element_vocab_size':len(consumed_elements),
                'fresh_elements_not_in_consumed_groups':sorted(fresh_elements-consumed_elements),
                'normalized_composition_overlap_with_consumed':0,
            },
            'target_field_names_only':old['target_fields'],'target_units_from_pinned_metadata':old['target_units'],
            'provenance':old['source_provenance'],
            'independence_limit':'Unselected identity/component groups in one previously curated table; not new laboratories, preparation batches, DFT software versions or independent external data.',
        }
        if ds=='ucsb_thermoelectrics':
            assert not {r['source_citation'] for r in fresh}&{r['source_citation'] for r in tainted}
            info['source_citation_overlap_with_consumed']=0
        result['datasets'][ds]=info
    out=OUT/'UNUSED_IDENTITY_FEASIBILITY.json'
    with out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=True,indent=2);f.write('\n')
    print(json.dumps({'receipt':str(out),'sha256':digest(out.read_bytes()),'datasets':{k:v['genuinely_unselected_identity_groups'] for k,v in result['datasets'].items()}},ensure_ascii=True))

if __name__=='__main__':
    main()
