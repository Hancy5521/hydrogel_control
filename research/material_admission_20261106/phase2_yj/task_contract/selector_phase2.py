"""Phase2 fixed-power metadata selection; no native values, source parser or model."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,hashlib,json,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def put(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def need(x,m):
    if not x:raise AssertionError(m)

def configuration():
    return {'version':'fixed_YJ_power_phase2_metadata_v1','dataset':'jarvis_dft_3d',
        'hash_prefix':'material_feedback_fixed_power_v1|','tie_break':'component_id ascending after SHA256 hexadecimal digest',
        'legacy_excluded_components':40,'phase1_development_excluded_components':256,'total_prior_excluded_components':296,
        'expected_remaining_before_selection':8546,'selected_development_components':256,
        'fit_components':64,'diagnostic_components':192,'remaining_after_selection':8290,
        'role_order':'first64fit,next192diagnostic;all256development',
        'representative_rule':'smallest eligible integer raw_row_zero_based within fullsource component',
        'whole_component_and_all_siblings_reserved':True,'fixed_transform_hypothesis':'Yeo-Johnson lambda0.5, all-real invertible; no parameter estimated by selector',
        'expected_followup_scope_only':'5lawroles x4masks x192diagnostic=3840forecasts; two pipelinefits need separate scientific approval',
        'actual_native_response_access_or_fit':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);args=p.parse_args()
    plan=read(args.plan);a=read(args.approval);out=Path(args.output_dir)
    need(plan['kind']=='PHASE2_FIXED_POWER_METADATA_SELECTION' and plan['status']=='REGISTERED','Actual registered metadata-only phase2 plan required')
    need(datetime.fromisoformat(plan['registered_actual_utc']).tzinfo is not None,'Timezone-aware registration required')
    need(plan['configuration']==configuration() and plan['source_sha256']==sha(__file__),'Selector/configuration changed')
    need(a.get('approved') is True and a['scope']=='METADATA_ONLY_FIXED_YJ_PHASE2_SELECTION' and a['source_sha256']==sha(__file__) and a['plan_sha256']==sha(args.plan),'Exact metadata-only phase2 approval required')
    need(not out.exists(),'Preserve previous selection output; no automatic repeat')
    need(len(plan['stable_input_sha256'])==16,'Exactly16stable provenance/source inputs expected')
    for path,pin in plan['stable_input_sha256'].items():
        need(not path.lower().endswith(('.gz','.npz','.npy')) and sha(path)==pin,'Stable input changed or forbidden native asset')
    out.mkdir(parents=True)
    put(out/'METADATA_SELECTION_START.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'approval_sha256':sha(args.approval),'raw_gzip_opened':False,'phase1_NO_GO_result_hashed_not_decoded':True})
    try:
        meta=read(plan['eligible_metadata_file'])['datasets']['jarvis_dft_3d'];legacy=read(plan['old_role_file'])['datasets']['jarvis_dft_3d'];phase1=read(plan['phase1_role_file'])
        allowed={'artifact_native_index','component_id','composition_amounts','jid','mpid','normalized_composition','raw_row_zero_based'}
        groups=collections.defaultdict(list)
        for r in meta:
            need(set(r)==allowed and type(r['raw_row_zero_based']) is int,'Metadata schema changed; native property fields forbidden');groups[r['component_id']].append(r)
        old={r['component_id'] for r in legacy['rows']};used1=set(phase1['new_development_consumed_component_ids']);remaining1=set(phase1['remaining_unselected_component_ids_in_registered_hash_order'])
        need(len(old)==40 and len(used1)==256 and not old&used1,'Legacy/phase1 wholecomponent exclusion changed')
        need(len(remaining1)==8546 and not remaining1&(old|used1),'Phase1 remaining component inventory changed')
        need(set(groups)==old|used1|remaining1 and set(legacy['unused_eligible_group_ids'])==used1|remaining1,'Full metadata inventory mismatch')
        ordered=sorted(remaining1,key=lambda c:(hashlib.sha256(('material_feedback_fixed_power_v1|'+c).encode()).hexdigest(),c));chosen=ordered[:256];remaining=ordered[256:]
        need(len(remaining)==8290,'Remaining component count differs')
        reps={'fit':[],'diagnostic':[]};selected=[]
        for rank,c in enumerate(chosen):
            role='fit' if rank<64 else 'diagnostic';siblings=sorted(groups[c],key=lambda r:r['raw_row_zero_based']);representative=dict(siblings[0],role=role);reps[role].append(representative)
            selected.append({'component_id':c,'rank_zero_based':rank,'role':role,'representative':representative,'eligible_siblings':siblings,'eligible_sibling_count':len(siblings),'whole_source_component_consumed_for_future_confirmation':True,'all_original_source_siblings_enumerated':False,'ineligible_sibling_scope':'Not listed by current eligible-only metadata; entireoriginalsourcecomponent reserved nonetheless.'})
        need(len({r['component_id'] for rs in reps.values() for r in rs})==256,'Roles not disjoint')
        result={'artifact_kind':'PHASE2_METADATA_ONLY_ADMISSION_ROLE_SELECTION','actual_utc':datetime.now(timezone.utc).isoformat(),'registered_plan_sha256':sha(args.plan),'selector_source_sha256':sha(__file__),'stable_input_sha256':plan['stable_input_sha256'],'administrative_read_time_provenance_only':plan['administrative_read_time_provenance_sha256'],'configuration':configuration(),'raw_asset_reference_not_read':plan['raw_asset_reference_not_read'],
            'phase1_role_sha256':plan['stable_input_sha256'][plan['phase1_role_file']],'phase1_NO_GO_decision_reference_hashed_not_decoded':plan['phase1_NO_GO_decision_reference'],
            'representatives':reps,'selected_groups':selected,'legacy40_excluded_component_ids':sorted(old),'phase1_development256_excluded_component_ids':sorted(used1),'old_excluded_component_ids':sorted(old|used1),'new_development_consumed_component_ids':chosen,'remaining_unselected_component_ids_in_registered_hash_order':remaining,
            'complete_hash_order_sha256':hashlib.sha256(json.dumps(ordered,separators=(',',':')).encode()).hexdigest(),
            'group_count':{'legacy_excluded':40,'phase1_development_excluded':256,'all_prior_excluded':296,'fit':64,'diagnostic':192,'selected_development':256,'remaining_unselected':8290},
            'response_exposure':{'fit_property_values_read':0,'diagnostic_proxy_values_read':0,'diagnostic_target_values_read':0,'future_unselected_property_values_read':0,'raw_gzip_physically_parsed':False},
            'interpretation':'Genuinely new metadata groups after phase1failure, but allphase2groups reserved development. Not independent confirmation of a tuned transformation; no label or statistical execution authorized by metadata selection.'}
        put(out/'ADMISSION_ROLE_SELECTION.json',result)
        for path,pin in plan['stable_input_sha256'].items():need(sha(path)==pin,'Stable input changed during selection')
        put(out/'METADATA_SELECTION_COMPLETE.json',{'status':'COMPLETE_PHASE2_METADATA_SELECTION_NO_PROPERTY_ACCESS','actual_utc':datetime.now(timezone.utc).isoformat(),'role_file_sha256':sha(out/'ADMISSION_ROLE_SELECTION.json'),'selector_source_sha256':sha(__file__),'registered_plan_sha256':sha(args.plan),'fit':64,'diagnostic':192,'prior_excluded_components':296,'remaining':8290,'known_eligible_sibling_rows_reserved':sum(x['eligible_sibling_count'] for x in selected),'new_native_property_reads_fit_or_scores':0})
        print(json.dumps({'status':'COMPLETE_PHASE2_METADATA_ONLY','role_sha256':sha(out/'ADMISSION_ROLE_SELECTION.json'),'fit':64,'diagnostic':192,'remaining':8290,'newpropertyreads':0},ensure_ascii=True))
    except Exception:
        put(out/'FAILED_STOP.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL_PRESERVED_NO_RETRY','traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
