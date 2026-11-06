"""Registered metadata-only admission selector; no native gzip/response/model access."""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('Duplicate JSON key: '+k)
            d[k]=v
        return d
    def invalid(v):raise ValueError('Nonstandard JSON constant: '+v)
    return json.loads(Path(p).read_text('utf-8'),object_pairs_hook=unique,parse_constant=invalid)
def put(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(x,msg):
    if not x:raise AssertionError(msg)

def configuration():
    return {'dataset':'jarvis_dft_3d','hash_prefix':'material_feedback_admission_v1|',
            'group_order':'SHA256(prefix+component_id) ascending hexadecimal digest, then component_id ascending',
            'exclude_all_old_formula_components':40,'expected_unused_components_before_selection':8802,
            'selected_components':256,'fit_components':64,'diagnostic_components':192,
            'role_order':'first64fit,next192diagnostic; all256development, no confirmation',
            'representative_rule':'smallest integer raw_row_zero_based among eligible metadata rows in each component',
            'all_component_siblings_reserved_development':True,'expected_remaining_components':8546,
            'statistical_admission_candidate':'full NIW; old star contract remains an unadopted candidate',
            'actual_statistical_fit_or_response_read':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--output-dir',required=True)
    args=ap.parse_args();plan=load(args.plan);approval=load(args.approval);out=Path(args.output_dir)
    require(plan['kind']=='MATERIAL_FEEDBACK_METADATA_ADMISSION_SELECTION' and plan['status']=='REGISTERED','Actual registered metadata-selection plan required')
    stamp=datetime.datetime.fromisoformat(plan['registered_actual_utc']);require(stamp.tzinfo is not None,'Registered UTC must be timezone aware')
    require(plan['source_sha256']==sha(__file__) and plan['configuration']==configuration(),'Frozen selector/config changed')
    require(approval.get('approved') is True and approval['scope']=='METADATA_ONLY_MATERIAL_FEEDBACK_ADMISSION_SELECTION','Exact metadata-only approval required')
    require(approval['source_sha256']==sha(__file__) and approval['plan_sha256']==sha(args.plan),'Approval source/plan binding mismatch')
    require(not out.exists(),'Preserve existing metadata outputs; no automatic rerun')
    for path,pin in plan['stable_input_sha256'].items():
        require(not path.lower().endswith(('.gz','.npy','.npz')),'Native numeric asset cannot be a read input')
        require(sha(path)==pin,'Stable input changed: '+path)
    require(sha(plan['contract_file'])==plan['contract_sha256'],'Source task contract changed')
    out.mkdir(parents=True)
    put(out/'METADATA_SELECTION_START.json',{'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'approval_sha256':sha(args.approval),'stable_input_sha256':plan['stable_input_sha256'],'administrative_read_time_provenance_only':plan['administrative_read_time_provenance_sha256'],'raw_asset_not_opened':True})
    try:
        meta=load(plan['eligible_metadata_file'])['datasets']['jarvis_dft_3d']
        previous=load(plan['old_role_file'])['datasets']['jarvis_dft_3d']
        known_fields={'artifact_native_index','component_id','composition_amounts','jid','mpid','normalized_composition','raw_row_zero_based'}
        groups=collections.defaultdict(list)
        seen_rows=set();seen_jids=set()
        for row in meta:
            require(set(row)==known_fields,'Unexpected metadata keys; no response fields allowed')
            require(type(row['raw_row_zero_based']) is int,'Integer native row locator required')
            require(row['raw_row_zero_based'] not in seen_rows and row['jid'] not in seen_jids,'Duplicate original row/JID')
            seen_rows.add(row['raw_row_zero_based']);seen_jids.add(row['jid']);groups[row['component_id']].append(row)
        old={r['component_id'] for r in previous['rows']};unused=set(previous['unused_eligible_group_ids'])
        require(len(old)==40 and len(unused)==8802 and not old&unused,'Old consumed/unused identity scope changed')
        require(set(groups)==old|unused,'Metadata group inventory differs from prior role ledger')
        ordered=sorted(unused,key=lambda c:(hashlib.sha256((configuration()['hash_prefix']+c).encode('utf-8')).hexdigest(),c))
        selected=ordered[:256];remaining=ordered[256:]
        require(len(remaining)==8546,'Unexpected remaining identity count')
        representatives={'fit':[],'diagnostic':[]};selected_groups=[]
        for rank,component in enumerate(selected):
            role='fit' if rank<64 else 'diagnostic'
            siblings=sorted(groups[component],key=lambda r:r['raw_row_zero_based'])
            representative=dict(siblings[0],role=role)
            representatives[role].append(representative)
            selected_groups.append({'component_id':component,'rank_zero_based':rank,'role':role,'representative':representative,
                'eligible_siblings':siblings,'eligible_sibling_count':len(siblings),
                'whole_source_component_consumed_for_future_confirmation':True,
                'all_original_source_siblings_enumerated':False,
                'ineligible_source_siblings':'Not enumerated by current eligible-only metadata. Still excluded by entire full-source component ID. Trusted extraction may later enumerate identity-only membership without changing selection.'})
        require(len(representatives['fit'])==64 and len(representatives['diagnostic'])==192,'Role counts changed')
        output={'artifact_kind':'METADATA_ONLY_ADMISSION_ROLE_SELECTION','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'registered_plan_sha256':sha(args.plan),'selector_source_sha256':sha(__file__),'stable_input_sha256':plan['stable_input_sha256'],
            'administrative_read_time_provenance_not_runtime_guards':plan['administrative_read_time_provenance_sha256'],
            'configuration':configuration(),'raw_asset_reference_not_read':plan['raw_asset_reference_not_read'],
            'representatives':representatives,'selected_groups':selected_groups,'old_excluded_component_ids':sorted(old),
            'new_development_consumed_component_ids':selected,'remaining_unselected_component_ids_in_registered_hash_order':remaining,
            'complete_hash_order_sha256':hashlib.sha256(json.dumps(ordered,separators=(',',':')).encode()).hexdigest(),
            'group_count':{'old_excluded':40,'fit':64,'diagnostic':192,'selected_development':256,'remaining_unselected':8546},
            'response_exposure':{'fit_property_values_read':0,'diagnostic_proxy_values_read':0,'diagnostic_target_values_read':0,'future_unselected_property_values_read':0,'raw_gzip_physically_parsed':False},
            'interpretation':'Metadata selection only. All256groups reserved development and all siblings excluded from later independent confirmation; no property learning, no independent validation. Roles authorize no native-response extraction by themselves.'}
        put(out/'ADMISSION_ROLE_SELECTION.json',output)
        for path,pin in plan['stable_input_sha256'].items():require(sha(path)==pin,'Stable source changed during metadata selection')
        require(sha(plan['contract_file'])==plan['contract_sha256'],'Contract changed during metadata selection')
        put(out/'METADATA_SELECTION_COMPLETE.json',{'status':'COMPLETE_METADATA_SELECTION_NO_PROPERTY_ACCESS','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role_file_sha256':sha(out/'ADMISSION_ROLE_SELECTION.json'),'selector_source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'selected_development_components':256,'representative_rows':256,'known_eligible_sibling_rows_reserved':sum(g['eligible_sibling_count'] for g in selected_groups),'all256_whole_components_reserved_including_unlisted_ineligible_siblings':True,'numeric_fit_or_score_calls':0,'raw_gzip_or_new_property_reads':0})
        print(json.dumps({'status':'COMPLETE_METADATA_SELECTION','role_sha256':sha(out/'ADMISSION_ROLE_SELECTION.json'),'fit':64,'diagnostic':192,'remaining_groups':8546,'property_reads':0},ensure_ascii=True))
    except Exception:
        put(out/'FAILED_STOP.json',{'status':'FAIL_METADATA_SELECTION_PRESERVED_NO_RETRY','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'traceback':traceback.format_exc(),'native_property_reads':0});raise

if __name__=='__main__':main()
