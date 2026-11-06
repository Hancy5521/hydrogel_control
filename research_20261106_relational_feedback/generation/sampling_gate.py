"""Unique fixed synthetic joint-sampling run; primitive summaries are not validation.

Requires root actual-UTC/source/input approval. Never reads material values or fits
models. Independent audit loads the saved numeric arrays without importing sampler.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import joint_sampler

HERE=Path(__file__).resolve().parent
SOURCE_FILES=('joint_sampler.py','sampling_gate.py')
SAMPLER_SHA='5b77cde6f6356eb977596ea89564da04e95a0e57688414d142d40442cd44bfae'
PROPOSAL_SHA='7c9a99e2ab34a2b64e3ad579f02357791a7df4b09eb3e4ed6035fc77f62a4ca8'
FIXTURE_SHA='27f834b26bcfb03fb610aeb0d9acdd2a28159afe8396f8cbcba8780775dfdef9'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def value_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text('utf-8'))
    if plan.get('approved_for_execution') is not True or not plan.get('actual_registered_utc'):
        raise ValueError('root actual-UTC registered approval required')
    registered=datetime.fromisoformat(plan['actual_registered_utc'])
    if registered.tzinfo is None or registered.utcoffset().total_seconds()!=0 or registered>datetime.now(timezone.utc):
        raise ValueError('actual UTC registration must precede execution')
    actual_sources={name:sha(HERE/name) for name in SOURCE_FILES}
    if actual_sources!=plan['source_sha256'] or actual_sources['joint_sampler.py']!=SAMPLER_SHA:
        raise ValueError('fixed sampler/driver source pin mismatch')
    proposal_path=HERE/'SAMPLING_GATE_PROPOSAL.json'
    if sha(proposal_path)!=PROPOSAL_SHA or plan['proposal_sha256']!=PROPOSAL_SHA:
        raise ValueError('fixed pre-output sampling proposal required')
    proposal=json.loads(proposal_path.read_text('utf-8'));fixture_path=Path(plan['fixture']['file'])
    if (sha(fixture_path)!=FIXTURE_SHA or plan['fixture']['sha256']!=FIXTURE_SHA
            or proposal['input_fixture']['sha256']!=FIXTURE_SHA):raise ValueError('original artificial fixture pin mismatch')
    if plan['seed']!=20261106 or plan['n_draws']!=131072 or plan['numpy_version']!=np.__version__:
        raise ValueError('fixed seed/draw count/registered NumPy version required')
    if args.output.exists():raise FileExistsError('one execution output must be absent')
    args.output.mkdir(parents=True)
    started=utc();write(args.output/'EXECUTION_START.json',{'actual_utc':started,'plan_sha256':sha(args.plan),
        'proposal_sha256':PROPOSAL_SHA,'source_sha256':actual_sources,'fixture_sha256':FIXTURE_SHA,
        'python':platform.python_version(),'numpy':np.__version__,'scope':'OLD_SYNTHETIC_FIXTURE_ONLY_NO_MATERIAL'})
    fixture=json.loads(fixture_path.read_text('utf-8'));ensemble=copy.deepcopy(fixture['initial_ensemble'])
    p=fixture['fixed_fixture'];contexts=np.asarray([p['query_context'],p['target_context']],dtype=float)
    response=copy.deepcopy(p['response_transform'])
    before={'ensemble':ensemble,'context_z':contexts.tolist(),'response':response};before_sha=value_sha(before)
    write(args.output/'INPUT_BINDING.json',{'fixture_file':str(fixture_path),'fixture_sha256':FIXTURE_SHA,
        'proposal_sha256':PROPOSAL_SHA,'plan_sha256':sha(args.plan),'source_sha256':actual_sources,
        'seed':plan['seed'],'n_draws':plan['n_draws'],'actual_input':copy.deepcopy(before),
        'input_sha256_before':before_sha,'artificial_input_not_material':True})
    timer=time.perf_counter()
    generated=joint_sampler.sample_joint(ensemble,contexts,response,seed=plan['seed'],n_draws=plan['n_draws'],return_auxiliary=True)
    elapsed=time.perf_counter()-timer
    after_sha=value_sha({'ensemble':ensemble,'context_z':contexts.tolist(),'response':response})
    if after_sha!=before_sha:raise AssertionError('pure sampler changed input state/context/response')
    auxiliary=generated['auxiliary'];arrays={'native':generated['native_draws'],'z':auxiliary['modeled_z_draws'],
        'H':auxiliary['H_index_per_joint_draw'],'sigma2':auxiliary['sigma2_per_joint_draw']}
    component_index=[]
    for h,name in enumerate(joint_sampler.MODEL_IDS):
        data=auxiliary['coefficient_draws_grouped_by_H'][name]
        arrays['coef_h%d'%h]=data['coefficient_draws'];arrays['indices_h%d'%h]=data['draw_indices']
        component_index.append({'h':h,'id':name,'coefficients_key':'coef_h%d'%h,'draw_indices_key':'indices_h%d'%h,
            'coefficient_dimension':joint_sampler.DIMENSIONS[h],'draws':len(data['draw_indices'])})
    if any(array.dtype.hasobject for array in arrays.values()):raise AssertionError('object arrays prohibited')
    npz=args.output/'draws_and_aux.npz'
    with npz.open('xb') as f:np.savez_compressed(f,**arrays)
    index={key:{'shape':list(array.shape),'dtype':str(array.dtype)} for key,array in arrays.items()}
    metadata=generated['metadata']|{'actual_start_utc':started,'actual_end_utc':utc(),'sampling_elapsed_seconds':elapsed,
        'source_sha256':actual_sources,'fixture_sha256':FIXTURE_SHA,'proposal_sha256':PROPOSAL_SHA,
        'plan_sha256':sha(args.plan),'npz_file':'draws_and_aux.npz','npz_sha256':sha(npz),'arrays':index,
        'component_array_index':component_index,'np_load_allow_pickle_required':False,
        'all_saved_arrays_nonobject':True,'input_sha256_before':before_sha,'input_sha256_after':after_sha,
        'input_unchanged':True,'independent_validation_status':'NOT_PERFORMED_BY_PRODUCER'}
    write(args.output/'metadata.json',metadata)
    z=arrays['z'];native=arrays['native'];H=arrays['H'];points=np.asarray(proposal['fixed_checks']['modeled_CDF_points_per_row'])
    thresholds=response['s0']*joint_sampler._inverse_yj_half(response['center']+response['sd']*points)
    result={'status':'SAMPLES_GENERATED_PRIMITIVES_SAVED_NOT_INDEPENDENTLY_VALIDATED',
        'actual_start_utc':started,'actual_end_utc':utc(),'sampling_elapsed_seconds':elapsed,
        'source_sha256':actual_sources,'fixture_sha256':FIXTURE_SHA,'proposal_sha256':PROPOSAL_SHA,'plan_sha256':sha(args.plan),
        'n_draws':plan['n_draws'],'rows':2,'H_counts':[int((H==h).sum()) for h in range(3)],
        'H_frequencies':[float((H==h).mean()) for h in range(3)],
        'modeled_z_empirical_mean':z.mean(axis=0).tolist(),'modeled_z_empirical_covariance_ddof1':np.cov(z,rowvar=False,ddof=1).tolist(),
        'modeled_CDF_points':points.tolist(),'modeled_empirical_CDF_by_row':[[float((z[:,row]<=point).mean()) for point in points] for row in range(2)],
        'native_CDF_thresholds':thresholds.tolist(),'native_empirical_CDF_by_row':[[float((native[:,row]<=point).mean()) for point in thresholds] for row in range(2)],
        'input_unchanged':True,'input_sha256_before':before_sha,'input_sha256_after':after_sha,
        'files':{'draws_and_aux.npz':sha(npz),'metadata.json':sha(args.output/'metadata.json'),
            'INPUT_BINDING.json':sha(args.output/'INPUT_BINDING.json')},'all_checks_reserved_for_independent_auditor':True,
        'statistical_generated_not_measurements':True,'new_material_N':0,'model_updates':0,'material_source_reads_or_fits':0}
    write(args.output/'RESULTS.json',result)
    write(args.output/'EXECUTION_COMPLETE.json',{'actual_utc':utc(),'status':result['status'],'results_sha256':sha(args.output/'RESULTS.json'),
        'draws_sha256':sha(npz),'n_draws_not_new_materialN':True,'new_material_N':0,'model_updates':0,'material_source_reads_or_fits':0})
    print(json.dumps({'status':result['status'],'results_sha256':sha(args.output/'RESULTS.json'),
        'draws_sha256':sha(npz),'npz_bytes':npz.stat().st_size,'n_draws':plan['n_draws'],'new_material_N':0},indent=2))

if __name__=='__main__':main()
