"""Independent stored-joint-draw audit: analytic NIG mixture laws and auxiliary residuals."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import traceback
import numpy as np
from scipy import special

IDS=('affine','interaction','fullquad')
DIMS=(3,4,6)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def jsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(x,m):
    if not x:raise AssertionError(m)


def design(name,contexts,expansion):
    a,b=contexts[:,0],contexts[:,1];terms=np.column_stack((a*b,a*a,b*b))
    if expansion is not None:terms=(terms-np.array(expansion['center']))/np.array(expansion['sd'])
    return np.column_stack((np.ones(len(contexts)),contexts,terms))[:,:DIMS[IDS.index(name)]]


def moments(array):
    n=len(array);mean=np.array([math.fsum(array[:,i])/n for i in range(array.shape[1])])
    covariance=np.array([[math.fsum((array[:,i]-mean[i])*(array[:,j]-mean[j]))/(n-1)
                          for j in range(array.shape[1])] for i in range(array.shape[1])])
    return mean,covariance


def forward_native(native,response):
    x=native/response['s0'];u=np.empty_like(x);mask=x>=0
    u[mask]=2*x[mask]/(np.sqrt(1+x[mask])+1)
    u[~mask]=-np.expm1(1.5*np.log1p(-x[~mask]))/1.5
    return (u-response['center'])/response['sd']


def inverse_threshold(point,response):
    u=response['center']+response['sd']*point
    g=u*(1+u/4) if u>=0 else -math.expm1((2/3)*math.log1p(-1.5*u))
    return response['s0']*g


def audit(plan):
    require(sha(__file__)==plan['script_sha256'],'audit source changed')
    loaded={}
    for name,pin in plan['inputs'].items():
        require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name)
        if name!='draw_archive':loaded[name]=read(pin['file'])
    proposal=loaded['proposal'];fixture=loaded['fixture'];result=loaded['producer_result'];metadata=loaded['metadata'];binding=loaded['input_binding']
    require(proposal['n_draws']==131072 and proposal['seed']==20261106,'fixed sample plan')
    expected={'ensemble':fixture['initial_ensemble'],'context_z':[fixture['fixed_fixture']['query_context'],fixture['fixed_fixture']['target_context']],
              'response':fixture['fixed_fixture']['response_transform']}
    require(binding['actual_input']==expected,'oldfixture exactinput only')
    expected_sha=jsha(expected)
    require(binding['input_sha256_before']==metadata['input_sha256_before']==metadata['input_sha256_after']==result['input_sha256_before']==result['input_sha256_after']==expected_sha,'unchangedfrozen input')
    require(result['plan_sha256']==metadata['plan_sha256']==plan['inputs']['producer_plan']['sha256'],'actual sampling plan binding')
    archive=plan['inputs']['draw_archive']['file']
    require(metadata['npz_sha256']==result['files']['draws_and_aux.npz']==sha(archive),'stored drawarchive hashes')
    require(result['files']['metadata.json']==plan['inputs']['metadata']['sha256'] and result['files']['INPUT_BINDING.json']==plan['inputs']['input_binding']['sha256'],'sampling reportpayload pins')
    with np.load(archive,allow_pickle=False) as data:
        keys={'native','z','H','sigma2'}|{'coef_h%d'%i for i in range(3)}|{'indices_h%d'%i for i in range(3)}
        require(set(data.files)==keys,'exact nonobject archivekeys');arrays={k:data[k] for k in data.files}
    for key,a in arrays.items():
        require(not a.dtype.hasobject and np.isfinite(a).all(),'finite nonobject arrays')
        require(metadata['arrays'][key]=={'shape':list(a.shape),'dtype':str(a.dtype)},'shape/dtype index')
    n=131072;z=arrays['z'];native=arrays['native'];hindex=arrays['H'];sigma2=arrays['sigma2']
    require(z.shape==native.shape==(n,2) and hindex.shape==sigma2.shape==(n,),'declared jointsample shapes')
    require(np.issubdtype(hindex.dtype,np.integer) and np.isin(hindex,[0,1,2]).all() and (sigma2>0).all(),'one valid H/sigma perdraw')
    ensemble=expected['ensemble'];contexts=np.array(expected['context_z']);response=expected['response']
    logw=np.array(ensemble['log_weights']);logw=logw-float(special.logsumexp(logw));weights=np.exp(logw)
    component_means=[];component_covariances=[];component_scales=[];nu=[];residual=np.empty_like(z)
    counts=[]
    for i,(wrapped,dimension) in enumerate(zip(ensemble['models'],DIMS)):
        require(wrapped['id']==IDS[i],'fixedHorder')
        state=wrapped['state'];m=np.array(state['mean']);v=np.array(state['V']);phi=design(IDS[i],contexts,ensemble.get('expansion_transform'))
        mean=phi@m;matrix=np.eye(2)+phi@v@phi.T
        component_means.append(mean);component_covariances.append(state['b']/(state['alpha']-1)*matrix)
        component_scales.append(np.sqrt(state['b']/state['alpha']*np.diag(matrix)));nu.append(2*state['alpha'])
        positions=arrays['indices_h%d'%i];coefficients=arrays['coef_h%d'%i];wanted=np.flatnonzero(hindex==i)
        require(np.array_equal(positions,wanted) and coefficients.shape==(len(wanted),dimension),'one groupedcoeffvector perdrawH')
        counts.append(len(wanted));residual[wanted]=(z[wanted]-coefficients@phi.T)/np.sqrt(sigma2[wanted])[:,None]
    analytic_mean=sum(w*m for w,m in zip(weights,component_means))
    analytic_cov=sum(w*(v+np.outer(m-analytic_mean,m-analytic_mean)) for w,m,v in zip(weights,component_means,component_covariances))
    mean,cov=moments(z);resid_mean,resid_cov=moments(residual);checks=proposal['fixed_checks'];errors={}
    errors['H_frequency']=float(np.max(np.abs(np.array(counts)/n-weights)))
    errors['latent_mean']=float(np.max(np.abs(mean-analytic_mean)))
    errors['latent_covariance']=float(np.max(np.abs(cov-analytic_cov)))
    errors['standard_residual_mean']=float(np.max(np.abs(resid_mean)))
    errors['standard_residual_covariance']=float(np.max(np.abs(resid_cov-np.eye(2))))
    require(errors['H_frequency']<=checks['mixture_model_H_frequency_max_abs_probability_error'],'fixed Hfreq tolerance')
    require(errors['latent_mean']<=checks['latent_mean_max_abs_error'],'fixed latentmean tolerance')
    require(errors['latent_covariance']<=checks['latent_covariance_max_abs_error'],'fixed latentcovariance tolerance')
    require(errors['standard_residual_mean']<=checks['auxiliary_standardized_row_residual_mean_max_abs'],'fixed residualmean tolerance')
    require(errors['standard_residual_covariance']<=checks['auxiliary_standardized_row_residual_covariance_max_abs_from_identity'],'fixed residualcov tolerance')
    points=checks['modeled_CDF_points_per_row'];cdf=[];native_cdf=[];references=[];binomial_SE=[]
    for row in range(2):
        observed=[];native_observed=[];expected_cdf=[];stderr=[]
        for point in points:
            empirical=float(np.count_nonzero(z[:,row]<=point)/n)
            threshold=inverse_threshold(point,response);empirical_native=float(np.count_nonzero(native[:,row]<=threshold)/n)
            target=math.fsum(float(w)*float(special.stdtr(v,(point-m[row])/s[row])) for w,m,s,v in zip(weights,component_means,component_scales,nu))
            observed.append(empirical);native_observed.append(empirical_native);expected_cdf.append(target);stderr.append(math.sqrt(target*(1-target)/n))
        cdf.append(observed);native_cdf.append(native_observed);references.append(expected_cdf);binomial_SE.append(stderr)
    errors['modeled_CDF']=float(np.max(np.abs(np.array(cdf)-references)))
    errors['native_CDF']=float(np.max(np.abs(np.array(native_cdf)-references)))
    require(errors['modeled_CDF']<=checks['mixture_marginal_CDF_max_abs_probability_error'] and errors['native_CDF']<=checks['mixture_marginal_CDF_max_abs_probability_error'],'fixed marginalCDF tolerance')
    recovered=forward_native(native,response);errors['deterministic_inverse_forward_maxabs']=float(np.max(np.abs(recovered-z)))
    require(np.all(np.abs(recovered-z)<=2e-9+2e-11*np.abs(z)),'deterministic samplecoordinate roundtrip')
    require(result['H_counts']==metadata['hypothesis_draw_counts']==counts,'H count summaries')
    require(np.max(np.abs(np.array(result['modeled_z_empirical_mean'])-mean))<=2e-12,'stored mean bookkeeping')
    require(np.max(np.abs(np.array(result['modeled_z_empirical_covariance_ddof1'])-cov))<=2e-11,'stored covariance bookkeeping')
    require(result['modeled_empirical_CDF_by_row']==cdf and result['native_empirical_CDF_by_row']==native_cdf,'stored CDFbookkeeping')
    for name,value in checks['metadata_counters_exact'].items():require(metadata[name]==value,'exactgeneratedcounter:'+name)
    require(metadata['rowwise_H_or_parameter_resampling'] is False and metadata['all_rows_share_H_sigma_and_coefficients_per_draw'] is True,'jointgeneration declaration')
    require(metadata['samples_are_measurements'] is False and result['new_material_N']==0 and result['model_updates']==0,'notmeasurements or pseudolabelupdates')
    return {'status':'PASS','fixed_draws':n,'distinct_response_rows_perdraw':2,'new_material_N':0,
            'analytic_latent_mean':analytic_mean.tolist(),'analytic_latent_covariance':analytic_cov.tolist(),
            'empirical_latent_mean':mean.tolist(),'empirical_latent_covariance':cov.tolist(),
            'standardized_row_residual_mean':resid_mean.tolist(),'standardized_row_residual_covariance':resid_cov.tolist(),
            'analytic_CDF_byrow':references,'empirical_modeled_CDF_byrow':cdf,'empirical_native_CDF_byrow':native_cdf,
            'bounded_CDF_binomial_standard_errors_descriptive':binomial_SE,'errors':errors,
            'sampler_imports_newdraws_or_fits':0,'fixedseed_stream_not_regenerated':True,
            'scope':'singleoldfixture finiteMC interface sanity; not calibration, modeltraining, materialeffect, or independent measurements',
            'moment_limit':'latent fourth moments finite atnu6; no Gaussian MCSE for potentially unstable native covariance'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();plan=read(a.plan);approval=read(a.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_STORED_JOINT_SAMPLES_ONCE'
            and approval['plan_sha256']==sha(a.plan) and approval['script_sha256']==sha(__file__),'exact root MCauditapproval')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat()
    try:result=audit(plan)
    except Exception as problem:result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
                   'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval)})
    f=out/'INDEPENDENT_JOINT_SAMPLE_AUDIT.json'
    with f.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
