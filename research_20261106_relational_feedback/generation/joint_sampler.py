"""Joint posterior predictive missing-Y samples from one frozen three-NIG ensemble.

Classic posterior simulation, not a new model/algorithm or calibration claim.
Each draw shares exactly one relation H, noise variance and coefficient vector
across all distinct physical rows; row-specific residuals remain independent.
There is no source-data I/O, model update, oracle call or generated-label training.
The caller must verify frozen state/source hashes before calling this pure API.
"""
import math
import numpy as np

MODEL_IDS=('affine','interaction','fullquad')
DIMENSIONS=(3,4,6)
CORE_REFERENCE_SHA256='1d74cde91519ce282c6ebee54e53807a2a79be0cebf19e06221eb4f16e8b7528'


def _design(model_id,contexts,expansion):
    terms=np.column_stack((contexts[:,0]*contexts[:,1],contexts[:,0]**2,contexts[:,1]**2))
    if expansion is not None:
        center=np.asarray(expansion['center'],dtype=float);sd=np.asarray(expansion['sd'],dtype=float)
        if center.shape!=(3,) or sd.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(sd).all() or (sd<=0).any():
            raise ValueError('three finite fixed expansion centers and positive SDs required')
        terms=(terms-center)/sd
    result=np.column_stack((np.ones(len(contexts)),contexts,terms))
    if not np.isfinite(result).all():raise ValueError('finite expanded contexts required')
    return result[:,:DIMENSIONS[MODEL_IDS.index(model_id)]]


def _state(model,dimension):
    state=model['state'];mean=np.asarray(state['mean'],dtype=float);cov=np.asarray(state['V'],dtype=float)
    alpha,b=float(state['alpha']),float(state['b'])
    if (mean.shape!=(dimension,) or cov.shape!=(dimension,dimension) or not np.isfinite(mean).all()
        or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-12,rtol=0)
        or not math.isfinite(alpha) or alpha<=1 or not math.isfinite(b) or b<=0):
        raise ValueError('finite SPD NIG with alpha>1 and b>0 required')
    return mean,np.linalg.cholesky(cov),alpha,b


def _inverse_yj_half(value):
    result=np.empty_like(value);positive=value>=0
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        result[positive]=value[positive]*(1+value[positive]/4)
        result[~positive]=-np.expm1((2/3)*np.log1p(-1.5*value[~positive]))
    if not np.isfinite(result).all():raise ArithmeticError('generated inverse YJ outside supported numerical range')
    return result


def sample_joint(ensemble,context_z,response,*,seed,n_draws,return_auxiliary=False):
    """Return native_draws[D,N], metadata, and optional simulated-parameter audit arrays.

    Rows represent distinct physical units with known two-proxy contexts. Identical
    contexts are legal for distinct units; physical identity deduplication belongs
    to the task controller. Reusing this seed/call is reproducibility, not new data.
    """
    if type(seed) is not int or seed<0 or type(n_draws) is not int or n_draws<=0:
        raise ValueError('nonnegative integer seed and positive integer n_draws required')
    contexts=np.asarray(context_z,dtype=float)
    if contexts.ndim!=2 or contexts.shape[1]!=2 or not len(contexts) or not np.isfinite(contexts).all():
        raise ValueError('nonempty finite N by2 fixed standardized contexts required')
    models=ensemble['models']
    if tuple(m['id'] for m in models)!=MODEL_IDS:raise ValueError('fixed ordered three-hypothesis ensemble required')
    logs=np.asarray(ensemble['log_weights'],dtype=float)
    if logs.shape!=(3,) or not np.isfinite(logs).all():raise ValueError('three finite retained log probabilities required')
    maximum=float(logs.max());normalizer=maximum+math.log(float(np.exp(logs-maximum).sum()))
    logweights=logs-normalizer;weights=np.exp(logweights)
    # Normalize only IEEE summation drift; no positive floor or discarded logweight.
    probabilities=weights/float(weights.sum())
    center,sd,s0=(float(response[key]) for key in ('center','sd','s0'))
    if not all(math.isfinite(x) for x in (center,sd,s0)) or sd<=0 or s0<=0:
        raise ValueError('finite shared response center and positive sd/s0 required')
    states=[_state(m,d) for m,d in zip(models,DIMENSIONS)]
    designs=[_design(m['id'],contexts,ensemble.get('expansion_transform')) for m in models]
    rng=np.random.Generator(np.random.PCG64(seed))
    h_index=rng.choice(3,size=n_draws,p=probabilities)
    latent=np.empty((n_draws,len(contexts)),dtype=float);sigma2_all=np.empty(n_draws,dtype=float)
    auxiliary_parameters={};counts=[]
    for h,(state,phi) in enumerate(zip(states,designs)):
        positions=np.flatnonzero(h_index==h);count=len(positions);counts.append(count)
        mean,chol,alpha,b=state
        if count:
            gamma=rng.gamma(shape=alpha,scale=1.,size=count)
            if not np.isfinite(gamma).all() or (gamma<=0).any():raise ArithmeticError('positive finite Gamma draws required')
            sigma2=b/gamma;sigma=np.sqrt(sigma2)
            coefficients=mean[None,:]+sigma[:,None]*(rng.standard_normal((count,len(mean)))@chol.T)
            # One coefficient vector and sigma per draw; each row gets a new residual.
            latent[positions,:]=coefficients@phi.T+sigma[:,None]*rng.standard_normal((count,len(contexts)))
            sigma2_all[positions]=sigma2
        else:
            coefficients=np.empty((0,len(mean)),dtype=float)
        if return_auxiliary:
            auxiliary_parameters[MODEL_IDS[h]]={'draw_indices':positions,'coefficient_draws':coefficients}
    with np.errstate(over='raise',invalid='raise'):
        native=s0*_inverse_yj_half(center+sd*latent)
    if not np.isfinite(native).all():raise ArithmeticError('finite native posterior predictive samples required')
    metadata={'kind':'statistical_generated_joint_posterior_predictive_missingY','statistical_generated':True,
        'number_draws_not_new_materialN':True,'n_draws':n_draws,'rows_per_joint_draw':len(contexts),
        'seed':seed,'rng':'numpy.Generator(PCG64)','numpy_version':np.__version__,
        'hypothesis_ids':list(MODEL_IDS),'hypothesis_draw_counts':counts,
        'retained_input_log_weights':logs.tolist(),'normalized_log_weights':logweights.tolist(),
        'display_probabilities':probabilities.tolist(),'display_weight_underflow_hypotheses':int((weights==0).sum()),
        'probability_floor_used':False,'H_draws':n_draws,'noise_variance_draws':n_draws,
        'coefficient_vector_draws':n_draws,'row_residual_draws':n_draws*len(contexts),
        'rowwise_H_or_parameter_resampling':False,'all_rows_share_H_sigma_and_coefficients_per_draw':True,
        'row_residuals_independent_given_shared_parameters':True,
        'prior_core_reference_sha256':CORE_REFERENCE_SHA256,'core_source_hash_read_or_verified_by_sampler':False,
        'frozen_input_binding_must_be_verified_by_caller':True,'data_source_reads':0,'model_updates':0,
        'samples_are_measurements':False,'calibration_or_material_effect_claim':False,
        'response_transform':{'center':center,'sd':sd,'s0':s0},
        'generative_law':'H~posterior; sigma2~IG(alpha_H,b_H); beta~N(m_H,sigma2 V_H); z_i=phi_H(x_i) beta+epsilon_i, epsilon_i iid N(0,sigma2)'}
    result={'native_draws':native,'metadata':metadata}
    if return_auxiliary:
        result['auxiliary']={'modeled_z_draws':latent,'H_index_per_joint_draw':h_index,
            'sigma2_per_joint_draw':sigma2_all,'coefficient_draws_grouped_by_H':auxiliary_parameters,
            'not_observed_or_generated_training_labels':True}
    return result
