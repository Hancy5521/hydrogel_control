"""Warm-only coordinate adapter and laws for the frozen three-relation NIG core.

No data I/O. This is a thin material task layer, not another regression model.
Caller must supply only authorized warm rows and revealed pool labels. All response
and expansion coordinates freeze after the common warm data; H evidence uses the
same response Jacobian, which cancels from odds. No probability floor or rescue fit.
"""
import copy
import math
import numpy as np
from scipy import special
import relation_posterior as posterior


def yj(value):
    value=np.asarray(value,dtype=float)
    if not np.isfinite(value).all():raise ValueError('finite native inputs required')
    result=np.empty_like(value);positive=value>=0
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        result[positive]=2*value[positive]/(np.sqrt(1+value[positive])+1)
        result[~positive]=-np.expm1(1.5*np.log1p(-value[~positive]))/1.5
    if not np.isfinite(result).all():raise ArithmeticError('YJ outside supported floating range')
    return result


def inverse_yj(value):
    value=np.asarray(value,dtype=float);result=np.empty_like(value);positive=value>=0
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        result[positive]=value[positive]*(1+value[positive]/4)
        result[~positive]=-np.expm1((2/3)*np.log1p(-1.5*value[~positive]))
    if not np.isfinite(result).all():raise ArithmeticError('inverse YJ outside supported floating range')
    return result


def proxy_z(transform,native_proxies):
    x=np.asarray(native_proxies,dtype=float)
    if x.ndim not in (1,2) or x.shape[-1]!=2 or not np.isfinite(x).all():raise ValueError('complete finite dual native proxies required')
    return (yj(x/np.asarray(transform['native_first_scale_s0'])[:2])-np.asarray(transform['u_center_warm'])[:2])/np.asarray(transform['u_sample_SD_warm'])[:2]


def response_z(response,native_target):
    if not np.isfinite(native_target).all():raise ValueError('finite authorized native target required')
    return (yj(np.asarray(native_target,dtype=float)/response['s0'])-response['center'])/response['sd']


def fit_warm(native_rows):
    native=np.asarray(native_rows,dtype=float)
    if native.shape!=(32,3) or not np.isfinite(native).all():raise ValueError('exactly32complete authorized warm rows required')
    first=native.std(axis=0,ddof=1)
    if not np.isfinite(first).all() or (first<=0).any():raise ValueError('nonconstant native warm fields; no arbitrary scale floors')
    u=yj(native/first);center=u.mean(axis=0);sd=u.std(axis=0,ddof=1)
    if not np.isfinite(sd).all() or (sd<=0).any():raise ValueError('nonconstant warm YJ fields required')
    z=(u-center)/sd
    expansion=posterior.fit_expansion_transform(z[:,:2]);ensemble=posterior.initial_ensemble(expansion)
    log_evidence=np.zeros(3);history=[]
    for i,row in enumerate(z):
        ensemble,diag=posterior.update_actual(ensemble,row[:2],float(row[2]))
        log_evidence+=np.asarray(diag['preupdate_logpredictive'])
        history.append({'warm_row_order_zero_based':i,'preupdate_logpredictive':diag['preupdate_logpredictive'],
            'posterior_log_weights':ensemble['log_weights']})
    t=native[:,2]/first[2]
    derivative_log=np.where(t>=0,-.5*np.log1p(np.maximum(t,0)),.5*np.log1p(np.maximum(-t,0)))
    common_jacobian=float(np.sum(derivative_log-math.log(first[2])-math.log(sd[2])))
    transform={'native_first_scale_s0':first.tolist(),'u_center_warm':center.tolist(),
        'u_sample_SD_warm':sd.tolist(),'warm_rows':32,'sample_SD_ddof':1,'frozen_after_warm':True}
    return {'ensemble':ensemble,'transform':transform,
        'response':{'center':float(center[2]),'sd':float(sd[2]),'s0':float(first[2])},
        'native_features':{'center':native[:,:2].mean(axis=0).tolist(),'sd':native[:,:2].std(axis=0,ddof=1).tolist()},
        'warm_modeled':z.tolist(),'log_evidence_by_H_modeled_response':log_evidence.tolist(),
        'common_response_native_logJacobian_sum':common_jacobian,
        'log_evidence_by_H_native_response':(log_evidence+common_jacobian).tolist(),
        'warm_evidence_history':history,'coordinates_conditioned_fixed_not_integrated_as_uncertain':True,
        'equal_H_prior_not_equal_posterior':True}


def update_native(ensemble,context_z,native_target,response):
    return posterior.update_actual(ensemble,context_z,float(response_z(response,native_target)))


def law(ensemble,context_z,response,logweights_override=None):
    posterior.validate_ensemble(ensemble)
    logs=posterior.normalize_logweights(ensemble['log_weights'] if logweights_override is None else logweights_override)
    components=[]
    for model in ensemble['models']:
        phi=posterior.feature_map(model['id'],context_z,ensemble.get('expansion_transform'))
        component=posterior.scalar_law(model['state'],phi)
        components.append({'id':model['id'],'location':response['center']+response['sd']*component['mu'],
            'scale':response['sd']*component['scale'],'df':component['nu']})
    return {'coordinate':'u=YJ_half(native/s0)','family':'mixture_student_t','components':components,
        'log_weights':logs,'weights_display_only':np.exp(logs).tolist(),'native_first_scale_s0':response['s0'],
        'lambda':.5,'unit':'source bulk GPa attribution, aggregation qualified',
        'model_weight_override':logweights_override is not None}


def affine_law(ensemble,context_z,response):
    predicted=law(ensemble,context_z,response);component=predicted['components'][0]
    return {'coordinate':predicted['coordinate'],'family':'student_t','location':component['location'],
        'scale':component['scale'],'df':component['df'],'native_first_scale_s0':response['s0'],
        'lambda':.5,'unit':predicted['unit']}


def mixture_cdf_native(predicted,native_y):
    u=yj(np.asarray(native_y,dtype=float)/predicted['native_first_scale_s0'])
    weights=np.exp(posterior.normalize_logweights(predicted['log_weights']))
    return sum(w*special.stdtr(c['df'],(u-c['location'])/c['scale']) for w,c in zip(weights,predicted['components']))


def mixture_sf_native(predicted,native_y):
    u=yj(np.asarray(native_y,dtype=float)/predicted['native_first_scale_s0'])
    weights=np.exp(posterior.normalize_logweights(predicted['log_weights']))
    return sum(w*special.stdtr(c['df'],-(u-c['location'])/c['scale']) for w,c in zip(weights,predicted['components']))


def openloop_from_warm(warm_ensemble,current_ensemble):
    """Warm means/expected noise/H weights; current label-independent V and count.

    Future hypothetical observation still uses the complete legal Bayesian joint
    lookahead through risk_field. The actual forecast ensemble is not this control.
    """
    result=copy.deepcopy(warm_ensemble)
    for warm,current,target in zip(warm_ensemble['models'],current_ensemble['models'],result['models']):
        step=current['state']['labeled_updates']-warm['state']['labeled_updates']
        if step<0:raise ValueError('current design cannot precede warm state')
        initial=warm['state'];state=target['state'];state['V']=copy.deepcopy(current['state']['V'])
        state['alpha']=initial['alpha']+step/2
        state['b']=(state['alpha']-1)*initial['b']/(initial['alpha']-1)
        state['labeled_updates']=current['state']['labeled_updates']
    result['true_label_updates']=current_ensemble['true_label_updates']
    result['acquisition_control']='warmmean_expectedvariance_and_Hweights;currentVdesignonly;hypotheticalBayeslookahead'
    return result
