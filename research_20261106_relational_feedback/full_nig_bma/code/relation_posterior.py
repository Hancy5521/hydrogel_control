"""Three fixed conditional NIG relations with exact sequential model evidence.

Classical conjugate regression/Bayesian model averaging. No data I/O, clipping of
model probabilities, actual-label oracle, particle approximation or learned gating.
Every model shares the same fixed response coordinate; its native Jacobian cancels
from posterior model odds. Keep log weights even when display probabilities underflow.
"""
import copy
import math
import numpy as np

MODEL_IDS = ('affine', 'interaction', 'fullquad')
DIMENSIONS = (3, 4, 6)


def feature_map(model_id, context, expansion_transform=None):
    x = np.asarray(context, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all():
        raise ValueError('two finite fixed-warm standardized known contexts required')
    terms = np.asarray([x[0]*x[1], x[0]**2, x[1]**2])
    if expansion_transform is not None:
        center = np.asarray(expansion_transform['center'], dtype=float)
        sd = np.asarray(expansion_transform['sd'], dtype=float)
        if center.shape != (3,) or sd.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(sd).all() or (sd <= 0).any():
            raise ValueError('three fixed warm expansion centers and positive SDs required')
        terms = (terms-center)/sd
    basis = np.concatenate(([1.], x, terms))
    try: dimension = DIMENSIONS[MODEL_IDS.index(model_id)]
    except ValueError as exc: raise ValueError('one of the three fixed feature hypotheses required') from exc
    if not np.isfinite(basis).all(): raise ValueError('finite feature expansion required')
    return basis[:dimension]


def fit_expansion_transform(warm_contexts):
    """Caller must register warm-only fitting; never invoke using pool/terminal Ys."""
    x = np.asarray(warm_contexts, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError('at least two complete warm contexts required')
    terms = np.column_stack((x[:,0]*x[:,1], x[:,0]**2, x[:,1]**2))
    sd = terms.std(axis=0, ddof=1)
    if not np.isfinite(terms).all() or (sd <= 0).any(): raise ValueError('nonconstant warm expansion terms required')
    return {'center':terms.mean(axis=0).tolist(), 'sd':sd.tolist(), 'ddof':1, 'frozen_after_warm':True}


def checked(state):
    m, v = np.asarray(state['mean'], dtype=float), np.asarray(state['V'], dtype=float)
    if (m.ndim != 1 or v.shape != (len(m),len(m)) or not np.isfinite(m).all() or not np.isfinite(v).all()
        or not np.allclose(v,v.T,rtol=0,atol=1e-12) or not math.isfinite(state['alpha']) or state['alpha'] <= 1
        or not math.isfinite(state['b']) or state['b'] <= 0):
        raise ValueError('finite SPD NIG with alpha>1,b>0 required')
    np.linalg.cholesky(v)
    return m,v


def checked_phi(state, phi):
    m,v=checked(state);phi=np.asarray(phi,dtype=float)
    if phi.shape != m.shape or not np.isfinite(phi).all(): raise ValueError('finite matching design vector required')
    return m,v,phi


def scalar_law(state, phi):
    m,v,phi=checked_phi(state,phi)
    return {'mu':float(phi@m), 'scale':math.sqrt(state['b']/state['alpha']*(1+float(phi@v@phi))), 'nu':2*state['alpha']}


def logpredictive(state, phi, modeled_label):
    if not math.isfinite(modeled_label): raise ValueError('finite observed response required')
    law=scalar_law(state,phi);nu=law['nu'];z=(modeled_label-law['mu'])/law['scale']
    return (math.lgamma((nu+1)/2)-math.lgamma(nu/2)-.5*math.log(nu*math.pi)
        -math.log(law['scale'])-(nu+1)/2*math.log1p(z*z/nu))


def update_nig(state,phi,modeled_label):
    m,v,phi=checked_phi(state,phi)
    if not math.isfinite(modeled_label): raise ValueError('finite actually observed modeled response required')
    projection=v@phi;denominator=1+float(phi@projection);residual=modeled_label-float(phi@m)
    result={'mean':(m+projection*residual/denominator).tolist(),
        'V':(v-np.outer(projection,projection)/denominator).tolist(),
        'alpha':state['alpha']+.5,'b':state['b']+.5*residual**2/denominator,
        'labeled_updates':state.get('labeled_updates',0)+1}
    checked(result)
    return result


def normalize_logweights(values):
    values=np.asarray(values,dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all(): raise ValueError('three finite log weights required; no flooring')
    maximum=float(values.max());normalizer=maximum+math.log(float(np.exp(values-maximum).sum()))
    return (values-normalizer).tolist()


def weights(ensemble):
    return np.exp(np.asarray(normalize_logweights(ensemble['log_weights']))).tolist()


def initial_ensemble(expansion_transform=None):
    states=[{'id':name,'state':{'mean':[0.]*dimension,'V':np.eye(dimension).tolist(),
        'alpha':3.,'b':1.5,'labeled_updates':0}} for name,dimension in zip(MODEL_IDS,DIMENSIONS)]
    return {'models':states,'log_weights':[-math.log(3)]*3,'expansion_transform':copy.deepcopy(expansion_transform),
        'response_coordinate':'shared_fixed_warm_YJ_half_then_affine','true_label_updates':0}


def validate_ensemble(ensemble):
    normalize_logweights(ensemble['log_weights'])
    if tuple(model['id'] for model in ensemble['models']) != MODEL_IDS: raise ValueError('fixed ordered three hypotheses required')
    for model,dimension in zip(ensemble['models'],DIMENSIONS):
        m,_=checked(model['state'])
        if len(m)!=dimension: raise ValueError('fixed hypothesis dimension mismatch')


def update_actual(ensemble, context, modeled_label):
    """First old-state predictive evidence, then component NIG label updates.

    This pure function cannot verify physical measurement provenance. An external
    controller must freeze each prediction before its trusted label oracle returns.
    """
    validate_ensemble(ensemble);logs=normalize_logweights(ensemble['log_weights']);new_models=[];increments=[]
    for model in ensemble['models']:
        phi=feature_map(model['id'],context,ensemble.get('expansion_transform'))
        increments.append(logpredictive(model['state'],phi,modeled_label))
        new_models.append({'id':model['id'],'state':update_nig(model['state'],phi,modeled_label)})
    posterior=normalize_logweights(np.asarray(logs)+np.asarray(increments))
    return {'models':new_models,'log_weights':posterior,'expansion_transform':copy.deepcopy(ensemble.get('expansion_transform')),
        'response_coordinate':ensemble.get('response_coordinate','shared_fixed_response'),
        'true_label_updates':ensemble.get('true_label_updates',0)+1}, {'preupdate_logpredictive':increments,
        'old_log_weights':logs,'new_log_weights':posterior,'new_weights_display_only':np.exp(posterior).tolist(),
        'evidence_before_NIG_update':True,'native_shared_response_Jacobian_cancels':True,'probability_floor':False}


def pairs(ensemble,query_context,target_context):
    validate_ensemble(ensemble);result=[]
    for model in ensemble['models']:
        state=model['state'];m,v=checked(state)
        q=feature_map(model['id'],query_context,ensemble.get('expansion_transform'))
        t=feature_map(model['id'],target_context,ensemble.get('expansion_transform'))
        qd,td=1+float(q@v@q),1+float(t@v@t);rho=float(q@v@t)/math.sqrt(qd*td)
        if not -1<rho<1: raise ArithmeticError('distinct physical rows require nondegenerate bivariate predictive law')
        result.append({'id':model['id'],'mu_query':float(q@m),'mu_target':float(t@m),
            'scale_query':math.sqrt(state['b']/state['alpha']*qd),'scale_target':math.sqrt(state['b']/state['alpha']*td),
            'nu':2*state['alpha'],'rho':rho,'query_phi':q.tolist(),'target_phi':t.tolist(),
            'different_rows_independent_residuals':True})
    return result


def conditional_target(pair,standardized_query):
    nu,rho=pair['nu'],pair['rho']
    return {'mu':pair['mu_target']+rho*pair['scale_target']*standardized_query,
        'scale':pair['scale_target']*math.sqrt((nu+standardized_query**2)/(nu+1)*(1-rho**2)), 'nu':nu+1}
