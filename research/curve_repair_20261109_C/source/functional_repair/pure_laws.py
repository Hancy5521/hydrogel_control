"""SOURCE ONLY: paired modulus mean transport, keeping full Gaussian residuals.

Only mathematical log angular frequency is shared with generic curve code.
No old viscosity mean, normalization, theta, source table or material fit loads.
The fixed transport is classical location displacement, not new flow training.
"""
from __future__ import annotations
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import nnls
from scipy.special import gammaln, stdtr
from scipy.stats import norm, t, multivariate_normal, multivariate_t

ROOT = Path(__file__).resolve().parent.parent
PINS = {'generalization/anchor_gp.py': '0220acc8ff8768d759a3ab87f22f7a2846aa121d628725bfb83480f16c771113',
        'generalization/generalized_maxwell_nnls.py': '5164b9b197016efc5a7ca960db1927830e676762f77b7e7d6726a27992f3882a'}
METHODS = ('full_anchor_gaussian', 'half_maxwell_mean_gaussian',
           'maxwell_mean_gaussian', 'half_maxwell_mean_student4', 'full_anchor_student4')
MAIN = METHODS[1]
POINT_METHODS = ('channel_relative_Maxwell_NNLS_point','raw_Pa_Maxwell_NNLS_point')
ALL_METHODS = METHODS+POINT_METHODS


def pinned(relative):
    path = ROOT/relative; raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[relative]:
        raise ValueError('Pinned probability/physical source differs')
    name = '_functional_repair_'+PINS[relative]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        exec(compile(raw, str(path), 'exec'), module.__dict__)
    return sys.modules[name]


def assemble_laws(gp_mean, gp_covariance, maxwell_points_Pa):
    m, c, point = map(lambda a: np.array(a, dtype=float, copy=True),
                      (gp_mean, gp_covariance, maxwell_points_Pa))
    if m.shape != (14,) or c.shape != (14,14) or point.shape != (14,):
        raise ValueError('Full fourteen-dimensional paired-target law required')
    if not all(np.isfinite(a).all() for a in (m,c,point)) or np.any(point <= 0):
        raise ValueError('Finite laws and strictly positive Maxwell points required; no floor')
    if not np.allclose(c,c.T,atol=1e-12,rtol=1e-12):
        raise ValueError('Symmetric complete residual covariance required')
    np.linalg.cholesky(c)
    mm = np.log(point); half = .5*(m+mm)
    laws = {}
    for method, mean in zip(METHODS[:3], (m,half,mm)):
        laws[method] = {'distribution':'multivariate_gaussian_log_modulus',
                        'mean':mean.tolist(),'covariance':c.tolist()}
    laws[METHODS[3]] = {'distribution':'multivariate_student4_log_modulus',
                       'location':half.tolist(),'scale_matrix':(.5*c).tolist(),
                       'predictive_covariance':c.tolist(),'df':4}
    laws[METHODS[4]] = {'distribution':'multivariate_student4_log_modulus',
                       'location':m.tolist(),'scale_matrix':(.5*c).tolist(),
                       'predictive_covariance':c.tolist(),'df':4}
    return laws


def weighted_maxwell(fo,gprime,gloss,ft,native):
    """Fixed channel-relative objective using only each channel own-six RMS.

    Relative channel LS is dimensionless, approximately compatible with relative
    errors. It is not exactly logarithmic LS and does not learn weights or tau.
    The original raw-Pa NNLS remains a separate point comparator.
    """
    physics=pinned('generalization/generalized_maxwell_nnls.py')
    fo,gprime,gloss,ft,native=map(lambda v:np.asarray(v,dtype=float),(fo,gprime,gloss,ft,native))
    if (fo.shape!=(6,) or gprime.shape!=(6,) or gloss.shape!=(6,) or ft.shape!=(7,) or
            native.shape!=(13,) or np.any(gprime<=0) or np.any(gloss<=0)):
        raise ValueError('Own-six positive paired observations and native6/7 required')
    tau=physics.fixed_relaxation_times(native); a=physics.design(fo,tau); at=physics.design(ft,tau)
    sigma=np.array([np.linalg.norm(gprime)/math.sqrt(6),np.linalg.norm(gloss)/math.sqrt(6)])
    if not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('Unresolved own-channel RMS, no weight floor/fallback')
    row_scale=np.repeat(sigma,6); y=np.concatenate((gprime,gloss))
    weighted_a=a/row_scale[:,None];weighted_y=y/row_scale
    column_scale=np.linalg.norm(weighted_a,axis=0)
    if np.linalg.matrix_rank(weighted_a)!=8 or np.any(column_scale<=0):
        raise ValueError('Unresolved weighted Maxwell design; no fallback')
    scaled,resnorm=nnls(weighted_a/column_scale,weighted_y,maxiter=600)
    coefficients=scaled/column_scale; predicted=at@coefficients
    residual=weighted_a@coefficients-weighted_y
    gradient=weighted_a.T@residual
    if not np.isfinite(predicted).all() or np.any(predicted<=0):
        raise ValueError('Nonpositive weighted Maxwell target point, no floor')
    return {'model':'classical_fixed6tau_channel_RMS_relative_Maxwell_NNLS',
            'prediction_kind':'point_only_no_stochastic_law',
            'channel_RMS_Pa':sigma.tolist(),'fixed_relaxation_times_s':tau.tolist(),
            'coefficient_order':'G_infinity, six G_modes, eta_solvent',
            'coefficients_Pa_and_Pa_s':coefficients.tolist(),
            'predicted_Gprime_Pa':predicted[:7].tolist(),
            'predicted_Gdoubleprime_Pa':predicted[7:].tolist(),
            'own_weighted_residual_sum_squares':float(residual@residual),
            'nnls_reported_weighted_residual_norm':float(resnorm),
            'KKT_gradient_weighted_objective':gradient.tolist(),
            'observed_frequency_budget':6,'observed_scalar_response_budget':12,
            'target_scalar_response_dimension':14,'target_responses_received':0,
            'loss':'sumchannel sumown6 ((G-Pa-model-Pa)/own-channel-RMS-Pa)^2',
            'weights_fitted_or_selected':False,'tau_fitted':False,
            'probabilistic_scores_supported':False}


def _attempt(call):
    try:
        return {'status':'OK','result':call()}
    except Exception as exc:
        return {'status':'FAIL','result':None,
                'failure':{'exception_type':type(exc).__name__,'message':str(exc)}}


def _point_admission(component):
    if component['status']=='OK':
        result=component['result']
        point=np.concatenate((result['predicted_Gprime_Pa'],result['predicted_Gdoubleprime_Pa']))
        if not np.isfinite(point).all() or np.any(point<=0):
            # Keep successful NNLS coefficients/forward result even when its
            # zero point cannot define the registered logarithmic comparison.
            component={**component,'status':'FAIL','fit_status':'OK',
                'failure':{'exception_type':'LogPointAdmissionFailure',
                           'message':'Nonpositive/nonfinite point; no logarithmic floor or rescue'}}
    return component


def assemble_components(components):
    """Pure failure isolation: no failed teacher may erase a valid full GP."""
    gp=components['gp']; weighted=components['weighted_maxwell']; raw=components['raw_Pa_maxwell']
    laws={};channels=None;m=None;covariance=None;point=None
    if gp['status']=='OK':
        channels=[c['result'] for c in gp['channels']]
        m=np.concatenate([c['mean'] for c in channels])
        covariance=block_diag(*[np.array(c['covariance']) for c in channels])
        np.linalg.cholesky(covariance)
        laws[METHODS[0]]={'distribution':'multivariate_gaussian_log_modulus',
                         'mean':m.tolist(),'covariance':covariance.tolist()}
        laws[METHODS[4]]={'distribution':'multivariate_student4_log_modulus',
                         'location':m.tolist(),'scale_matrix':(.5*covariance).tolist(),
                         'predictive_covariance':covariance.tolist(),'df':4}
    if weighted['status']=='OK':
        result=weighted['result']
        point=np.concatenate((result['predicted_Gprime_Pa'],result['predicted_Gdoubleprime_Pa']))
        if gp['status']=='OK':
            laws=assemble_laws(m,covariance,point)
    dependencies={METHODS[0]:['gp'],METHODS[1]:['gp','weighted_maxwell'],
        METHODS[2]:['gp','weighted_maxwell'],METHODS[3]:['gp','weighted_maxwell'],
        METHODS[4]:['gp'],
        POINT_METHODS[0]:['weighted_maxwell'],POINT_METHODS[1]:['raw_Pa_maxwell']}
    availability={}
    for method,needed in dependencies.items():
        missing=[name for name in needed if components[name]['status']!='OK']
        availability[method]={'available':not missing,'missing_or_failed_components':missing}
    count=sum(v['available'] for v in availability.values())
    return {'status':'OK' if count==len(ALL_METHODS) else ('PARTIAL' if count else 'FAIL'),
            'components':components,'method_availability':availability,'laws':laws,
            'gp_channels':channels,'maxwell_point':weighted['result'],
            'raw_Pa_maxwell_point':raw['result'],
            'gp_mean':None if m is None else m.tolist(),
            'full_residual_covariance':None if covariance is None else covariance.tolist(),
            'maxwell_log_location':None if point is None else np.log(point).tolist()}


def build_context(frequency_Hz, observed_indices, target_indices, observed_moduli_Pa):
    f = np.asarray(frequency_Hz,dtype=float); o = np.asarray(observed_indices,dtype=int)
    target = np.asarray(target_indices,dtype=int); observed = np.asarray(observed_moduli_Pa,dtype=float)
    if f.shape!=(13,) or o.shape!=(6,) or target.shape!=(7,) or observed.shape!=(2,6):
        raise ValueError('Native13 frequency6/7 partition with12 scalar observations required')
    if (len(np.unique(f))!=13 or np.any(f<=0) or set(o)&set(target) or
            set(o)|set(target)!=set(range(13))):
        raise ValueError('Positive native inputs and exact disjoint partition required')
    x = np.log(2*math.pi*f)
    gp = pinned('generalization/anchor_gp.py')
    def one_channel(channel):
        values=observed[channel]
        if not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('Nonpositive/nonfinite own channel: no log floor/deletion')
        return gp.fit_predict(x[o],np.log(values),x[target])
    channels=[_attempt(lambda channel=channel:one_channel(channel)) for channel in range(2)]
    gp_status='OK' if all(c['status']=='OK' for c in channels) else (
        'PARTIAL' if any(c['status']=='OK' for c in channels) else 'FAIL')
    weighted=_point_admission(_attempt(lambda:weighted_maxwell(f[o],observed[0],observed[1],f[target],f)))
    raw=_point_admission(_attempt(lambda:pinned('generalization/generalized_maxwell_nnls.py').fit_predict(
        f[o],observed[0],observed[1],f[target],f)))
    result=assemble_components({'gp':{'status':gp_status,'channels':channels},
                                'weighted_maxwell':weighted,'raw_Pa_maxwell':raw})
    return {**result,
            'target_order':'Gprime seven native targets then Gdoubleprime same seven',
            'cross_channel_residual_covariance':'zero by construction; not learned coupling',
            'observed_frequency_budget':6,'observed_scalar_budget':12,
            'target_scalar_dimension':14,'target_response_values_received':0,
            'generic_GP_semantic_adapter':{'x':'ln(omega/(1 rad/s))','y':'ln(G/(1 Pa))',
                'legacy_log_gamma_or_log_eta_diagnostic_key_names':'mathematical coordinate/response labels only; interpret all numbers with this wrapper units',
                'each_channel_receives_only_own6':True},
            'old_viscosity_parameters_loaded':False,'learned_transport_fraction':False,
            'full_residual_covariance_retained':True,
            'generated_residual_draws_guaranteed_Maxwell_or_passive':False}


def score_context(payload, target_moduli_Pa):
    raw = np.asarray(target_moduli_Pa,dtype=float)
    if raw.shape!=(2,7) or not np.isfinite(raw).all() or np.any(raw<=0):
        raise ValueError('Nonpositive/nonfinite target modulus; retain failed context without score/floor/deletion')
    y = np.log(raw.reshape(14)); scores=[]
    for method in METHODS:
        if not payload['method_availability'][method]['available']:
            continue
        law=payload['laws'][method]
        if law['distribution']=='multivariate_gaussian_log_modulus':
            mean=np.array(law['mean']);c=np.array(law['covariance']);sd=np.sqrt(np.diag(c))
            z=(y-mean)/sd
            crps=sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))
            logp=float(multivariate_normal.logpdf(y,mean=mean,cov=c))
            quantiles=mean+norm.ppf([.05,.5,.95])[:,None]*sd
        else:
            mean=np.array(law['location']);shape=np.array(law['scale_matrix']);sd=np.sqrt(np.diag(shape))
            z=(y-mean)/sd
            crps=sd*(z*(2*stdtr(4,z)-1)+2*(4+z*z)*t.pdf(z,4)/3-15*math.pi/64)
            logp=float(multivariate_t.logpdf(y,loc=mean,shape=shape,df=4))
            quantiles=mean+t.ppf([.05,.5,.95],4)[:,None]*sd
        channel=[]
        for i,name in enumerate(('Gprime','Gdoubleprime')):
            sl=slice(7*i,7*(i+1))
            channel.append({'channel':name,'log_rmse':float(np.sqrt(np.mean((y[sl]-mean[sl])**2))),
                'log_crps':float(crps[sl].mean()),
                'coverage90':float(np.mean((y[sl]>=quantiles[0,sl])&(y[sl]<=quantiles[2,sl]))),
                'log_width90':float(np.mean(quantiles[2,sl]-quantiles[0,sl]))})
        scores.append({'method':method,'joint_nll_per_dim':-logp/14,
            'log_crps':float(crps.mean()),'pointwise_log_crps':crps.tolist(),
            'log_rmse':float(np.sqrt(np.mean((y-mean)**2))),
            'coverage90':float(np.mean((y>=quantiles[0])&(y<=quantiles[2]))),
            'log_width90':float(np.mean(quantiles[2]-quantiles[0])),
            'channels':channel,'Pa_point_semantics':'exp(log-location) is median, no Student arithmetic mean'})
    points=[]
    for key,method in [('maxwell_point','channel_relative_Maxwell_NNLS_point'),
                       ('raw_Pa_maxwell_point','raw_Pa_Maxwell_NNLS_point')]:
        if not payload['method_availability'][method]['available']:
            continue
        maxwell=payload[key]
        m=np.log(np.concatenate((maxwell['predicted_Gprime_Pa'],maxwell['predicted_Gdoubleprime_Pa'])))
        points.append({'method':method,'log_rmse':float(np.sqrt(np.mean((y-m)**2))),
               'channels':[{'channel':name,'log_rmse':float(np.sqrt(np.mean((y[7*i:7*(i+1)]-m[7*i:7*(i+1)])**2)))}
                           for i,name in enumerate(('Gprime','Gdoubleprime'))],
               'probability_scores_defined':False})
    return scores,points
