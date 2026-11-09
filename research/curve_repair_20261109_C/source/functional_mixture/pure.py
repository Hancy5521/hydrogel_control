"""SOURCE ONLY: classical whole-curve mixtures and same-moment controls.

No fitter, source reader, learned weights, scale tuning or parameter search.
The mixture chooses one branch for the entire fourteen-dimensional curve.
Original complete residual C is retained inside each component. Disagreement
adds .25*d*d' to total covariance; it does not replace C by a low-rank law.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm,t,multivariate_normal,multivariate_t

METHODS=('whole_curve_gaussian_density_mixture','whole_curve_student4_density_mixture',
         'moment_matched_gaussian','moment_matched_student4')
DIM=14


def _parameters(gp_mean,maxwell_mean,covariance):
    mg,mm,c=map(lambda v:np.array(v,dtype=float,copy=True),(gp_mean,maxwell_mean,covariance))
    if mg.shape!=(DIM,) or mm.shape!=(DIM,) or c.shape!=(DIM,DIM):
        raise ValueError('Fixed complete fourteen-dimensional endpoint laws required')
    if not all(np.isfinite(v).all() for v in (mg,mm,c)) or not np.allclose(c,c.T,rtol=1e-12,atol=1e-12):
        raise ValueError('Finite means and symmetric complete covariance required')
    np.linalg.cholesky(c)
    return mg,mm,c


def build_laws(gp_mean,maxwell_mean,covariance):
    mg,mm,c=_parameters(gp_mean,maxwell_mean,covariance)
    difference=mm-mg;half=.5*(mg+mm);total=c+.25*np.outer(difference,difference)
    np.linalg.cholesky(total)
    return {
        METHODS[0]:{'distribution':'whole_curve_equal_gaussian_mixture',
            'component_locations':[mg.tolist(),mm.tolist()],'component_covariance':c.tolist(),
            'weights':[.5,.5],'mean':half.tolist(),'predictive_covariance':total.tolist(),
            'branch_scope':'one Bernoulli for whole14D curve, never per column'},
        METHODS[1]:{'distribution':'whole_curve_equal_student4_mixture','df':4,
            'component_locations':[mg.tolist(),mm.tolist()],'component_scale_matrix':(.5*c).tolist(),
            'component_covariance':c.tolist(),'weights':[.5,.5],'mean':half.tolist(),
            'predictive_covariance':total.tolist(),'branch_scope':'one Bernoulli for whole14D curve, never per column'},
        METHODS[2]:{'distribution':'moment_matched_gaussian','mean':half.tolist(),
            'covariance':total.tolist()},
        METHODS[3]:{'distribution':'moment_matched_student4','df':4,'location':half.tolist(),
            'scale_matrix':(.5*total).tolist(),'predictive_covariance':total.tolist()}}


def logpdf(law,values):
    y=np.asarray(values,dtype=float)
    if y.shape!=(DIM,) or not np.isfinite(y).all():raise ValueError('Fourteen finite log-moduli required')
    kind=law['distribution']
    if kind=='whole_curve_equal_gaussian_mixture':
        logs=[multivariate_normal.logpdf(y,mean=m,cov=law['component_covariance']) for m in law['component_locations']]
        result=logsumexp(logs)-math.log(2)
    elif kind=='whole_curve_equal_student4_mixture':
        logs=[multivariate_t.logpdf(y,loc=m,shape=law['component_scale_matrix'],df=4) for m in law['component_locations']]
        result=logsumexp(logs)-math.log(2)
    elif kind=='moment_matched_gaussian':
        result=multivariate_normal.logpdf(y,mean=law['mean'],cov=law['covariance'])
    elif kind=='moment_matched_student4':
        result=multivariate_t.logpdf(y,loc=law['location'],shape=law['scale_matrix'],df=4)
    else:raise ValueError('Unregistered full-curve law')
    if not np.isfinite(result):raise ArithmeticError('Nonfinite complete density')
    return float(result)


def _abs_normal(z):return z*(2*norm.cdf(z)-1)+2*norm.pdf(z)
def _abs_student(z):return z*(2*t.cdf(z,4)-1)+2*(4+z*z)*t.pdf(z,4)/3


def _mixture_standard_quantile(probability,half_separation,family):
    """Unit component-scale, symmetric mixture at +/-a; no probability clipping."""
    p=float(probability);a=abs(float(half_separation))
    if not 0<p<1:raise ValueError('Interior probability required')
    if p>.5:return -_mixture_standard_quantile(1-p,a,family)
    if p==.5:return 0.
    distribution=norm if family=='gaussian' else t
    q=float(distribution.ppf(p) if family=='gaussian' else distribution.ppf(p,4))
    if a==0:return q
    def cdf(v):
        if family=='gaussian':return .5*(norm.cdf(v+a)+norm.cdf(v-a))
        return .5*(t.cdf(v+a,4)+t.cdf(v-a,4))
    return float(brentq(lambda v:cdf(v)-p,q-a-1e-11,q+a+1e-11,xtol=1e-12,rtol=1e-12,maxiter=200))


def _student_mixture_half_gini(scale,separation):
    a=abs(float(separation))/scale
    # Cross energy E|m1+s*T1-m2-s*T2| is one Student-variable integral
    # of the analytic absolute-deviation function. Symmetry maps it to
    # one positive infinite tail; there is no CRPS quantile inversion.
    value,error=quad(lambda v:t.pdf(v,4)*(_abs_student(v+a)+_abs_student(a-v)),
                     0.,np.inf,epsabs=1e-10,epsrel=2e-9,limit=180)
    cross=scale*value
    # .5 E|mixture-mixture'| = .5*s*k4+.25*cross for equal scales,
    # where k4=.5 E|T4-T4'|=15*pi/64.
    return .5*scale*15*math.pi/64+.25*cross,.25*scale*error


def marginal_quantiles(law,probabilities=(.05,.5,.95)):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or np.any(p<=0) or np.any(p>=1):raise ValueError('Interior quantile probabilities required')
    kind=law['distribution'];out=np.empty((len(p),DIM))
    if kind.startswith('whole_curve_equal_'):
        mg,mm=np.asarray(law['component_locations']);center=.5*(mg+mm)
        family='gaussian' if kind=='whole_curve_equal_gaussian_mixture' else 'student'
        matrix=law['component_covariance'] if family=='gaussian' else law['component_scale_matrix']
        scale=np.sqrt(np.diag(matrix))
        for i,prob in enumerate(p):
            for j in range(DIM):
                out[i,j]=center[j]+scale[j]*_mixture_standard_quantile(prob,(mm[j]-mg[j])/(2*scale[j]),family)
    elif kind=='moment_matched_gaussian':
        out=np.asarray(law['mean'])+norm.ppf(p)[:,None]*np.sqrt(np.diag(law['covariance']))
    elif kind=='moment_matched_student4':
        out=np.asarray(law['location'])+t.ppf(p,4)[:,None]*np.sqrt(np.diag(law['scale_matrix']))
    else:raise ValueError('Unregistered quantile family')
    if not np.isfinite(out).all():raise ArithmeticError('Unrepresentable marginal quantile')
    return out


def marginal_crps(law,values):
    y=np.asarray(values,dtype=float);kind=law['distribution'];errors=[]
    if kind.startswith('whole_curve_equal_'):
        mg,mm=np.asarray(law['component_locations'])
        if kind=='whole_curve_equal_gaussian_mixture':
            s=np.sqrt(np.diag(law['component_covariance']))
            cross=math.sqrt(2)*s*_abs_normal((mm-mg)/(math.sqrt(2)*s))
            value=.5*s*(_abs_normal((y-mg)/s)+_abs_normal((y-mm)/s))-.5*s/math.sqrt(math.pi)-.25*cross
        else:
            s=np.sqrt(np.diag(law['component_scale_matrix']));half_gini=[]
            for scale,difference in zip(s,mm-mg):
                value,error=_student_mixture_half_gini(float(scale),float(difference))
                half_gini.append(value);errors.append(error)
            value=.5*s*(_abs_student((y-mg)/s)+_abs_student((y-mm)/s))-np.array(half_gini)
    elif kind=='moment_matched_gaussian':
        m=np.asarray(law['mean']);s=np.sqrt(np.diag(law['covariance']));z=(y-m)/s
        value=s*(_abs_normal(z)-1/math.sqrt(math.pi))
    elif kind=='moment_matched_student4':
        m=np.asarray(law['location']);s=np.sqrt(np.diag(law['scale_matrix']));z=(y-m)/s
        value=s*(_abs_student(z)-15*math.pi/64)
    else:raise ValueError('Unregistered CRPS family')
    if not np.isfinite(value).all() or np.any(value<0):raise ArithmeticError('Invalid CRPS; no clipping')
    return np.asarray(value),max(errors) if errors else 0.


def score(law,values):
    y=np.asarray(values,dtype=float)
    if y.shape!=(DIM,) or not np.isfinite(y).all():raise ValueError('Fourteen finite log-moduli required')
    mean=np.asarray(law.get('mean',law.get('location')))
    crps,error=marginal_crps(law,y);q=marginal_quantiles(law)
    channels=[]
    for i,name in enumerate(('Gprime','Gdoubleprime')):
        sl=slice(7*i,7*(i+1));channels.append({'channel':name,
            'log_rmse':float(np.sqrt(np.mean((y[sl]-mean[sl])**2))),
            'log_crps':float(crps[sl].mean()),
            'coverage90':float(np.mean((y[sl]>=q[0,sl])&(y[sl]<=q[2,sl]))),
            'log_width90':float(np.mean(q[2,sl]-q[0,sl]))})
    return {'joint_nll_per_dim':-logpdf(law,y)/DIM,'log_crps':float(crps.mean()),
        'pointwise_log_crps':crps.tolist(),'log_rmse':float(np.sqrt(np.mean((y-mean)**2))),
        'coverage90':float(np.mean((y>=q[0])&(y<=q[2]))),'log_width90':float(np.mean(q[2]-q[0])),
        'channels':channels,'max_quantile_Gini_quadrature_error':error,
        'point_semantics':'common log-location; exp is marginal median; no Student Pa arithmetic mean'}


def sample(law,count,rng,return_latents=False):
    if not isinstance(rng,np.random.Generator) or isinstance(count,bool) or not isinstance(count,int) or count<=0:
        raise ValueError('Positive integer count and explicit Generator required')
    kind=law['distribution'];branch=None;chi=None
    if kind.startswith('whole_curve_equal_'):
        branch=rng.integers(0,2,size=count);locations=np.asarray(law['component_locations'])[branch]
        student=kind=='whole_curve_equal_student4_mixture'
        covariance=law['component_scale_matrix'] if student else law['component_covariance']
    else:
        student=kind=='moment_matched_student4';locations=np.broadcast_to(law.get('mean',law.get('location')),(count,DIM))
        covariance=law['scale_matrix'] if student else law['covariance']
    lower=np.linalg.cholesky(covariance);normal=rng.standard_normal((count,DIM))
    residual=normal@lower.T
    if student:
        chi=rng.chisquare(4,size=count);residual/=np.sqrt(chi[:,None]/4)
    draws=locations+residual
    if not np.isfinite(draws).all():raise ArithmeticError('Unrepresentable sample; no clipping')
    if return_latents:return {'draws':draws.tolist(),'whole_curve_branch':None if branch is None else branch.tolist(),
        'standard_normal':normal.tolist(),'chi_square4_per_curve':None if chi is None else chi.tolist()}
    return draws
