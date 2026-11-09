"""Independent density/CDF-domain mathematics for fixed whole-curve mixtures.

SOURCE ONLY until root audit registration. No SUT imports, optimization, raw
data access, fitting, source selection, or generation of material measurements.
Gaussian and t4 components have identical full covariance C. Student scale
matrices are C/2; moment matched t4 uses (C+dd'/4)/2.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import integrate, linalg, special, stats


def moments(m_gp,m_teacher,C):
    mg=np.asarray(m_gp,dtype=float);mt=np.asarray(m_teacher,dtype=float);cov=np.asarray(C,dtype=float)
    if mg.ndim!=1 or mt.shape!=mg.shape or cov.shape!=(mg.size,mg.size):
        raise ValueError('Dimension mismatch')
    if not np.isfinite(mg).all() or not np.isfinite(mt).all() or not np.isfinite(cov).all():
        raise ValueError('Nonfinite input')
    if not np.allclose(cov,cov.T,rtol=1e-13,atol=1e-15):raise ValueError('Non symmetric covariance')
    np.linalg.cholesky(cov)
    delta=mt-mg
    return (mg+mt)/2,cov+np.outer(delta,delta)/4


def component_logpdf(y,m,C,tail):
    """C denotes full covariance for BOTH component tails."""
    y=np.asarray(y,float);m=np.asarray(m,float);C=np.asarray(C,float);d=len(m)
    scale=C if tail=='gaussian' else C/2 if tail=='student4' else None
    if scale is None:raise ValueError('Fixed gaussian/student4 family only')
    L=np.linalg.cholesky(scale);z=linalg.solve_triangular(L,y-m,lower=True);q=float(z@z)
    half_logdet=float(np.log(np.diag(L)).sum())
    if tail=='gaussian':return -.5*q-half_logdet-.5*d*math.log(2*math.pi)
    return float(special.gammaln((4+d)/2)-special.gammaln(2)-half_logdet-
                 .5*d*math.log(4*math.pi)-.5*(4+d)*math.log1p(q/4))


def mixture_logpdf(y,m_gp,m_teacher,C,tail):
    # ONE whole-vector mixture indicator: logsumexp of full joint densities.
    a=component_logpdf(y,m_gp,C,tail);b=component_logpdf(y,m_teacher,C,tail)
    return float(np.logaddexp(a,b)-math.log(2))


def moment_matched_logpdf(y,m_gp,m_teacher,C,tail):
    mean,total=moments(m_gp,m_teacher,C)
    return component_logpdf(y,mean,total,tail)


def absolute_normal(offset,sd):
    z=offset/sd
    return 2*sd*np.exp(-z*z/2)/math.sqrt(2*math.pi)+offset*special.erf(z/math.sqrt(2))


def gaussian_mixture_crps_energy(y,mg,mt,sd):
    """Analytic energy expression, used only to cross-check CDF-domain oracle."""
    return float(.5*(absolute_normal(y-mg,sd)+absolute_normal(y-mt,sd))-
                 .25*(absolute_normal(0,math.sqrt(2)*sd)+absolute_normal(mt-mg,math.sqrt(2)*sd)))


def _split_integral(fun,lo,hi,breakpoints):
    knots=[lo]+sorted({float(v) for v in breakpoints if lo<v<hi})+[hi]
    values=[];errors=[]
    for a,b in zip(knots,knots[1:]):
        value,error=integrate.quad(fun,a,b,epsabs=3e-10,epsrel=3e-10,limit=250)
        values.append(value);errors.append(error)
    return math.fsum(values),math.fsum(errors)


def marginal_crps_cdf(y,mg,mt,variance,tail,representation):
    """Independent response-CDF-squared integral, including both infinite tails.

    Standardize by the component distribution's scale, then split around its
    two centers and the observation. No sampled targets/quantile truncation.
    """
    if variance<=0:raise ValueError('Positive variance required')
    middle=(mg+mt)/2;delta=mt-mg
    total=variance+delta*delta/4
    if representation=='density_mixture':
        scale=math.sqrt(variance if tail=='gaussian' else variance/2)
        a=delta/(2*scale)
        if tail=='gaussian':
            cdf=lambda z:.5*special.ndtr(z-a)+.5*special.ndtr(z+a)
            sf=lambda z:.5*special.ndtr(a-z)+.5*special.ndtr(-z-a)
        elif tail=='student4':
            cdf=lambda z:.5*special.stdtr(4,z-a)+.5*special.stdtr(4,z+a)
            sf=lambda z:.5*special.stdtr(4,a-z)+.5*special.stdtr(4,-z-a)
        else:raise ValueError('Tail')
        # Explicit local breaks resolve narrow transitions even when means are
        # very far apart. Otherwise adaptive Gauss nodes on a huge plateau
        # could miss the short transition near an integration endpoint.
        splits=[0]+[center+offset for center in (-abs(a),abs(a))
                    for offset in (-64,-16,-4,-1,0,1,4,16,64)]
    elif representation=='moment_matched':
        scale=math.sqrt(total if tail=='gaussian' else total/2)
        if tail=='gaussian':cdf=special.ndtr;sf=lambda z:special.ndtr(-z)
        elif tail=='student4':cdf=lambda z:special.stdtr(4,z);sf=lambda z:special.stdtr(4,-z)
        else:raise ValueError('Tail')
        splits=[-64,-16,-4,-1,0,1,4,16,64]
    else:raise ValueError('Representation')
    z=(y-middle)/scale
    left,le=_split_integral(lambda t:float(cdf(t)**2),-np.inf,z,splits)
    right,re=_split_integral(lambda t:float(sf(t)**2),z,np.inf,splits)
    return scale*(left+right),scale*(le+re)


def marginal_cdf(y,mg,mt,variance,tail,representation):
    mid=(mg+mt)/2;delta=mt-mg;total=variance+delta*delta/4
    if tail not in ('gaussian','student4'):raise ValueError('Tail')
    cdf=special.ndtr if tail=='gaussian' else lambda z:special.stdtr(4,z)
    if representation=='density_mixture':
        s=math.sqrt(variance if tail=='gaussian' else variance/2)
        return float(.5*cdf((y-mg)/s)+.5*cdf((y-mt)/s))
    if representation=='moment_matched':
        s=math.sqrt(total if tail=='gaussian' else total/2)
        return float(cdf((y-mid)/s))
    raise ValueError('Representation')


def matched_marginal_quantile(p,mg,mt,variance,tail):
    if not 0<p<1:raise ValueError('Open probability interval only')
    m=(mg+mt)/2;v=variance+(mt-mg)**2/4
    return float(m+math.sqrt(v if tail=='gaussian' else v/2)*
                 (stats.norm.ppf(p) if tail=='gaussian' else stats.t.ppf(p,4)))
