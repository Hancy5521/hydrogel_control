"""Task readouts over one fixed full NIG-BMA posterior; no oracle or model fitting.

The three objectives are native cross-target CRPS information value, local total
native CRPS risk, and model/label mutual information. They share all components
and exact predictive evidence; only the acquisition objective changes. Finite
quadrature and timing diagnostics remain explicit, never a success certificate.
"""
import math
import time
import numpy as np
from scipy import special,stats
import relation_posterior as posterior
import risk_field as field


def _marginals(ensemble,context):
    result=[]
    for model in ensemble['models']:
        phi=posterior.feature_map(model['id'],context,ensemble.get('expansion_transform'))
        result.append({'id':model['id'],**posterior.scalar_law(model['state'],phi)})
    return result


def _target_domain(laws,transform):
    center,sd,s0=field.checked_transform(transform)
    lower=min(c['mu']-field.LV*c['scale'] for c in laws)
    upper=max(c['mu']+field.LV*c['scale'] for c in laws)
    split=sorted(set([lower,upper]+[x for x in (0.,-center/sd) if lower<x<upper]))
    return center,sd,s0,lower,upper,split


def marginal_native_risk(ensemble,context,transform,count):
    """Bayes expected CRPS at this queried location: integral F(1-F) d nativeY."""
    laws=_marginals(ensemble,context);logs=np.asarray(posterior.normalize_logweights(ensemble['log_weights']));weights=np.exp(logs)
    center,sd,s0,lower,upper,split=_target_domain(laws,transform);parts=[]
    for low,high in zip(split[:-1],split[1:]):
        z,wt=field.interval(count,low,high)
        cdf=sum(w*special.stdtr(c['nu'],(z-c['mu'])/c['scale']) for w,c in zip(weights,laws))
        sf=sum(w*special.stdtr(c['nu'],-(z-c['mu'])/c['scale']) for w,c in zip(weights,laws))
        value=float((wt*s0*sd*field.derivative(center+sd*z))@(cdf*sf))
        parts.append(value)
    # The same component tail envelope bounds mixture F(1-F); no query tail here.
    tails=[]
    for w,c in zip(weights,laws):
        nu=c['nu'];mu=center+sd*c['mu'];scale=sd*c['scale']
        pdf=float(stats.t.pdf(field.LV,nu));sf=float(special.stdtr(nu,-field.LV))
        i0=(nu+field.LV**2)/(nu-1)*pdf
        i1=.5*(nu*sf+field.LV*(nu+field.LV**2)*pdf)/(nu-2)
        tails.append(float(w)*2*s0*scale*((1+.5*abs(mu))*i0+.5*scale*i1))
    value=math.fsum(parts)
    if not math.isfinite(value) or value<0:raise ArithmeticError('nonnegative finite native CRPS risk required')
    return {'value':value,'target_modeled_union':[lower,upper],'outer_tail_bound_ideal_arithmetic':math.fsum(tails),
        'query_truncation':False,'finite_quadrature_not_certified':True}


def model_information(ensemble,context,count):
    """I(H;future Y at known context), one-dimensional full mixture t quadrature.

    Native Jacobians cancel in the likelihood ratio; the exact objective is
    invariant under the common invertible response map. Numerical cuts are finite.
    """
    laws=_marginals(ensemble,context);logs=np.asarray(posterior.normalize_logweights(ensemble['log_weights']));weights=np.exp(logs)
    pieces=[];mass=[]
    for j,c in enumerate(laws):
        nu=c['nu'];theta,wt=field.interval(count,math.atan(-field.LZ/math.sqrt(nu)),math.atan(field.LZ/math.sqrt(nu)))
        density=np.exp((nu-1)*np.log(np.cos(theta))-special.betaln(.5,nu/2));query=c['mu']+c['scale']*math.sqrt(nu)*np.tan(theta)
        logpdf=[]
        for component in laws:
            v=component['nu'];z=(query-component['mu'])/component['scale']
            logpdf.append(special.gammaln((v+1)/2)-special.gammaln(v/2)-.5*math.log(v*math.pi)
                -math.log(component['scale'])-(v+1)/2*np.log1p(z*z/v))
        logpdf=np.column_stack(logpdf);logmixture=special.logsumexp(logpdf+logs[None,:],axis=1)
        pieces.append(float(weights[j])*float((wt*density)@(logpdf[:,j]-logmixture)))
        mass.append(float(weights[j])*2*float(special.stdtr(nu,-field.LZ)))
    value=math.fsum(pieces)
    if not math.isfinite(value):raise ArithmeticError('finite model-information integral required')
    return {'value':value,'component_integrals_signed':pieces,'omitted_query_probability':math.fsum(mass),
        'mutual_information_tail_not_claimed_rigorously_bounded':True,'no_negative_clipping':True,
        'exact_objective':'I(H;Yq|currentdata,knownXq)','deterministic_finite_quadrature':True,
        'native_response_Jacobian_cancels':True}


def risk_values(ensemble,pool_z,terminal_z,transform,mode='full_native',count=64):
    """Return peravailable-query scores; caller fixes role-order ties and budgets.

    Full native values average over the supplied target set; never target labels.
    This wrapper intentionally calls the frozen full-mixture core unchanged.
    """
    if count not in field.LEVELS:raise ValueError('one fixed32/64/128/256 GL level required')
    pool=np.asarray(pool_z,dtype=float);targets=np.asarray(terminal_z,dtype=float)
    if pool.ndim!=2 or pool.shape[1]!=2 or not len(pool) or not np.isfinite(pool).all():raise ValueError('finite nonempty available dual-context pool required')
    if targets.ndim!=2 or targets.shape[1]!=2 or not len(targets) or not np.isfinite(targets).all():raise ValueError('finite nonempty target context set required')
    if mode not in ('full_native','marginal_native','model_MI'):raise ValueError('fixed objective mode required')
    started=time.perf_counter();values=[];rows=[];query_seconds=[]
    for q in pool:
        before=time.perf_counter()
        if mode=='full_native':
            per_target=[]
            for t in targets:
                r=field.native_value(ensemble,q,t,transform,count)
                per_target.append({'full':r['native_full_VOI'],'within':r['native_weighted_within_total'],
                    'within_components':r['native_weighted_within_components'],'signed_between':r['native_signed_between'],
                    'B_before':r['native_B_before'],'E_B_after':r['native_E_B_after'],'closure':r['decomposition_closure'],
                    'tail':r['tail']['total_value_tail_bound_ideal_arithmetic']})
            value=float(np.mean([r['full'] for r in per_target]));rows.append({'mean_full_value':value,
                'mean_within_value':float(np.mean([r['within'] for r in per_target])),
                'mean_signed_between':float(np.mean([r['signed_between'] for r in per_target])),
                'mean_tail_bound_ideal_arithmetic':float(np.mean([r['tail'] for r in per_target])),
                'maximum_absolute_closure':max(abs(r['closure']) for r in per_target),
                'per_target_values':per_target})
        elif mode=='marginal_native':
            r=marginal_native_risk(ensemble,q,transform,count);value=r['value'];rows.append(r)
        else:
            r=model_information(ensemble,q,count);value=r['value'];rows.append(r)
        values.append(value);query_seconds.append(time.perf_counter()-before)
    return {'values':values,'diagnostics':{'mode':mode,'nodes_per_segment':count,'available_queries':len(pool),
        'target_contexts':len(targets),'target_labels_read':0,'full_pair_evaluations':len(pool)*len(targets) if mode=='full_native' else 0,
        'query_elapsed_seconds':query_seconds,'elapsed_seconds':time.perf_counter()-started,'per_query':rows,
        'mean_over_targets_not_sum':mode=='full_native','no_weight_floor_or_signed_between_clipping':True,
        'model_conditional_values_not_guaranteed_actual_gain':True,'deterministic_no_MC_catalog':True}}
