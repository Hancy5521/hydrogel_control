"""Pure low-dimensional convex native-CRPS calibration.

Caller supplies only legally completed forecast/true-Y pairs. No file IO,
forest fitting, disclosure oracle, feature discovery data or synthetic labels.
"""
import math
import numpy as np
from empirical_native import P, row_terms, crps, canonical

MAX_ITERATIONS=200
KKT_TOLERANCE=1e-9
ARMIJO_C1=1e-4
BACKTRACK_FACTOR=.5
BACKTRACK_MAX_STEPS=80
CONTEXT_DEGENERACY_TOLERANCE=1e-12


def make_statistics(laws, targets, g=None):
    laws=list(laws); targets=list(targets); n=len(laws)
    if n==0 or len(targets)!=n:
        raise ValueError('nonempty matched completed calibration records required')
    context=g is not None
    if context:
        gv=np.asarray(g,dtype=float)
        if gv.shape!=(n,) or not np.isfinite(gv).all() or (gv<0).any() or (gv>1).any():
            raise ValueError('finite matched context values in[0,1] required')
        mean_g=float(math.fsum(gv.tolist())/n)
        degenerate=(float(np.ptp(gv))<=CONTEXT_DEGENERACY_TOLERANCE or mean_g in (0.,1.))
    else:
        gv=None; mean_g=None; degenerate=False
    blocks=2 if context and not degenerate else 1
    Q=np.zeros((4*blocks,4*blocks)); b=np.zeros(4*blocks)
    constants=[]; risks=[]; reference_losses=[]; all_singleton=True
    for i,(law,y) in enumerate(zip(laws,targets)):
        term=row_terms(law,y)
        all_singleton=all_singleton and len(canonical(law)['support'])==1
        if blocks==1:
            Q+=term['Q']; b+=term['b']
        else:
            A=np.concatenate(((1-gv[i])*np.eye(4),gv[i]*np.eye(4)),axis=1)
            Q+=A.T@term['Q']@A; b+=A.T@term['b']
        constants.append(term['c']); risks.append(term['reference_risk'])
        reference_losses.append(crps(law,y))
    mean_risk=float(math.fsum(risks)/n)
    if mean_risk==0 and not all_singleton:
        raise ArithmeticError('zero floating risk for nonsingleton reference; cannot invent regularization')
    lam=(10./n)*mean_risk
    if mean_risk>0 and (not math.isfinite(lam) or lam<=0):
        raise ArithmeticError('regularization not representable')
    weights=np.asarray([1.] if blocks==1 else [1-mean_g,mean_g])
    return {'Q':Q/n,'b':b/n,'c':float(math.fsum(constants)/n),'n':n,
            'blocks':blocks,'mean_g':mean_g,'mean_reference_risk':mean_risk,
            'lambda':lam,'endpoint_KL_weights':weights,
            'reference_mean_CRPS':float(math.fsum(reference_losses)/n),
            'context_requested':context,'context_degenerate':degenerate,
            'all_reference_laws_singleton':all_singleton,
            'constant_context_rule':'fallback global;theta0=theta1;no singular two-block fit',
            'arithmetic':'floating native quadratic statistics,not certified interval arithmetic'}


def objective_gradient_hessian(stats, flat_theta):
    q=np.asarray(flat_theta,dtype=float)
    blocks=int(stats['blocks']); d=4*blocks
    if q.shape!=(d,) or not np.isfinite(q).all() or (q<=0).any():
        raise ValueError('finite positive direct simplex coordinates required')
    Q=np.asarray(stats['Q'],dtype=float); b=np.asarray(stats['b'],dtype=float)
    if Q.shape!=(d,d) or b.shape!=(d,):
        raise ValueError('inconsistent statistics')
    lam=float(stats['lambda']); weights=np.asarray(stats['endpoint_KL_weights'])
    penalty=0.; grad=2*(Q@q-b); H=2*Q.copy()
    for r in range(blocks):
        sl=slice(4*r,4*r+4); z=q[sl]; coefficient=lam*weights[r]
        penalty+=coefficient*float(np.sum(P*(np.log(P)-np.log(z))))
        grad[sl]-=coefficient*P/z
        H[sl,sl]+=np.diag(coefficient*P/(z*z))
    value=float(q@Q@q-2*b@q+stats['c']+penalty)
    if not math.isfinite(value) or not np.isfinite(grad).all() or not np.isfinite(H).all():
        raise ArithmeticError('objective/derivative not representable')
    return value,grad,H


def _projected_gradient(grad, blocks):
    parts=[]
    for r in range(blocks):
        z=grad[4*r:4*r+4]
        parts.extend((z-float(np.mean(z))).tolist())
    return float(np.max(np.abs(parts)))


def _reduced_basis(blocks):
    Z=np.zeros((4*blocks,3*blocks))
    for r in range(blocks):
        Z[4*r:4*r+3,3*r:3*r+3]=np.eye(3)
        Z[4*r+3,3*r:3*r+3]=-1.
    return Z


def _output(stats, q, status, iterations, history, reason=None):
    blocks=int(stats['blocks']); value,gradient,_=objective_gradient_hessian(stats,q)
    theta0=q[:4].tolist(); theta1=q[4:].tolist() if blocks==2 else theta0.copy()
    return {'status':status,'kind':'context' if blocks==2 else 'global',
            'theta0':theta0,'theta1':theta1,
            'diagnostics':{'iterations':iterations,'objective_native':value,
            'reference_mean_CRPS_native':stats['reference_mean_CRPS'],
            'projected_KKT_residual':_projected_gradient(gradient,blocks),
            'simplex_sum_errors':[float(math.fsum(q[4*r:4*r+4])-1.) for r in range(blocks)],
            'minimum_parameter':float(np.min(q)),'lambda_native':stats['lambda'],
            'mean_reference_Bayes_CRPS_risk_native':stats['mean_reference_risk'],
            'completed_calibration_records':stats['n'],
            'mean_g':stats['mean_g'],'context_degenerate':stats['context_degenerate'],
            'all_reference_laws_singleton':stats['all_reference_laws_singleton'],
            'identity_fallback_reason':reason,'line_search_history':history,
            'probabilistic_scope':'convex penalized CRPS estimator,not ordinary generative Bayes posterior'}}


def fit_simplex(stats):
    """Damped Newton in an equality-constraint nullspace, positive Armijo steps.

    A failed optimizer is explicitly returned as failed+identity,never as a fit.
    No parameter clipping, probability floor,entropy reserve or softmax.
    """
    blocks=int(stats['blocks']); q=np.tile(P,blocks); history=[]
    if stats['all_reference_laws_singleton']:
        if stats['lambda']!=0:
            raise ValueError('singleton regularization must be exactly0')
        return _output(stats,q,'IDENTITY_SINGLETON_NO_TRANSPORT_INFORMATION',0,history,
                       'all reference CDFs singleton;T composed with them is unchanged')
    if stats['lambda']<=0:
        raise ValueError('positive risk requires positive regularization')
    Z=_reduced_basis(blocks)
    for it in range(MAX_ITERATIONS+1):
        value,grad,H=objective_gradient_hessian(stats,q)
        residual=_projected_gradient(grad,blocks)
        if residual<=KKT_TOLERANCE:
            return _output(stats,q,'CONVERGED_KKT',it,history)
        if it==MAX_ITERATIONS:
            break
        try:
            d=Z@np.linalg.solve(Z.T@H@Z,-Z.T@grad)
        except np.linalg.LinAlgError:
            return _output(stats,np.tile(P,blocks),'FAILED_RETURNED_IDENTITY',it,history,
                           'reduced Hessian solve failed;no ridge added')
        slope=float(grad@d)
        if not np.isfinite(d).all() or not math.isfinite(slope) or slope>=0:
            return _output(stats,np.tile(P,blocks),'FAILED_RETURNED_IDENTITY',it,history,
                           'Newton direction not finite strict descent')
        negative=d<0
        alpha=min(1.,.99*float(np.min(-q[negative]/d[negative]))) if negative.any() else 1.
        accepted=False
        for bt in range(BACKTRACK_MAX_STEPS):
            trial=q+alpha*d
            # Equality retraction only;does not clamp any probability.
            for r in range(blocks):
                trial[4*r+3]=1.-math.fsum(trial[4*r:4*r+3].tolist())
            if np.isfinite(trial).all() and (trial>0).all():
                new_value,_,_=objective_gradient_hessian(stats,trial)
                if new_value<=value+ARMIJO_C1*alpha*slope:
                    accepted=True; break
            alpha*=BACKTRACK_FACTOR
        history.append({'iteration':it,'objective_native':value,'KKT_residual':residual,
                        'step':alpha,'backtracks':bt,'accepted':accepted})
        if not accepted:
            return _output(stats,np.tile(P,blocks),'FAILED_RETURNED_IDENTITY',it+1,history,
                           'positive Armijo step not found')
        q=trial
    return _output(stats,np.tile(P,blocks),'FAILED_RETURNED_IDENTITY',MAX_ITERATIONS,history,
                   'maximum iterations without KKT clearance')


def fit_global(laws, targets):
    return fit_simplex(make_statistics(laws,targets))


def fit_context(laws, targets, g):
    return fit_simplex(make_statistics(laws,targets,g))
