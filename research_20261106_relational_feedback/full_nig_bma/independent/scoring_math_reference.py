"""Direct native-y scoring for the two saved synthetic full-NIG laws; no SUT."""
import math
import numpy as np
from scipy import special,stats
from scipy.integrate import quad
from scipy.optimize import brentq
import full_native_reference as core

PROBABILITIES=(.025,.05,.1,.5,.9,.95,.975)


def law_from_fixed_proposal(proposal,observed_modeled_query=None):
    states=proposal['models'];weights=np.array(proposal['weights'])
    if observed_modeled_query is not None:
        pairs=[core.pair(s,proposal['query_context'],proposal['target_context']) for s in states]
        weights,_=core.weights_at(weights,pairs,observed_modeled_query)
        states=[core.dense_update(s,proposal['query_context'],observed_modeled_query) for s in states]
    response=proposal['response_transform'];components=[]
    for s in states:
        marginal=core.marginal(s,proposal['target_context'])
        components.append({'id':s['id'],'location':response['center']+response['sd']*marginal['mu'],
                           'scale':response['sd']*marginal['scale'],'df':marginal['nu']})
    return {'family':'mixture_student_t','coordinate':'u=YJ_half(native/s0)',
            'log_weights':np.log(weights).tolist(),'components':components,
            'native_first_scale_s0':response['s0'],'lambda':.5}


def cdf(law,y,survival=False):
    u=core.forward_yj(y/law['native_first_scale_s0']);weights=np.exp(law['log_weights'])
    return math.fsum(float(w)*float(special.stdtr(c['df'],((-1 if survival else 1)*(u-c['location']))/c['scale']))
                     for w,c in zip(weights,law['components']))


def direct_crps(law,y):
    # Native coordinates and entire real tails; no component scores or disagreement subtraction.
    points=[0.]+[law['native_first_scale_s0']*core.inverse_yj(c['location']) for c in law['components']]
    left=sorted(set([x for x in points if x<y]+[y]));right=sorted(set([y]+[x for x in points if x>y]))
    left=[-math.inf]+left;right=right+[math.inf];values=[];errors=[]
    for cuts,survival in ((left,False),(right,True)):
        for a,b in zip(cuts[:-1],cuts[1:]):
            value,error=quad(lambda x:cdf(law,x,survival)**2,a,b,epsabs=2e-9,epsrel=2e-10,limit=250)
            values.append(value);errors.append(error)
    return math.fsum(values),math.fsum(errors)


def quantile(law,probability):
    lower,upper=-1.,1.
    for unused in range(80):
        if cdf(law,lower)<=probability<=cdf(law,upper):break
        lower*=2;upper*=2
    else:raise ArithmeticError('independent native mixture quantile unbracketed')
    return float(brentq(lambda y:cdf(law,y)-probability,lower,upper,xtol=1e-11,rtol=8*np.finfo(float).eps))


def log_density(law,y):
    s0=law['native_first_scale_s0'];v=y/s0;u=core.forward_yj(v)
    logjac=-math.log(s0)+(-.5*math.log1p(v) if v>=0 else .5*math.log1p(-v))
    terms=[float(w)+float(stats.t.logpdf(u,df=c['df'],loc=c['location'],scale=c['scale']))
           for w,c in zip(law['log_weights'],law['components'])]
    return float(special.logsumexp(terms))+logjac
