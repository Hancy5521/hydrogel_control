"""Independent fixed Gaussian-product hypotheses and native-coordinate CDF integration."""
import math
from scipy import special
from scipy.integrate import quad

TARGET_CUT=12.
QUERY_CUT=14.
EPSABS=1e-10
EPSREL=2e-9


def pdf(z,mu,sd):
    return math.exp(-.5*((z-mu)/sd)**2)/(math.sqrt(2*math.pi)*sd)


def posterior(weights,z,means,sd):
    terms=[w*pdf(z,m,s) for w,m,s in zip(weights,means,sd)]
    total=math.fsum(terms)
    if not total>0:raise ArithmeticError('positive fixed query mixture density required')
    return [x/total for x in terms],total


def fixtures():
    prior=[1/3]*3
    negative,_=posterior(prior,-1.,[-1.,0.,1.],[.75]*3)
    positive,_=posterior(prior,1.,[-1.,0.,1.],[.75]*3)
    result=[]
    for stem,weights in [('prior',prior),('after_negative',negative),('after_positive',positive)]:
        for name,means in [('qL',[-2.,2.,2.]),('qR',[-2.,-2.,2.])]:
            result.append({'id':stem+'_'+name,'weights':list(weights),'query_means':means,
                           'query_SD':[1.]*3,'target_means':[-1.,0.,1.],'target_SD':[1.]*3})
    result.append({'id':'null_query_after_negative','weights':negative,'query_means':[0.]*3,
                   'query_SD':[1.]*3,'target_means':[-1.,0.,1.],'target_SD':[1.]*3})
    result.append({'id':'null_target_after_negative','weights':negative,'query_means':[-2.,2.,2.],
                   'query_SD':[1.]*3,'target_means':[0.]*3,'target_SD':[1.]*3})
    return result


def integrate(function,lower,upper,points):
    splits=sorted(set([lower,upper]+[x for x in points if lower<x<upper]))
    values=[];errors=[]
    for a,b in zip(splits[:-1],splits[1:]):
        value,error=quad(function,a,b,epsabs=EPSABS/len(splits),epsrel=EPSREL,limit=160)
        values.append(value);errors.append(error)
    return math.fsum(values),math.fsum(errors)


def native_risk_gain(case):
    """Positive squared native-target CDF differences; no pair-distance formula."""
    weights=case['weights'];max_inner_error=0.;calls=0
    def inner(query):
        nonlocal max_inner_error,calls
        updated,density=posterior(weights,query,case['query_means'],case['query_SD'])
        delta=[a-b for a,b in zip(updated,weights)]
        def square(target):
            difference=math.fsum(a*float(special.ndtr((target-m)/s))
                                 for a,m,s in zip(delta,case['target_means'],case['target_SD']))
            return difference*difference
        value,error=integrate(square,-TARGET_CUT,TARGET_CUT,[0.]+case['target_means'])
        max_inner_error=max(max_inner_error,error);calls+=1
        return density*value
    value,error=integrate(inner,-QUERY_CUT,QUERY_CUT,[0.]+case['query_means'])
    sf11=float(special.ndtr(-11.));sf12=float(special.ndtr(-12.))
    target_tail=2*(pdf(11.,0.,1.)-11*sf11)
    query_tail=48*sf12
    return {'value':value,'outer_quad_error_estimate':error,'max_inner_quad_error_estimate':max_inner_error,
            'summed_error_estimate_diagnostic':error+max_inner_error,
            'inner_quad_calls':calls,'ideal_arithmetic_target_tail':target_tail,
            'ideal_arithmetic_query_tail':query_tail,'query_density_renormalized':False,
            'route':'direct native-query z mixture density times native-target y CDF squared; no D/C producer import',
            'error_estimates_not_rigorous_floating_point_certificate':True}


def covariance(case):
    """Independent centered-weight covariance integral, with untruncated tail account."""
    weights=case['weights'];matrix=[[0.]*3 for _ in range(3)];maximum_error=0.
    for i in range(3):
        for j in range(i,3):
            def integrand(z):
                updated,density=posterior(weights,z,case['query_means'],case['query_SD'])
                return density*(updated[i]-weights[i])*(updated[j]-weights[j])
            value,error=integrate(integrand,-QUERY_CUT,QUERY_CUT,[0.]+case['query_means'])
            matrix[i][j]=matrix[j][i]=value;maximum_error=max(maximum_error,error)
    return matrix,maximum_error


def distances(case):
    """Absolute normal difference formula only as an additional saved-D sanity check."""
    result=[]
    for i in range(3):
        row=[]
        for j in range(3):
            d=abs(case['target_means'][i]-case['target_means'][j])
            s=math.hypot(case['target_SD'][i],case['target_SD'][j])
            row.append(s*math.sqrt(2/math.pi)*math.exp(-.5*(d/s)**2)+d*math.erf(d/(s*math.sqrt(2))))
        result.append(row)
    return result
