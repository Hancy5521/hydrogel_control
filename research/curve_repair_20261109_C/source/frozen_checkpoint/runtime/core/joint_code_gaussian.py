"""Five-parameter Gaussian replacement of a fixed standardized two-dimensional code.

No file I/O, material identity, masking policy or experiment loop is implemented
here. The full reference residual is retained. This is a classical Gaussian
identity, an attribution control rather than a claim of new probability theory.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

PARAMETERS=('a0','a1','log_L00','L10','log_L11')
BOUNDS=((-1.,1.),(-1.,1.),(-.75,.75),(-1.,1.),(-.75,.75))
OPTIONS={'maxiter':400,'maxfun':800,'ftol':1e-14,'gtol':1e-7,'maxls':40}
RIDGE=.02
PG_GATE=1e-5


def array(value,name,ndim):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'iuf' or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):
        raise ValueError(name+' must be real numeric, not boolean')
    a=np.array(raw,dtype=float,copy=True)
    if a.ndim!=ndim or not np.isfinite(a).all():raise ValueError(name+' has invalid shape or values')
    a.setflags(write=False);return a


def unpack(theta):
    theta=array(theta,'theta',1)
    if theta.shape!=(5,):raise ValueError('Exactly5 parameters required')
    with np.errstate(over='raise',under='raise',invalid='raise'):
        l=np.array([[np.exp(theta[2]),0.],[theta[3],np.exp(theta[4])]])
        s=l@l.T
    if not np.isfinite(s).all():raise ValueError('Nonfinite code covariance')
    # No clipping, jitter, density-ratio constraint, or learned code rotation.
    cho_factor(s,lower=True)
    ds=np.array([[[2*l[0,0]**2,l[0,0]*l[1,0]],[l[0,0]*l[1,0],0.]],
                 [[0.,l[0,0]],[l[0,0],2*l[1,0]]],
                 [[0.,0.],[0.,2*l[1,1]**2]]])
    return theta[:2],s,ds


def indices(value,n):
    v=np.asarray(value)
    if v.ndim!=1 or v.dtype.kind not in 'iu' or v.dtype.kind=='b' or len(set(v.tolist()))!=len(v) or (v<0).any() or (v>=n).any():raise ValueError('Unique valid integer indices required')
    return np.array(v,dtype=int)


@dataclass(frozen=True)
class CodeReference:
    mean:object
    covariance:object
    cross:object

    def __post_init__(self):
        m=array(self.mean,'mean',1);k=array(self.covariance,'covariance',2);b=array(self.cross,'cross',2)
        n=len(m)
        if n==0 or k.shape!=(n,n) or b.shape!=(n,2):raise ValueError('Reference andcross shapes differ')
        if not np.allclose(k,k.T,atol=1e-12,rtol=1e-12):raise ValueError('Symmetric covariance required')
        cho_factor(k,lower=True)
        residual=(k-b@b.T);residual=(residual+residual.T)/2
        tol=1e-10*max(1.,np.linalg.norm(k,2))
        if np.linalg.eigvalsh(residual)[0] < -tol:raise ValueError('Invalid Cov(Y,z) for a unit-Gaussian code')
        object.__setattr__(self,'mean',m);object.__setattr__(self,'covariance',k);object.__setattr__(self,'cross',b)

    def joint(self,theta):
        a,s,_=unpack(theta)
        m=self.mean+self.cross@a;k=self.covariance+self.cross@(s-np.eye(2))@self.cross.T
        return m,(k+k.T)/2

    def condition(self,theta,observed,values,target):
        o=indices(observed,len(self.mean));h=indices(target,len(self.mean));v=array(values,'observations',1)
        if len(v)!=len(o) or np.intersect1d(o,h).size:raise ValueError('Disjoint context/target and matching observations required')
        m,k=self.joint(theta)
        if not len(o):return m[h],k[np.ix_(h,h)]
        f=cho_factor(k[np.ix_(o,o)],lower=True);b=k[np.ix_(h,o)]
        mm=m[h]+b@cho_solve(f,v-m[o]);vv=k[np.ix_(h,h)]-b@cho_solve(f,b.T)
        return mm,(vv+vv.T)/2

    def nll_gradient(self,theta,values,subset=None):
        y=array(values,'values',1);h=np.arange(len(self.mean)) if subset is None else indices(subset,len(self.mean))
        if y.shape!=(len(h),) or not len(h):raise ValueError('Nonempty target-aligned values required')
        a,s,ds=unpack(theta);b=self.cross[h];m=self.mean[h]+b@a
        k=self.covariance[np.ix_(h,h)]+b@(s-np.eye(2))@b.T;k=(k+k.T)/2
        f=cho_factor(k,lower=True);e=y-m;w=cho_solve(f,e);wb=b.T@w
        value=.5*(len(h)*np.log(2*np.pi)+2*np.log(np.diag(f[0])).sum()+e@w)
        gs=.5*(b.T@cho_solve(f,b)-np.outer(wb,wb))
        gradient=np.r_[-wb,np.einsum('ij,kij->k',gs,ds)]
        return float(value),gradient


@dataclass(frozen=True)
class CurveTask:
    reference:CodeReference
    values:object
    masks:object

    def __post_init__(self):
        y=array(self.values,'measured curve',1)
        if y.shape!=self.reference.mean.shape:raise ValueError('One actual complete training curve per task')
        masks=tuple(indices(v,len(y)) for v in self.masks)
        if not masks or any(len(v)==0 or len(v)==len(y) for v in masks):raise ValueError('Nonempty proper observed masks required')
        object.__setattr__(self,'values',y);object.__setattr__(self,'masks',masks)


class ConditionalObjective:
    """Exact low-dimensional Gaussian likelihood-ratio expression.

    For A=C.T K^-1 C, h=C.T K^-1(y-m), T=S-I, and
    G=(I+T A)^-1 T, the Gaussian determinant/Woodbury identities give
    deltaNLL=.5*(logdet(I+T A)-2a.T h+a.T A a-u.T G u), u=h-Aa.
    Avoids subtracting large39-dimensional log densities at every evaluation.
    The parameter gradient, optimum, bounds and regularizer are unchanged.
    """
    def __init__(self,tasks,ridge=RIDGE):
        self.tasks=tuple(tasks)
        if not self.tasks:raise ValueError('At leastone independently identified curve required')
        if not np.isfinite(ridge) or ridge<0:raise ValueError('Nonnegative ridge required')
        self.ridge=float(ridge);self.constants=[];self.records=[];zero=np.zeros(5)
        for task in self.tasks:
            full=task.reference.nll_gradient(zero,task.values)[0]
            context=np.mean([task.reference.nll_gradient(zero,task.values[o],o)[0] for o in task.masks])
            self.constants.append(full-context)
            self.records.append([self.summary(task,np.arange(len(task.values)))]+[self.summary(task,o) for o in task.masks])
        self.constant=float(np.mean(self.constants))

    @staticmethod
    def summary(task,index):
        prior=task.reference;b=prior.cross[index];k=prior.covariance[np.ix_(index,index)];f=cho_factor(k,lower=True)
        a=b.T@cho_solve(f,b);h=b.T@cho_solve(f,task.values[index]-prior.mean[index])
        return (a+a.T)/2,h

    @staticmethod
    def delta(theta,a,h):
        mu,s,ds=unpack(theta);t=s-np.eye(2);system=np.eye(2)+t@a
        sign,ld=np.linalg.slogdet(system)
        if sign<=0:raise ValueError('Nonpositive Gaussian determinant ratio')
        g=np.linalg.solve(system,t);g=(g+g.T)/2;u=h-a@mu;w=u-a@g@u
        value=.5*(ld-2*mu@h+mu@a@mu-u@g@u)
        gs=.5*(a-a@g@a-np.outer(w,w))
        return value,np.r_[-w,np.einsum('ij,kij->k',gs,ds)]

    def __call__(self,theta):
        theta=array(theta,'theta',1);values=[];gradients=[]
        for records in self.records:
            out=[self.delta(theta,*record) for record in records]
            values.append(out[0][0]-np.mean([v[0] for v in out[1:]]))
            gradients.append(out[0][1]-np.mean([v[1] for v in out[1:]],axis=0))
        return (float(np.mean(values)+self.ridge*(theta@theta)),np.mean(gradients,axis=0)+2*self.ridge*theta)


def projected_gradient(theta,g):
    b=np.array(BOUNDS);g=np.array(g,copy=True);t=np.asarray(theta)
    g[((t<=b[:,0]+1e-9)&(g>0))|((t>=b[:,1]-1e-9)&(g<0))]=0.
    return g


class FitFailure(RuntimeError):
    def __init__(self,receipt):super().__init__('Joint-code Gaussian fit not admitted');self.receipt=receipt


def fit(tasks):
    objective=ConditionalObjective(tasks);initial=np.zeros(5)
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=BOUNDS,options=OPTIONS)
    value,g=objective(result.x);pg=float(np.max(abs(projected_gradient(result.x,g))))
    admitted=bool(result.success and np.isfinite(value) and np.isfinite(g).all() and pg<=PG_GATE and value<=1e-10)
    a,s,_=unpack(result.x)
    receipt={'theta':result.x.tolist(),'parameter_names':list(PARAMETERS),'a':a.tolist(),'S':s.tolist(),
             'centered_objective':value,'raw_objective':value+objective.constant,'removed_constant':objective.constant,
             'gradient':g.tolist(),'projected_gradient_inf':pg,'solver_success':bool(result.success),'message':str(result.message),
             'iterations':int(result.nit),'evaluations':int(result.nfev),'admitted':admitted,
             'initial_theta':initial.tolist(),'optimizer_runs':1,'bounds':[list(b) for b in BOUNDS],
             'ridge':RIDGE,'optimizer':OPTIONS,'curve_tasks':len(tasks),'masks_per_curve':[len(t.masks) for t in tasks],
             'no_jitter':True,'no_fallback_or_retry':True,'parameter_uncertainty_integrated':False}
    if not admitted:raise FitFailure(receipt)
    return receipt
