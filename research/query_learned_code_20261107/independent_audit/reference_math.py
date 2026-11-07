"""Independent saved-only exact CRPS/risk-code arithmetic.

No system-under-test imports, no model fitting, no generated cases, no raw-table
access. This source is dormant until the root's actual pre-execution protocol.
"""
from fractions import Fraction
import math

F = Fraction
ZERO = F(0)
P = (F(1,20), F(9,20), F(9,20), F(1,20))
EDGES = (ZERO, F(1,20), F(1,2), F(19,20), F(1))


def scalar(x):
    if isinstance(x, bool):
        raise TypeError('boolean is not a scalar')
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError('finite input required')
        return F(*x.as_integer_ratio())
    return F(x)


def vec(x):
    return tuple(map(scalar, x))


def matrix(x):
    return tuple(vec(row) for row in x)


def inner(a,b):
    if len(a) != len(b):
        raise ValueError('dimension mismatch')
    return sum((x*y for x,y in zip(a,b)),ZERO)


def times(a,b):
    return tuple(inner(row,b) for row in a)


def canonical(law):
    if isinstance(law,dict):
        assert law['family']=='weighted_empirical'
        assert law['coordinate']=='native'
        assert len(law['support_native'])==len(law['weight_fractions'])
        data=zip(law['support_native'],law['weight_fractions'])
    else:
        data=law
    out={}
    for x,w in data:
        x,w=scalar(x),scalar(w)
        if w<0:
            raise ValueError('negative probability')
        if w:
            out[x]=out.get(x,ZERO)+w
    if not out or sum(out.values(),ZERO)!=1:
        raise ValueError('exact probability one required')
    return tuple(sorted(out.items()))


def pit_components(law):
    """Split atom probability intervals against the four fixed PIT bins."""
    atoms=canonical(law)
    components=[[ZERO]*len(atoms) for _ in range(4)]
    for j in range(4):
        cumulative=ZERO
        for k,(_,mass) in enumerate(atoms):
            a,b=cumulative,cumulative+mass
            overlap=max(ZERO,min(b,EDGES[j+1])-max(a,EDGES[j]))
            components[j][k]=overlap/P[j]
            cumulative=b
        assert sum(components[j],ZERO)==1
    return atoms,tuple(tuple(row) for row in components)


def law_from_theta(law,theta,g=None):
    atoms,comps=pit_components(law)
    theta=vec(theta)
    blocks=1 if g is None else 2
    if len(theta)!=4*blocks:
        raise ValueError('theta dimension')
    for k in range(blocks):
        b=theta[4*k:4*k+4]
        assert sum(b,ZERO)==1 and all(t>=P[j]/100 for j,t in enumerate(b))
    if g is None:
        effective=theta
    else:
        g=scalar(g)
        assert ZERO<=g<=1
        effective=tuple((1-g)*theta[j]+g*theta[j+4] for j in range(4))
    weights=tuple(sum((effective[j]*comps[j][k] for j in range(4)),ZERO)
                  for k in range(len(atoms)))
    return canonical(zip((x for x,_ in atoms),weights))


def energy_crps(law,y):
    """Sorted weighted-rank energy formula, independent of SUT prefix loss."""
    atoms,y=canonical(law),scalar(y)
    absolute=sum((w*abs(x-y) for x,w in atoms),ZERO)
    before=ZERO
    rank_term=ZERO
    for x,w in atoms:
        rank_term+=w*x*(2*before+w-1)
        before+=w
    return absolute-rank_term


def geometry(law,y=None,g=None):
    """Direct integration at every finite interval, without SUT c prefixes."""
    atoms,comps=pit_components(law)
    n=len(atoms)
    d=3 if g is None else 6
    lift=(F(1),) if g is None else (1-scalar(g),scalar(g))
    if g is not None and not ZERO<=scalar(g)<=1:
        raise ValueError('g outside [0,1]')
    Q=[[ZERO]*d for _ in range(d)]
    c=[ZERO]*d if y is not None else None
    CDF=[ZERO]*4
    base=ZERO
    cmin=[ZERO]*d
    integrals=[ZERO]*d
    integral_rows=[tuple(integrals)]
    hrows=[]
    y=scalar(y) if y is not None else None
    for k in range(n-1):
        base+=atoms[k][1]
        for j in range(4):
            CDF[j]+=comps[j][k]
        hbase=tuple(CDF[j]-CDF[3] for j in range(3))
        h=tuple(t*v for t in lift for v in hbase)
        a,b=atoms[k][0],atoms[k+1][0]
        length=b-a
        hrows.append(h)
        for j in range(d):
            cmin[j]+=length*h[j]*(base-1)
            integrals[j]+=length*h[j]
            if c is not None:
                # Length of [a,b] at/above y; no artificial target clipping.
                above=max(ZERO,b-max(a,y)) if y<b else ZERO
                c[j]+=h[j]*(base*length-above)
            for ell in range(d):
                Q[j][ell]+=length*h[j]*h[ell]
        integral_rows.append(tuple(integrals))
    knot_c=tuple(tuple(cmin[j]+row[j] for j in range(d)) for row in integral_rows)
    bounds=tuple((min(row[j] for row in knot_c),max(row[j] for row in knot_c))
                 for j in range(d))
    return {'Q':tuple(map(tuple,Q)),'c':None if c is None else tuple(c),
            'bounds':bounds,'knot_c':knot_c,'h':tuple(hrows),'atoms':atoms}


def theta_beta(theta):
    theta=vec(theta)
    if len(theta) not in (4,8):
        raise ValueError('theta length')
    for k in range(0,len(theta),4):
        assert sum(theta[k:k+4],ZERO)==1
        assert all(theta[k+j]>=P[j]/100 for j in range(4))
    return tuple(theta[k+j]-P[j] for k in range(0,len(theta),4) for j in range(3))


def exact_range(Q):
    """Rational Gram-Schmidt of columns, different from SUT RREF/inverse."""
    Q=matrix(Q)
    n=len(Q)
    assert n and all(len(row)==n for row in Q)
    assert all(Q[i][j]==Q[j][i] for i in range(n) for j in range(n))
    basis=[]
    norms=[]
    for j in range(n):
        v=tuple(Q[i][j] for i in range(n))
        for u,norm in zip(basis,norms):
            coefficient=inner(v,u)/norm
            v=tuple(x-coefficient*y for x,y in zip(v,u))
        norm=inner(v,v)
        if norm:
            basis.append(v)
            norms.append(norm)
    projector=tuple(tuple(sum((v[i]*v[j]/norm for v,norm in zip(basis,norms)),ZERO)
                          for j in range(n)) for i in range(n))
    assert times(projector,tuple(ZERO for _ in range(n)))==tuple(ZERO for _ in range(n))
    return len(basis),projector


def objective(theta,Q,c,prior,tau=F(1,4),count=1,penalty_Q=None):
    beta,prior_beta=theta_beta(theta),theta_beta(prior)
    Q,c=matrix(Q),vec(c)
    penalty=Q if penalty_Q is None else matrix(penalty_Q)
    delta=tuple(x-y for x,y in zip(beta,prior_beta))
    value=(inner(beta,times(Q,beta))+2*inner(beta,c)
           +scalar(tau)*inner(delta,times(penalty,delta)))/count
    direction=tuple(2*(a+b+scalar(tau)*r)/count
                    for a,b,r in zip(times(Q,beta),c,times(penalty,delta)))
    gradient=[]
    for start in range(0,len(beta),3):
        gradient.extend(F(99,100)*direction[j] for j in range(start,start+3))
        gradient.append(ZERO)
    return value,tuple(gradient)


def policy(widths,gaps):
    widths,gaps=vec(widths),vec(gaps)
    n=len(widths)
    assert n and len(gaps)==n and min(widths+gaps)>=0
    u=F(1,n)
    a,b=sum(widths,ZERO),sum(gaps,ZERO)
    return tuple(u/4+F(3,8)*(w/a if a else u)+F(3,8)*(g/b if b else u)
                 for w,g in zip(widths,gaps))


def ticket_position(q,ticket,total):
    q=vec(q)
    ticket,total=int(ticket),int(total)
    assert sum(q,ZERO)==1 and min(q)>0 and ZERO<=ticket<total
    # A valid exact-ticket receipt must allocate an integer count to every unit.
    counts=tuple(v*total for v in q)
    assert all(c.denominator==1 for c in counts)
    cumulative=0
    for j,c in enumerate(counts):
        if cumulative<=ticket<cumulative+int(c):
            return j
        cumulative+=int(c)
    raise AssertionError('unallocated integer ticket')
