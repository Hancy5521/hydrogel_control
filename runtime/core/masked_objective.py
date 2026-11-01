"""One joint energy, conditional normalizer ratio, true-observation cases only."""
from dataclasses import dataclass
import numpy as np
from latent_reference import indices,real,nonnegative
from energy_correction import GaussianExpectation


@dataclass(frozen=True)
class MaskedCase:
    context:object
    context_values:object
    target:object
    target_values:object
    weight:float=1.
    group_id:str='generated_or_registered_group'


class MaskedObjective:
    def __init__(self,projection,potential,cases,*,order=32,ridge=0.,max_nodes=65536):
        ridge=nonnegative(ridge,'ridge')
        if projection.rank!=potential.rank:raise ValueError('Compatible projection/potential required')
        self.projection=projection;self.potential=potential;self.ridge=ridge;self.records=[]
        for case in cases:
            c=indices(case.context,projection.reference.dimension);t=indices(case.target,projection.reference.dimension)
            yc=real(case.context_values,'context values',1);yt=real(case.target_values,'true target values',1)
            if len(t)==0 or yc.shape!=c.shape or yt.shape!=t.shape or np.intersect1d(c,t).size:raise ValueError('Finite measured disjoint nonempty target case required')
            weight=nonnegative(case.weight,'group weight')
            if weight<=0:raise ValueError('Positive explicit material/group weight required')
            d=np.r_[c,t];yd=np.r_[yc,yt]
            mc,vc,_=projection.posterior(c,yc);md,vd,_=projection.posterior(d,yd)
            qc=GaussianExpectation(mc,vc,potential,order=order,max_nodes=max_nodes)
            qd=GaussianExpectation(md,vd,potential,order=order,max_nodes=max_nodes)
            baseline=projection.reference.logpdf(d,yd)-projection.reference.logpdf(c,yc)
            self.records.append((float(case.weight),baseline,qc,qd,str(case.group_id)))
        if not self.records:raise ValueError('Nonempty actual-observation case list required')
        self.total_weight=sum(r[0] for r in self.records)

    def __call__(self,theta):
        theta=real(theta,'theta',1)
        if len(theta)!=self.potential.parameter_count:raise ValueError('Parameter dimension mismatch')
        value=0.;gradient=np.zeros_like(theta)
        for weight,base,qc,qd,group in self.records:
            ac,gc=qc.evaluate(theta);ad,gd=qd.evaluate(theta)
            value+=weight*(-base-ad+ac);gradient+=weight*(-gd+gc)
        return value/self.total_weight+self.ridge*(theta@theta),gradient/self.total_weight+2*self.ridge*theta

    def conditional_logpdf(self,theta,index):
        _,base,qc,qd,_=self.records[index];ac,_=qc.evaluate(theta);ad,_=qd.evaluate(theta)
        return base+ad-ac
