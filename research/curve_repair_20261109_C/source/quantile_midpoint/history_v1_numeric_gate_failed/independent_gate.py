"""Registered fixed-fixture gate with independent t4 algebra and normal-domain CRPS.

Production is imported only AFTER root pins and this auditor's actual-UTC
registration pass. Expected quantiles, Jacobians and scores never call SUT.
No material predictions, target artifacts, fitting or tau search are read/run.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import multivariate_normal,t

SUT_SHA="79a3d5e5b96c710780870a2875d7cd3daac9f01ed235936d71bf69ac295cac12"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write_new(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

def t4_algebraic(z):
    """Invert t4 CDF by the cubic u^3-3u+2-4p=0, stable near u=1.

    delta=1-u=1-cos(d)+sqrt(3)sin(d), d=2asin(sqrt(p))/3.
    Unlike production this never calls scipy's Student inverse CDF.
    """
    a=np.asarray(z,float)
    p=ndtr(-np.abs(a))
    angle=(2/3)*np.arcsin(np.sqrt(p))
    delta=2*np.sin(angle/2)**2+math.sqrt(3)*np.sin(angle)
    value=2*(1-delta)/np.sqrt(delta*(2-delta))
    return np.where(a==0,0.,np.sign(a)*value)

def mp_h_and_jacobian(z,mg,sg,ms,ss):
    with mp.workdps(160):
        z,mg,sg,ms,ss=[mp.mpf(float(v)) for v in [z,mg,sg,ms,ss]]
        p=mp.erfc(abs(z)/mp.sqrt(2))/2
        angle=mp.mpf(2)/3*mp.asin(mp.sqrt(p))
        delta=2*mp.sin(angle/2)**2+mp.sqrt(3)*mp.sin(angle)
        if z==0:a=mp.mpf(0)
        else:a=mp.sign(z)*2*(1-delta)/mp.sqrt(delta*(2-delta))
        phi=mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
        pdf=mp.mpf(3)/8*(1+a*a/4)**(-mp.mpf(5)/2)
        value=(mg+sg*z+ms+ss*a)/2
        derivative=(sg+ss*phi/pdf)/2
        return float(value),float(derivative),float(mp.log(derivative))

def independent_h(z,mg,sg,ms,ss):
    return .5*(mg+sg*np.asarray(z)+ms+ss*t4_algebraic(z))

def independent_inverse(y,mg,sg,ms,ss):
    return brentq(lambda z:float(independent_h(z,mg,sg,ms,ss))-y,-16.,16.,xtol=4e-14,rtol=1e-14)

def normal_domain_crps(y,mg,sg,ms,ss,student_half_energy,limit=12):
    crossing=independent_inverse(y,mg,sg,ms,ss)
    points=[crossing] if -limit<crossing<limit else None
    absolute,error=quad(lambda z:abs(float(independent_h(z,mg,sg,ms,ss))-y)
        *math.exp(-z*z/2)/math.sqrt(2*math.pi),-limit,limit,
        points=points,epsabs=2e-11,epsrel=2e-12,limit=220)
    # Gini half-energy is linear along a comonotonic quantile interpolation.
    half_energy=.5*(sg/math.sqrt(math.pi)+ss*student_half_energy)
    return absolute-half_energy,error

class Checks:
    def __init__(self):self.rows=[]
    def close(self,name,a,b,atol=2e-8,rtol=2e-8):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(name+":shape/nonfinite")
        err=float(np.max(np.abs(a-b),initial=0))
        self.rows.append({"check":name,"elements":int(a.size),"max_absolute_error":err,"atol":atol,"rtol":rtol})
        if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError(name+":error="+str(err))

def audit(sut,checks):
    mg=np.array([.1,-.4,.6]);sg=np.array([.2,.4,.9])
    ms=np.array([.2,-.1,.3]);ss=np.array([.3,.7,1.1])
    r=np.array([[1,.4,-.2],[.4,1,.3],[-.2,.3,1.]])
    law=sut.QuantileMidpoint(mg,sg,ms,ss,r)
    probes=np.array([[-9.,0.,9.],[9.,-9.,0.],[0.,0.,0.],[-2.,1.,.5],[.1,-.2,3.]])
    actual_y=law.from_latent(probes)
    expected_y=np.zeros_like(actual_y);expected_logjac=np.zeros_like(actual_y)
    for i in range(len(probes)):
        for j in range(3):
            expected_y[i,j],_,expected_logjac[i,j]=mp_h_and_jacobian(probes[i,j],mg[j],sg[j],ms[j],ss[j])
    checks.close("mp160_quantile_map",actual_y,expected_y,atol=3e-8,rtol=3e-9)
    checks.close("inverse_tail_roundtrip",law.to_latent(actual_y),probes,atol=3e-11,rtol=3e-11)
    checks.close("mp160_log_jacobian",law.log_derivative(probes),expected_logjac,atol=3e-9,rtol=3e-9)
    step=2e-5
    derivative=(law.from_latent(probes+step)-law.from_latent(probes-step))/(2*step)
    checks.close("finite_difference_positive_jacobian",derivative,np.exp(expected_logjac),atol=3e-8,rtol=3e-7)
    if np.any(derivative<=0):raise AssertionError("Transport not increasing")
    expected_logpdf=multivariate_normal.logpdf(probes,mean=np.zeros(3),cov=r)-np.sum(expected_logjac,axis=1)
    checks.close("non_diagonal_joint_density",law.logpdf(actual_y),expected_logpdf,atol=3e-8,rtol=3e-8)
    two=sut.QuantileMidpoint(mg[:2],sg[:2],ms[:2],ss[:2],np.array([[1,.6],[.6,1.]]))
    z2=np.array([[-2.,.5],[0.,0.],[2.,3.],[-9.,9.]])
    y2=two.from_latent(z2);ind_z=np.zeros_like(z2);ind_jac=np.zeros_like(z2)
    for i in range(len(z2)):
        for j in range(2):
            ind_z[i,j]=independent_inverse(y2[i,j],mg[j],sg[j],ms[j],ss[j])
            _,_,ind_jac[i,j]=mp_h_and_jacobian(ind_z[i,j],mg[j],sg[j],ms[j],ss[j])
    expected2=multivariate_normal.logpdf(ind_z,mean=np.zeros(2),cov=[[1,.6],[.6,1.]])-ind_jac.sum(axis=1)
    checks.close("independent_2D_inverse_joint_density",two.logpdf(y2),expected2,atol=3e-8,rtol=3e-8)
    scalar=sut.QuantileMidpoint(mg[:1],sg[:1],ms[:1],ss[:1],np.eye(1))
    mass,mass_error=quad(lambda y:math.exp(scalar.logpdf(np.array([y]))),-np.inf,np.inf,
        epsabs=3e-9,epsrel=3e-9,limit=220)
    checks.close("one_dimensional_density_normalization",mass,1.,atol=2e-7,rtol=0)
    student_energy,energy_error=quad(lambda a:2*t.cdf(a,4)*t.sf(a,4),0,np.inf,
                                    epsabs=2e-12,epsrel=2e-12,limit=180)
    score_trace=[]
    for i,y in enumerate(actual_y):
        score192=law.marginal_crps(y,order=192)
        for j in range(3):
            expected,error=normal_domain_crps(y[j],mg[j],sg[j],ms[j],ss[j],student_energy,12)
            wider,wider_error=normal_domain_crps(y[j],mg[j],sg[j],ms[j],ss[j],student_energy,14)
            checks.close("normal_domain_CRPS_tail_truncation",expected,wider,atol=5e-8,rtol=3e-12)
            checks.close("192_point_quantile_CRPS_vs_independent_normal_domain",score192[j],expected,atol=2e-7,rtol=3e-11)
            score_trace.append({"probe":i,"dimension":j,"value":float(y[j]),"latent":float(probes[i,j]),
                "SUT_CRPS_order192":float(score192[j]),"independent_normal_CRPS":expected,
                "normal_domain_quad_error":error,"limit12_vs14_error":abs(expected-wider),
                "absolute_error":abs(float(score192[j])-expected)})
    return {"fixture":{"gaussian_location":mg.tolist(),"gaussian_sd":sg.tolist(),
            "student_location":ms.tolist(),"student_scale":ss.tolist(),"correlation":r.tolist()},
        "probes":probes.tolist(),"density_normalization":mass,"normalization_quad_error":mass_error,
        "Student4_Gini_half_energy":student_energy,"Student4_Gini_quad_error":energy_error,
        "CRPS_trace":score_trace,"checks":checks.rows,"synthetic_draws":0,
        "new_measurements":0,"new_material_eta":0,"material_predictions_or_targets_opened":False,
        "new_fits":0,"tau_search":False,"expected_calls_SUT":False,
        "interpretation":"software/math gate only; no generalization or material-score gain"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root-registration",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=load(args.root_registration)
    if root["schema"]!="fixed-midpoint-gate-v1" or root["tau"]!=.5 or root["df"]!=4:
        raise PermissionError("Exact root fixed-midpoint gate authorization required")
    stamp=dt.datetime.fromisoformat(root["actual_utc"])
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):raise PermissionError("Actual nonfuture UTC required")
    for name,pin in root["source_pins"].items():
        if sha(name)!=pin:raise PermissionError("Root registered source changed: "+name)
    source=Path(__file__).with_name("quantile_law.py")
    if sha(source)!=SUT_SHA:raise PermissionError("Registered QuantileMidpoint revision required")
    write_new(Path(__file__).with_name("INDEPENDENT_GATE_REGISTERED.json"),{
        "actual_utc":utc(),"auditor_source_sha256":sha(__file__),"root_registration_sha256":sha(args.root_registration),
        "SUT_source_sha256":sha(source),"expected":"t4 cubic quantile,mp160 Jacobian,MVNormal library,normal-domain energy CRPS",
        "tests":"fixed synthetic 1D/2D/3D,latent0/±9,CRPS192 absolute2e-7/relative3e-11,normal mass tolerance2e-7",
        "new_material_eta":0,"new_fits":0,"no_tau_search":True})
    checks=Checks()
    try:
        spec=importlib.util.spec_from_file_location("_root_registered_midpoint_"+SUT_SHA,source)
        sut=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sut;spec.loader.exec_module(sut)
        result=audit(sut,checks)
        report={"status":"PASS","actual_utc":utc(),"auditor_source_sha256":sha(__file__),
                "SUT_source_sha256":sha(source),"root_registration_sha256":sha(args.root_registration),**result}
    except Exception as error:
        report={"status":"FAIL","actual_utc":utc(),"error_type":type(error).__name__,"error":str(error),
            "auditor_source_sha256":sha(__file__),"SUT_source_sha256":sha(source),"checks":checks.rows,
            "new_material_eta":0,"new_fits":0,"tau_search":False}
        write_new(args.output,report);raise
    write_new(args.output,report)
    print("PASS fixed midpoint independent gate;checks="+str(len(checks.rows)))

if __name__=="__main__":main()
