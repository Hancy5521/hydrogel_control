"""Registered synthetic-only Maxwell/mean-transport numerical gates.

No DaRUS file reader and no measured material response. Fits here are explicitly
synthetic controls. All actual material actions require experiment registration.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from pure_laws import weighted_maxwell,assemble_laws,pinned,METHODS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_forward(f,coeff,tau):
    out=np.zeros((2,len(f)))
    for j,frequency in enumerate(f):
        omega=2*math.pi*frequency
        out[0,j]=coeff[0];out[1,j]=coeff[-1]*omega
        for k,relaxation in enumerate(tau):
            u=omega*relaxation
            out[0,j]+=coeff[k+1]*u*u/(1+u*u)
            out[1,j]+=coeff[k+1]*u/(1+u*u)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    auth=json.loads(Path(args.authorization).read_text(encoding='utf-8'))
    assert auth['scope']=='functional_repair_synthetic_toy_gate' and auth['source_sha256']==sha(__file__)
    stamp=dt.datetime.fromisoformat(auth['actual_utc'])
    assert stamp.tzinfo and stamp<=dt.datetime.now(dt.timezone.utc)
    for path,digest in auth['source_pins'].items():
        # Toy authorization pins code only. Original DaRUS CSVs are not part
        # of this synthetic gate, even for hashing.
        assert Path(path).suffix=='.py'
        assert sha(path)==digest
    frequency=np.array([.01,.0178,.0316,.0562,.1,.178,.316,.562,1.,1.78,3.16,5.62,10.])
    observed=np.array([0,2,4,8,10,12]);target=np.array([1,3,5,6,7,9,11])
    tau=np.geomspace(1/(2*math.pi*10),1/(2*math.pi*.01),6)
    fixtures=[]
    for coeff in (np.array([1500.,80.,220.,170.,260.,140.,330.,.4]),
                  np.array([700.,120.,0.,220.,100.,0.,240.,0.])):
        truth=direct_forward(frequency,coeff,tau)
        result=weighted_maxwell(frequency[observed],truth[0,observed],truth[1,observed],frequency[target],frequency)
        prediction=np.array([result['predicted_Gprime_Pa'],result['predicted_Gdoubleprime_Pa']])
        error=float(np.max(np.abs(prediction-truth[:,target])))
        assert error<=2e-7*max(1.,float(truth.max()))
        physics=pinned('generalization/generalized_maxwell_nnls.py')
        a=physics.design(frequency[observed],tau)
        sigma=np.sqrt(np.mean(truth[:,observed]**2,axis=1))
        row_scale=np.repeat(sigma,6)
        aw=a/row_scale[:,None];yw=truth[:,observed].reshape(12)/row_scale
        fitted_coeff=np.array(result['coefficients_Pa_and_Pa_s'])
        gradient=aw.T@(aw@fitted_coeff-yw)
        tolerance=1e-8*max(1.,float(np.linalg.norm(aw)*np.linalg.norm(aw@fitted_coeff-yw)))
        assert np.min(fitted_coeff)>=0 and np.min(gradient)>=-tolerance
        assert np.max(np.abs(fitted_coeff*gradient))<=1e-7
        assert np.allclose(gradient,result['KKT_gradient_weighted_objective'],rtol=1e-8,atol=1e-12)
        # The same channel expressed in another numerical unit rescales both
        # its design and observations and own RMS; normalized LS is identical.
        unit_rows=np.repeat(np.array([1e-6,1e-3]),6)
        aw_units=(a*unit_rows[:,None])/(row_scale*unit_rows)[:,None]
        yw_units=(truth[:,observed].reshape(12)*unit_rows)/(row_scale*unit_rows)
        assert np.allclose(aw_units,aw,rtol=1e-13,atol=1e-14)
        assert np.allclose(yw_units,yw,rtol=1e-13,atol=1e-14)
        scaled=weighted_maxwell(frequency[observed],truth[0,observed]*1e-6,
            truth[1,observed]*1e-6,frequency[target],frequency)
        scaled_prediction=1e6*np.array([scaled['predicted_Gprime_Pa'],scaled['predicted_Gdoubleprime_Pa']])
        assert np.allclose(scaled_prediction,prediction,rtol=2e-7,atol=2e-7)
        raw=physics.fit_predict(frequency[observed],truth[0,observed],truth[1,observed],frequency[target],frequency)
        raw_prediction=np.array([raw['predicted_Gprime_Pa'],raw['predicted_Gdoubleprime_Pa']])
        assert np.allclose(raw_prediction,truth[:,target],rtol=2e-7,atol=2e-7)
        mean=np.linspace(4.,8.,14)
        k=.04*np.exp(-.4*np.abs(np.arange(7)[:,None]-np.arange(7)[None,:]))+.01*np.eye(7)
        covariance=block_diag(k,1.3*k)
        laws=assemble_laws(mean,covariance,prediction.reshape(14))
        expected=.5*(mean+np.log(prediction.reshape(14)))
        assert np.array_equal(laws[METHODS[0]]['covariance'],covariance)
        assert np.array_equal(laws[METHODS[1]]['covariance'],covariance)
        assert np.array_equal(laws[METHODS[2]]['covariance'],covariance)
        assert np.allclose(laws[METHODS[1]]['mean'],expected,rtol=1e-14,atol=1e-14)
        assert np.allclose(2*np.array(laws[METHODS[3]]['scale_matrix']),covariance,rtol=1e-14,atol=1e-14)
        assert np.linalg.matrix_rank(covariance)==14
        fixtures.append({'true_coefficients':coeff.tolist(),'relaxation_times_s':tau.tolist(),
            'frequency_Hz':frequency.tolist(),'O':observed.tolist(),'T':target.tolist(),
            'synthetic_truth_Pa':truth.tolist(),'weighted_result':result,'raw_result':raw,
            'heldout_forward_max_abs_error_Pa':error,'weighted_KKT_gradient':gradient.tolist(),
            'KKT_fixed_tolerance':tolerance,'unit_rescaling_prediction_Pa':scaled_prediction.tolist(),
            'gaussian_fixture_mean':mean.tolist(),'gaussian_fixture_covariance':covariance.tolist(),
            'assembled_laws':laws})
    stamp=dt.datetime.now(dt.timezone.utc)
    output={'schema':'functional-repair-synthetic-gate-v1','status':'PASS','actual_utc':stamp.isoformat(),
        'source_sha256':sha(__file__),'authorization_sha256':sha(args.authorization),
        'synthetic_fixtures':fixtures,'synthetic_NNLS_fit_calls':6,'material_NNLS_or_GP_fits':0,
        'numeric_DaRUS_G_cells_read':0,'DaRUS_files_opened':0,
        'scope':'forward/KKT/unit-reparameterization/full-covariance gate, not material performance'}
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({'status':'PASS','output':args.output,'sha256':sha(args.output)}))


if __name__=='__main__':main()
