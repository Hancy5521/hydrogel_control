"""Sample the saved C-stage law; never train or read measured targets."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import numpy as np

PURE_SHA='3ab397115f88981ca5932829a39e577faf60b8bfdccd61ff4a6ca011ca75632f'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--context',required=True)
    parser.add_argument('--method',default='whole_curve_student4_density_mixture')
    parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=args.predictions.read_bytes()
    if hashlib.sha256(data).hexdigest()!=args.sha256:
        raise ValueError('Saved prediction SHA differs')
    path=Path(__file__).resolve().parents[1]/'functional_mixture/pure.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PURE_SHA:
        raise ValueError('Saved C distribution implementation differs')
    spec=importlib.util.spec_from_file_location('_checkpoint_C_example',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[r for r in json.loads(data)['forecasts'] if r['context']==args.context]
    if len(rows)!=1 or args.method not in rows[0]['laws']:
        raise ValueError('One exact available context/method required')
    samples=module.sample(rows[0]['laws'][args.method],args.count,
                          np.random.default_rng(args.seed),return_latents=True)
    args.output.mkdir(parents=True,exist_ok=False)
    arrays={'generated_log_G_over_Pa':np.asarray(samples['draws'])}
    if samples['whole_curve_branch'] is not None:
        arrays['one_branch_for_whole_curve']=np.asarray(samples['whole_curve_branch'])
    if samples['chi_square4_per_curve'] is not None:
        arrays['one_chisquare4_for_whole_curve']=np.asarray(samples['chi_square4_per_curve'])
    np.savez_compressed(args.output/'GENERATED_NOT_MEASURED.npz',**arrays)
    meta={'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'kind':'GENERATED_MODEL_DRAWS_NOT_EXPERIMENTAL_MEASUREMENTS',
        'source_prediction_sha256':args.sha256,'source_code_sha256':PURE_SHA,
        'context':args.context,'method':args.method,'count':args.count,'seed':args.seed,
        'units':'ln(G/(1Pa))','order':'seven Gprime targets, then seven Gdoubleprime targets',
        'new_fits':0,'new_target_reads':0,'new_material_measurements':0,
        'all_fourteen_coordinates_share_branch_and_Student_radial':True,
        'generated_residual_draws_are_not_guaranteed_Maxwell_passive':True,
        'Student_Pa_arithmetic_mean_does_not_exist':args.method.endswith('student4_density_mixture')}
    with (args.output/'GENERATION_METADATA.json').open('x',encoding='utf-8') as file:
        json.dump(meta,file,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
