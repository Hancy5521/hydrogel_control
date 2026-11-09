"""Original xanthan flow CSV inputs only: never convert Var2/Var3/Var4.

Var1 is shear rate by original README and Data in Brief ordered schema.
Response strings necessarily enter the CSV text reader but are never cast,
printed, aggregated, exported or used for validity filtering.
"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import io
import json
import math
import zipfile


HERE = Path(__file__).resolve().parent
ZIP_SHA = 'c50696f0776fba1a721e6a2804480b2d2aa900e232f3e93eacbfd35403c904a4'
PAPER_KEYS = {'NaCl_0_XG','NaCl_0.1_XG','NaCl_0.3_XG','NaCl_0.4_XG','NaCl_0.5_XG','NaCl_0.7_XG',
              'XG_NaCl_0.1','XG_NaCl_0.3','XG_NaCl_0.5','XG_NaCl_0.7','XG_NaCl_0.9'}


def main():
    raw = (HERE/'Rheological_data.zip').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ZIP_SHA:
        raise ValueError('Original ZIP changed')
    records=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith('/flow_data.csv'):
                continue
            payload=archive.read(member)
            reader=csv.reader(io.StringIO(payload.decode('utf-8-sig')))
            if next(reader)!=['Var1','Var2','Var3','Var4']:
                raise ValueError('Unanticipated original flow header')
            units=next(reader)
            if units!=['shear rate [1/s]','shear stress [Pa]','viscosity [Pa s]','Torque [μN m]']:
                raise ValueError('Unexpected original units row')
            coordinates=[]
            for line,row in enumerate(reader,3):
                if len(row)!=4:
                    raise ValueError('Unexpected original CSV width')
                gamma=float(row[0])  # The only conversion of original numeric cells.
                if not math.isfinite(gamma) or gamma<=0:
                    raise ValueError('Nonpositive/nonfinite original input coordinate')
                coordinates.append({'source_line_1based':line,'gamma_per_s':gamma,
                                    'gamma_raw_token':row[0]})
            key=member.split('/')[1]
            values=[row['gamma_per_s'] for row in coordinates]
            diffs=[b-a for a,b in zip(values,values[1:])]
            role=('primary_method_documented' if key in PAPER_KEYS else
                  'legacy2019_exclude' if key=='NaCl_0_XG_2019' else
                  'extra0p05_not_in_paper11_exclude_pending_role')
            records.append({'member':member,'preparation_group':key,'role_prepared_only':role,
                            'source_member_sha256':hashlib.sha256(payload).hexdigest(),
                            'source_rows':len(values),'unique_input_coordinates':len(set(values)),
                            'gamma_min':min(values),'gamma_max':max(values),
                            'positive_steps':sum(v>0 for v in diffs),
                            'negative_steps':sum(v<0 for v in diffs),
                            'equal_steps':sum(v==0 for v in diffs),
                            'coordinates':coordinates,
                            'no_numeric_stress_viscosity_torque_decoded':True,
                            'rheology_runs_per_group_observed':1,
                            'settling_runs_not_counted_as_rheology_replicates':True})
    if {r['preparation_group'] for r in records if r['role_prepared_only']=='primary_method_documented'}!=PAPER_KEYS:
        raise ValueError('Original method11 preparation-group set missing')
    result={'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'role':'SOURCE_INPUT_ONLY_NO_FIT_NO_RESPONSE_REVEAL',
            'source_zip_sha256':ZIP_SHA,'records':records,
            'primary_groups':11,'all_flow_files':13,
            'source_unit_mapping':'Var1 shear_rate[1/s],Var2 stress[Pa],Var3 viscosity[Pa.s],Var4 torque[µNm] by linked original method/readme order',
            'temperature_C':21,'material_state':'aqueous XG1g/L plus NaCl, no photocure protocol',
            'response_cells_numerically_decoded':0,
            'temperature_or_sample_rep_not_invented':True}
    with (HERE/'COORDINATES_ONLY_V2.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps([{k:r[k] for k in ('preparation_group','role_prepared_only','source_rows',
          'unique_input_coordinates','gamma_min','gamma_max','positive_steps','negative_steps','equal_steps')}
          for r in records],ensure_ascii=False))


if __name__=='__main__':
    main()
