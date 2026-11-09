import pathlib,json,datetime,zipfile,xml.etree.ElementTree as ET,collections,hashlib,re
ROOT=pathlib.Path(__file__).resolve().parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
save=lambda n,v:(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
save('INPUT_IDENTITY_PLAN.json',{'utc':now(),'21759156_allow':['A Conc','B rep','C Angular Frequency or Shear rate; numeric coordinate only'],'21759156_excluded':['D:E all response numerical cells','print workbook all rows beyond header (categorical exposure already disclosed)'],'19699588_allow':'README and header-only CSV lines; no numeric rows','15384569_allow':'README metadata only; Origin data not opened','no_fit':True})
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
res=[]
for key in ['01_frequency_sweep_modulus_data.xlsx.xlsx','02_flow_curve_viscosity_data.xlsx.xlsx']:
 z=zipfile.ZipFile(ROOT/'21759156'/key); tree=ET.fromstring(z.read('xl/worksheets/sheet1.xml')); groups=collections.Counter(); coords={}; string_inputs=[]
 for row in tree.findall('m:sheetData/m:row',NS):
  if row.attrib['r']=='1': continue
  inp={}
  for c in row.findall('m:c',NS):
   col=re.match('[A-Z]+',c.attrib['r']).group()
   if col not in ['A','B','C']: continue
   v=c.find('m:v',NS)
   if v is not None: inp[col]=v.text
  if set(inp)=={'A','B','C'}:
   k=(inp['A'],inp['B']);groups[k]+=1;coords.setdefault(k,set()).add(inp['C'])
 res.append({'file':key,'rows':sum(groups.values()),'formulation_keys':sorted(set(k[0] for k in groups)),'rep_keys':sorted(set(k[1] for k in groups)),'curve_key_count':len(groups),'per_key':[{'conc_input_raw':k[0],'rep_input_raw':k[1],'row_count':v,'distinct_coordinate_count':len(coords[k])} for k,v in sorted(groups.items())],'responses_decoded':0,'units_from_file':'not present in header'})
for rid,key in [('19699588','Rheological_Data_For_Syntactic_Foam.zip'),('15384569','20250512-Rheology-Fig2.zip')]:
 z=zipfile.ZipFile(ROOT/rid/key);r={'record':rid,'readmes':[],'csv_headers':[]}
 for name in z.namelist():
  if name.lower().endswith(('.md','.txt')): r['readmes'].append({'member':name,'text':z.read(name).decode('utf-8-sig',errors='replace')})
  elif name.lower().endswith('.csv'):
   b=z.read(name); lines=b.splitlines(); headers=[]
   # Return only lines containing explicit column name, never following values.
   for line in lines:
    if b'Shear Rate' in line or b'Viscosity' in line and b'Temperature' in line:
     headers.append(line.decode('utf-8-sig',errors='replace'))
   r['csv_headers'].append({'member':name,'header_lines':headers[:3],'sha256':hashlib.sha256(b).hexdigest()})
 res.append(r)
save('INPUT_IDENTITY_RESULT.json',res)
for r in res:
 if 'csv_headers' in r and len(r['csv_headers'])>3:
  r=dict(r);r['csv_header_file_count']=len(r['csv_headers']);r['csv_headers']=r['csv_headers'][:2]
 print(json.dumps(r,ensure_ascii=False))
