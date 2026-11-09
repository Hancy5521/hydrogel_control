import pathlib,zipfile,json,datetime,hashlib,re,collections
ROOT=pathlib.Path(__file__).resolve().parent
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
save('RESIN_HEADER_PLAN.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allowed':'CSV BOM/encoding, exact header lines containing Viscosity and Temperature, corresponding immediately next unit line only if it contains [ or cP; no numeric data rows, ZIP member hash identity; no targets decoded','role':'public polymer-response stress-test; never hydrogel or independent synthesis-batch confirmation'})
z=zipfile.ZipFile(ROOT/'19699588'/'Rheological_Data_For_Syntactic_Foam.zip'); rec=[]; groups=collections.Counter()
for name in z.namelist():
 if not name.lower().endswith('.csv'):continue
 b=z.read(name);enc='utf-16' if b.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'; lines=b.decode(enc,errors='replace').splitlines();headers=[]
 for i,l in enumerate(lines):
  if 'Viscosity' in l and 'Temperature' in l:
   headers.append({'line_1based':i+1,'header':l})
   if i+1<len(lines) and ('[' in lines[i+1] or 'cP' in lines[i+1]):headers[-1]['units']=lines[i+1]
 m=re.search(r'/(0_\d+gcc_\d+%|Neat)_Resin',name);groups[m.group(1) if m else 'unparsed']+=1
 rec.append({'member':name,'sha256':hashlib.sha256(b).hexdigest(),'encoding':enc,'headers':headers})
out={'per_recipe_filename_run_count':dict(groups),'csv_count':len(rec),'distinct_member_sha256':len({r['sha256'] for r in rec}),'members':rec}
save('RESIN_HEADER_RESULT.json',out)
print(json.dumps({**{k:v for k,v in out.items() if k!='members'},'first_header_example':rec[0]},ensure_ascii=False,indent=2))
