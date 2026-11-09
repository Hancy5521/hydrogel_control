import json, hashlib, urllib.request, datetime, pathlib, zipfile, io, re, xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parent
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,v): (ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
allow={
 '21759156':['01_frequency_sweep_modulus_data.xlsx.xlsx','02_flow_curve_viscosity_data.xlsx.xlsx','03_print_width_and_printability_data.xlsx.xlsx'],
 '17748668':['figure7_frequency_sweep.csv'],
 '19699588':['Rheological_Data_For_Syntactic_Foam.zip'],
 '15384569':['20250512-Rheology-Fig2.zip','Readme-Experimental conditions.txt']}
save('DOWNLOAD_SCHEMA_PLAN.json',{'utc':now(),'allowlist':allow,'read_scope':'ZIP names/CRC/size only. XLSX sheet names/dimensions and textual cells in first 8 rows only; numeric cells not decoded. CSV first header line only; numeric target arrays not decoded. Text README stored pending scoped metadata read. No target distributions, curve plots, interpolation outputs, model notebooks or score tables. No fit/score/seed search.'})
out=[]
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
for rid,keys in allow.items():
 d=json.loads((ROOT/(rid+'_metadata.json')).read_text(encoding='utf-8'))
 folder=ROOT/rid; folder.mkdir(exist_ok=True)
 for key in keys:
  f=next(x for x in d['files'] if x['key']==key); url=f['links']['self']
  rec={'record':rid,'key':key,'url':url,'utc':now(),'official_checksum':f['checksum']}
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'polymer-metadata-admission/1.0'}),timeout=35) as response: b=response.read(); rec['http_status']=response.status
   (folder/key).write_bytes(b)
   rec.update({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'md5':hashlib.md5(b).hexdigest(),'checksum_pass':f['checksum']=='md5:'+hashlib.md5(b).hexdigest()})
   if key.endswith('.zip'):
    z=zipfile.ZipFile(io.BytesIO(b)); rec['members']=[{'name':v.filename,'size':v.file_size,'crc32':format(v.CRC,'08x')} for v in z.infolist()]
   elif '.xlsx' in key:
    z=zipfile.ZipFile(io.BytesIO(b)); wb=ET.fromstring(z.read('xl/workbook.xml'))
    rec['sheets']=[v.attrib for v in wb.findall('m:sheets/m:sheet',NS)]
    ss=[]
    if 'xl/sharedStrings.xml' in z.namelist():
     ss=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS)]
    rec['schema_text']=[]
    for member in z.namelist():
     if re.fullmatch(r'xl/worksheets/sheet\d+\.xml',member):
      tree=ET.fromstring(z.read(member)); dim=tree.find('m:dimension',NS); vals=[]
      for row in tree.findall('m:sheetData/m:row',NS):
       if int(row.attrib['r'])>8: continue
       for c in row.findall('m:c',NS):
        t=c.attrib.get('t'); val=None
        if t=='s': val=ss[int(c.find('m:v',NS).text)]
        elif t=='inlineStr': val=''.join(c.find('m:is',NS).itertext())
        if val is not None and not re.fullmatch(r'[\s\d.,+eE%−-]+',val): vals.append({'cell':c.attrib['r'],'text':val})
      rec['schema_text'].append({'member':member,'dimension':None if dim is None else dim.attrib,'first8_text_cells':vals})
   elif key.endswith('.csv'):
    rec['header']=b.splitlines()[0].decode('utf-8-sig')
   else: rec['content_read_scope']='stored bytes only, not read'
  except Exception as e: rec['error']=str(e); rec['retry']='none'
  out.append(rec)
save('DOWNLOAD_SCHEMA_RECEIPTS.json',out)
for r in out:
 print(json.dumps(r,ensure_ascii=False))
