import json, hashlib, urllib.request, datetime, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
IDS = ['21759156', '17748668', '19699588', '15384569']
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
plan = {
 'utc_before_metadata': now(), 'scope': 'exactly four Zenodo metadata records; raw curve bytes only after source/schema plan; no response numbers decoded, no fit/score/seed search',
 'records': IDS, 'protected': ['old MLST25','2944 consumed groups','permanent1024','Hu','TuneGlue','Q5','86','21eval','AMBench'],
 'allowed_metadata': ['record title','authors','version','license','relations','description','file names','file size','file checksums','file URLs'],
 'selection': 'choose one by authentic curve/schema/recipe/state/replicate support and rights, never model outcomes',
 'prior_exposure': '19699588 article title/PMC link appeared in old 20261022 WEB_TOOL_RESPONSES.json search excerpt; exact new original record/files not found in docs/artifacts. Other exact identifiers not found. Complete global novelty not claimed.'
}
save('PLAN.json', plan)
receipts=[]
for rid in IDS:
    url='https://zenodo.org/api/records/'+rid
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'polymer-metadata-admission/1.0'}), timeout=35) as f:
            b=f.read(); status=f.status
        (ROOT / (rid+'_metadata.json')).write_bytes(b)
        d=json.loads(b); m=d.get('metadata',{})
        rec={'id':rid,'url':url,'status':status,'utc':now(),'bytes':len(b),'sha256':sha(b),'title':m.get('title'),'version':m.get('version'),'license':m.get('license'),'access_right':m.get('access_right'),'publication_date':m.get('publication_date'),'created':d.get('created'),'modified':d.get('modified'),'doi':d.get('doi'),'conceptdoi':d.get('conceptdoi'),'related_identifiers':m.get('related_identifiers'),'files':[{'key':v.get('key'),'size':v.get('size'),'checksum':v.get('checksum'),'links':v.get('links')} for v in d.get('files',[])]}
    except Exception as e: rec={'id':rid,'url':url,'utc':now(),'error':str(e),'retry':'none'}
    receipts.append(rec)
save('METADATA_RECEIPTS.json',receipts)
print(json.dumps(receipts,ensure_ascii=False,indent=2))
