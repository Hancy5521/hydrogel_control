"""Fetch hash-bound public source files once; never upload, retry or install."""
import argparse,hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-map',type=Path,default=Path(__file__).with_name('SOURCE_CELL_MAP.json'))
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();mapping=json.loads(a.source_map.read_text(encoding='utf-8'))
    if a.receipt.exists():raise FileExistsError('Receipt exists; do not overwrite a prior attempt')
    a.raw_dir.mkdir(parents=True,exist_ok=True);records=[]
    for key,spec in mapping['source_files'].items():
        if not spec['required_for_curve_rebuild']:continue
        if Path(key).name!=key or '/' in key or '\\' in key:raise ValueError('Unsafe file key')
        target=a.raw_dir/key
        r={'source_file_key':key,'actual_utc':datetime.now(timezone.utc).isoformat(),'url':spec['retrieval_url']}
        try:
            if target.exists():
                if not target.is_file() or target.stat().st_size!=spec['bytes'] or sha(target)!=spec['sha256']:raise ValueError('Existing file differs; it was not replaced')
                r['status']='VERIFIED_EXISTING_NO_DOWNLOAD'
            else:
                req=urllib.request.Request(spec['retrieval_url'],headers={'User-Agent':'Hydrogel-source-reproduction/1.0'})
                with urllib.request.urlopen(req,timeout=45) as response:
                    raw=response.read(spec['bytes']+1)
                if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Original version/bytes differ; stop without substituting data')
                with target.open('xb') as handle:handle.write(raw)
                r['status']='DOWNLOADED_EXACT_SOURCE'
            r['sha256']=spec['sha256'];records.append(r)
        except Exception as exc:
            r.update(status='FAILED_NO_RETRY',error=f'{type(exc).__name__}: {exc}');records.append(r);break
    status='PASS' if len(records)==sum(s['required_for_curve_rebuild'] for s in mapping['source_files'].values()) and all(r['status']!='FAILED_NO_RETRY' for r in records) else 'STOPPED'
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x',encoding='utf-8') as handle:json.dump({'actual_utc':datetime.now(timezone.utc).isoformat(),'status':status,'source_map_sha256':sha(a.source_map),'records':records,'upstream_terms_not_relicensed':True},handle,indent=2)
    print(json.dumps({'status':status,'files':len(records),'no_retries':True}))
    if status!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
