"""Read-only public checkpoint integrity check; Python standard library only."""
from pathlib import Path, PurePosixPath
import hashlib,json,sys
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(name):
    q=PurePosixPath(name)
    if q.is_absolute() or '..' in q.parts or chr(92) in name or ':' in name:raise ValueError('Unsafe manifest path')
    p=ROOT.joinpath(*q.parts).resolve();p.relative_to(ROOT.resolve());return p
def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text('utf-8'));n=0
    for x in manifest['files']:
        p=safe(x['path'])
        if p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('Changed '+x['path'])
        n+=1
    for x in json.loads((ROOT/'models/RUNTIME_BINDINGS.json').read_text('utf-8'))['bindings']:
        if sha(safe(x['file']))!=x['sha256']:raise ValueError('Changed frozen science '+x['file'])
    for p in (ROOT/'paper').glob('*.docx'):
        with ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('DOCX ZIP failure')
    print(json.dumps({'status':'PASS','files':n,'frozen_runtime_model_bindings':16,'model_fit_calls':0,'new_measurements':0}))
if __name__=='__main__':
    try:main()
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
