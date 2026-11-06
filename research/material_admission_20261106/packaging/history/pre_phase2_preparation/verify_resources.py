"""Standard-library resource SHA/size and optional ZIP CRC check; runs no experiment."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def safe(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:raise ValueError('Unsafe manifest path')
    return p

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--zip',type=Path);args=a.parse_args()
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text('utf-8'));expected={}
    for e in manifest['files']:
        p=ROOT.joinpath(*safe(e['path']).parts).resolve();p.relative_to(ROOT)
        if e['path'] in expected:raise ValueError('Duplicate manifest path')
        if p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']:raise ValueError('Resource changed: '+e['path'])
        expected[e['path']]=e['sha256']
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if actual!=set(expected)|{'PACKAGE_MANIFEST.json'}:raise ValueError('Missing or extra local payload')
    crc=None
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC failure')
            names=z.namelist();wanted={manifest['fixed_topfolder']+'/'+n for n in actual}
            if len(names)!=len(set(names)) or set(names)!=wanted:raise ValueError('ZIP wrapper/member scope differs')
            for name in names:
                safe(name)
                rel=name[len(manifest['fixed_topfolder'])+1:]
                pin=digest(ROOT/rel) if rel=='PACKAGE_MANIFEST.json' else expected[rel]
                if hashlib.sha256(z.read(name)).hexdigest()!=pin:raise ValueError('ZIP payload changed')
        crc=True
    print(json.dumps({'status':'PASS','payload_files':len(expected),'zip_CRC_and_bytes':crc,'model_or_experiment_runs':0,'native_property_decoding':0}))

if __name__=='__main__':main()
