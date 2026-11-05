"""Fetch two exact public data assets into a new directory; no model execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
META = ('ELIGIBLE_METADATA.json', 'PILOT_ELIGIBILITY.json', 'DATA_MANIFEST.json',
        'SOURCE_GATE.json', 'SOURCE_GATE_PILOT_ADDENDUM.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = json.loads((ROOT / 'pilot/PREFIT_SOURCE_RECEIPT.json').read_text('utf-8'))['source_input_sha256']
    for name in META:
        if sha(ROOT / 'data' / name) != pins[name]:
            raise ValueError('Pinned source metadata changed: ' + name)
    manifest = json.loads((ROOT / 'data/DATA_MANIFEST.json').read_text('utf-8'))
    receipts = []
    try:
        for name in META:
            shutil.copyfile(ROOT / 'data' / name, output / name)
        for item in manifest['datasets']:
            name = item['file']
            if Path(name).name != name or item['bytes'] > 100_000_000:
                raise ValueError('Unexpected pinned data asset')
            url = item['stable_download_url']
            if not url.startswith('https://ndownloader.figshare.com/files/'):
                raise ValueError('Only the reviewed stable Figshare URLs are allowed')
            request = urllib.request.Request(url, headers={'User-Agent': 'materials-research-reproduction'})
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=30) as source, (output / name).open('xb') as target:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > item['bytes']:
                        raise ValueError('Asset exceeds registered byte count')
                    digest.update(chunk)
                    target.write(chunk)
            if count != item['bytes'] or digest.hexdigest() != item['sha256']:
                raise ValueError('Pinned size/SHA mismatch for ' + name)
            # Signed redirect queries/headers are intentionally not recorded.
            receipts.append({'file': name, 'stable_url': url, 'bytes': count,
                             'sha256': digest.hexdigest(), 'actual_utc': datetime.now(timezone.utc).isoformat()})
        status = {'status': 'PINNED_PUBLIC_DATA_RETRIEVED', 'receipts': receipts,
                  'metadata_sha256': {name: sha(output / name) for name in META},
                  'new_measurements': 0, 'model_execution': False}
        with (output / 'RETRIEVAL_RECEIPT.json').open('x', encoding='utf-8') as stream:
            json.dump(status, stream, indent=2)
        print(json.dumps({'status': status['status'], 'assets': len(receipts), 'new_measurements': 0}))
    except Exception as error:
        with (output / 'RETRIEVAL_FAILED.json').open('x', encoding='utf-8') as stream:
            json.dump({'actual_utc': datetime.now(timezone.utc).isoformat(),
                       'error_type': type(error).__name__, 'message': str(error),
                       'completed_assets': receipts, 'automatic_retry': False}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
