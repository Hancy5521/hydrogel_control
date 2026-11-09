"""Download admitted raw originals; disclose headers, never property numbers."""
from __future__ import annotations
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def one(spec):
    result = {**spec, 'started_utc': utc(), 'numeric_property_cells_decoded': 0}
    try:
        request = urllib.request.Request(spec['url'], headers={'User-Agent': 'MaterialResearchSourceAdmission/1.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError('Original exceeded2MiB per-file declared bound')
            result.update(status_code=response.status, final_url=response.url.split('?', 1)[0],
                          redirect_query_retained=False, content_type=response.headers.get('Content-Type'))
        path = ROOT / spec['source'] / spec['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        actual_md5 = hashlib.md5(raw).hexdigest()
        result.update(path=str(path.relative_to(ROOT)), bytes=len(raw), md5=actual_md5,
                      sha256=hashlib.sha256(raw).hexdigest())
        if actual_md5 != spec['expected_md5']:
            raise ValueError('Official original MD5 mismatch; retained quarantined original, no header parse')
        # A lexical header-only guard: a numeric-leading field is redacted before
        # display and never converted to float. Data rows are never parsed.
        prefix = raw.splitlines()[:2]
        decoded = [line.decode('utf-8-sig', errors='replace') for line in prefix]
        first = decoded[0] if decoded else ''
        fields = re.split('[;,\t]', first)
        result['first_line_tokens'] = [
            '<numeric-redacted>' if re.match(r'^\s*[\"\s]*[+\-]?(?:\d|\.\d)', token) else token
            for token in fields]
        if len(decoded) > 1 and not re.search('[0-9]', decoded[1]):
            result['second_line_nonnumeric_only'] = decoded[1]
        result['raw_line_count'] = raw.count(b'\n')
        result['status'] = 'ORIGINAL_MD5_PASS_HEADER_ONLY'
    except Exception as error:
        result.update(status='FAILED_NO_RETRY', error_type=type(error).__name__, error=str(error))
    result['finished_utc'] = utc()
    return result


def main():
    darus = json.loads((ROOT / 'darus_smp_metadata.json').read_text())['data']['latestVersion']
    pcl = json.loads((ROOT / 'zenodo_pcl_metadata.json').read_text())
    specs = []
    for item in darus['files']:
        f = item['dataFile']
        if f.get('originalFileName', '').startswith('frequency_sweep_'):
            if item['restricted']:
                raise PermissionError('Restricted file never downloaded')
            specs.append({'source': 'darus_smp_originals', 'filename': f['originalFileName'],
                          'url': 'https://darus.uni-stuttgart.de/api/access/datafile/' + str(f['id']) + '?format=original',
                          'expected_md5': f['md5'], 'role': 'measured frequency-sweep original only'})
    for f in pcl['files']:
        if f['key'].startswith(('TensileTestAtTdef_', 'DMTA_')):
            specs.append({'source': 'pcl_originals', 'filename': f['key'], 'url': f['links']['self'],
                          'expected_md5': f['checksum'].split(':')[1], 'role': 'measured tensile/DMA original only'})
    if len(specs) != 18:
        raise AssertionError('Exactly6 native frequency CSV plus9 tensile+3DMA originals declared')
    with (ROOT / 'ORIGINAL_DOWNLOAD_PLAN.json').open('x', encoding='utf-8') as stream:
        json.dump({'actual_utc': utc(), 'specifications': specs, 'files': 18,
                   'numeric_property_cells_decoded': 0, 'fits': 0,
                   'excluded': 'mastercurves/shift factors/DSC/shape-memory cycles are not new raw rheology measurements'}, stream, indent=2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(one, specs))
    with (ROOT / 'ORIGINAL_DOWNLOAD_RECEIPTS.json').open('x', encoding='utf-8') as stream:
        json.dump({'actual_utc': utc(), 'records': result}, stream, ensure_ascii=False, indent=2)
    for row in result:
        print(json.dumps({k: v for k, v in row.items() if k not in ['url', 'expected_md5', 'role']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
