"""Bounded four-source metadata admission; no numerical property parser."""
from __future__ import annotations
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'dryad_alginate': 'https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.cjsxksnnh',
    'darus_smp': 'https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi%3A10.18419%2Fdarus-2021',
    'zenodo_pcl': 'https://zenodo.org/api/records/10995304',
    'zenodo_alginate_restricted': 'https://zenodo.org/api/records/21701484',
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fetch(item):
    name, url = item
    record = {'source': name, 'url': url, 'started_utc': utc(), 'role': 'metadata_only',
              'numeric_property_cells_decoded': 0, 'fits': 0}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'MaterialResearchMetadata/1.0'})
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError('Metadata larger than declared8MiB bound')
            record.update(status_code=response.status, final_url=response.url,
                          content_type=response.headers.get('Content-Type'))
        payload = json.loads(raw)
        path = ROOT / (name + '_metadata.json')
        with path.open('xb') as stream:
            stream.write(raw)
        record.update(status='DOWNLOADED_METADATA_ONLY', path=path.name, bytes=len(raw),
                      sha256=hashlib.sha256(raw).hexdigest(), top_level_keys=list(payload))
    except Exception as error:
        record.update(status='FAILED_NO_RETRY', error_type=type(error).__name__, error=str(error))
    record['finished_utc'] = utc()
    return record


def main():
    plan = {'actual_utc': utc(), 'scope': 'four official metadata endpoints once; source/header/input-only',
            'sources': SOURCES, 'numeric_property_cells_decoded': 0, 'fits': 0,
            'old_closed_sources_not_reopened': ['GelMA21759156', 'PEO15384569', 'Carbopol29376416',
                                                'restricted old Si-HA', 'Richbourg', 'NIST CHIPS'],
            'restrictions': 'No restricted file access, no contact, no raw numeric property Y/fit, no saved results execution'}
    with (ROOT / 'PLAN.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    with (ROOT / 'METADATA_RECEIPTS.json').open('x', encoding='utf-8') as stream:
        json.dump({'actual_utc': utc(), 'records': records}, stream, ensure_ascii=False, indent=2)
    for row in records:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
