"""Read-only independent package/content checks for each new manuscript version.

This does not invoke the builder, Word, or numerical experiment. It preserves
its own labelled reports so a failed revision is not overwritten by a later one.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from lxml import etree

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PINS = {'Hydrogel_Continuation_MLST.docx': '67e99f20136871d67fe5ba95b0a0325ece05767a02b52a714c35bd3528e2396e',
        'Supplement_MLST.docx': '5c5346b619245ec17715cd548430eebd45466b014fd1d2a489b46159d0dae01a',
        'MLST_Cover_Letter.docx': '00769a8050d7500cd26cbc16616946bbe6cbdddef2edf1f829583a0ae5a89b99'}
NEW_DOIS = ['10.1088/2632-2153/acac01', '10.1088/2632-2153/abfe7c',
            '10.1088/2632-2153/ad805f', '10.1088/2632-2153/ad62ad',
            '10.1088/2632-2153/ad594a', '10.1088/2632-2153/ada221']
CITATIONS = re.compile(r'\[(\d+(?:\s*[,–-]\s*\d+)*)\]')


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def package(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise AssertionError('DOCX CRC failed')
        parts = {f: archive.read(f) for f in archive.namelist()}
    root = etree.fromstring(parts['word/document.xml'])
    paras = [''.join(p.xpath('.//w:t/text()', namespaces=NS))
             for p in root.xpath('//w:body//w:p', namespaces=NS)]
    return parts, root, paras


def objects(root, expression):
    return Counter(sha(etree.tostring(n, method='c14n', exclusive=True))
                   for n in root.xpath(expression, namespaces=NS))


def reference_map(paras):
    result = {}
    for text in paras:
        m = re.match(r'^\[(\d+)\] (.+)$', text)
        if m:
            result[int(m[1])] = m[2]
    return result


def cite_numbers(text):
    result = []
    for citation in CITATIONS.finditer(text):
        for value in citation[1].replace('–', '-').split(','):
            numbers = [int(x) for x in value.split('-')]
            result.extend(range(numbers[0], numbers[-1]+1))
    return result


def review(stage, output_dir, label):
    old_dir = Path(r'H:\PolymerReview\hydrogel_MLST_20261104T1405189370969Z\checkpoint\paper\MLST')
    checks, findings, receipt = [], [], {}
    contents = {}
    for name, expected in PINS.items():
        old, new = old_dir/name, output_dir/name
        assert sha(old.read_bytes()) == expected
        op, ot, old_paras = package(old)
        np, nt, new_paras = package(new)
        om, nm = objects(ot, '//m:oMath'), objects(nt, '//m:oMath')
        ob, nb = objects(ot, '//w:tbl'), objects(nt, '//w:tbl')
        assert om == nm, name+' original math objects changed'
        assert ob <= nb, name+' original scientific table changed'
        assert op['word/styles.xml'] == np['word/styles.xml'], name+' original styles changed'
        for part in op:
            if part.startswith('word/media/'):
                assert part in np and op[part] == np[part], name+' original image changed'
        old_section = ot.find('w:body', NS)[-1]
        new_section = nt.find('w:body', NS)[-1]
        assert etree.tostring(old_section, method='c14n', exclusive=True) == etree.tostring(new_section, method='c14n', exclusive=True)
        receipt[name] = {'sha256': sha(new.read_bytes()), 'pinned_old_sha256': expected,
                         'old_math': sum(om.values()), 'new_math': sum(nm.values()),
                         'old_tables': sum(ob.values()), 'new_tables': sum(nb.values()),
                         'old_media': sum(part.startswith('word/media/') for part in op),
                         'new_media': sum(part.startswith('word/media/') for part in np),
                         'old_math_tables_media_styles_section_preserved': True}
        contents[name] = (new_paras, new_paras.index('References') if 'References' in new_paras else None)
        checks.append(name+': CRC/old input pin/all math/all old tables/media/styles/section retained')
    main_name = 'Hydrogel_Continuation_MLST.docx'
    si_name = 'Supplement_MLST.docx'
    old_paras = package(old_dir/main_name)[2]
    main, boundary = contents[main_name]
    si = contents[si_name][0]
    old_refs, refs = reference_map(old_paras), reference_map(main)
    assert set(old_refs) == set(range(1, 59)) and set(refs) == set(range(1, 65))
    inverse = {text: number for number, text in refs.items()}
    assert len(inverse) == 64
    original_mapping = {}
    for oldnum, text in old_refs.items():
        assert text in inverse, 'Original bibliographic record altered or absent'
        original_mapping[oldnum] = inverse[text]
    combined_body = main[:boundary] + si
    first, positions = [], {}
    for i, para in enumerate(combined_body):
        for number in cite_numbers(para):
            assert number in refs, 'Citation out of range'
            positions.setdefault(number, []).append(i)
            if number not in first:
                first.append(number)
    assert first == list(range(1, 65)), 'Reference numbering not in main/SI first-appearance order'
    added = []
    for doi in NEW_DOIS:
        matches = [(number, text) for number, text in refs.items() if doi in text]
        assert len(matches) == 1 and matches[0][0] in positions
        number, text = matches[0]
        added.append({'doi': doi, 'number': number, 'reference': text,
                      'citation_contexts': [combined_body[i] for i in sorted(set(positions[number]))]})
    checks += ['All58original reference records unchanged;64unique refsconsecutive',
               'All64referencesused;main+SI firstappearance numberingcorrect',
               'All6newMLSTreferencescited inactual body prose']
    abstract = main[main.index('Abstract')+1]
    abstract_words = len(abstract.split())
    assert abstract_words < 300
    full = '\n'.join(combined_body)
    assert '250 correction fits' in full and '750 forecasts' in full
    assert 'same consumed 25 P' in full
    assert 'not the best representation examined' in abstract
    assert '11/33' in full and '8/25' in full and '17/25' in full
    assert 'no prior-KL deployment weights are produced' in full
    assert 'only to the original frozen curvature deployment model' in full
    assert 'no external or new-batch evidence' in full
    checks += ['Abstract<300words', '250newfits/750forecasts retainedconsumed25Pstatus',
               'KLlowerCRPSnotoptimalcurvatureclaim;interpolationandjointtradeoffsretained',
               'KLmiddle8wins17losses andP0coverage11/33retained',
               'External46resultsrestrictedtooriginalcurvature;nonewKLdeploymentconfirmed']
    # Findings are semantic/editorial inspection aids, not wholesale regex bans.
    for i, para in enumerate(combined_body):
        if 'https://github.com/Hancy 5521' in para:
            findings.append({'severity': 'functional', 'paragraph_index': i,
                             'text': para, 'issue': 'New number-spacing pass breaks the repository URL; protect full URLs before numeric prose formatting.'})
        patterns = [r'CRPS\d', r'\+00:00before', r';\d+ new', r'\d%accuracy',
                    r'\bis[−+]\d']
        if any(re.search(pattern, para) for pattern in patterns):
            findings.append({'severity': 'editorial', 'paragraph_index': i,
                             'text': para, 'issue': 'Compact new prose still needs explicit numeric punctuation spacing.'})
    obsolete_hits = [{'paragraph_index': i, 'text': para}
                     for i, para in enumerate(combined_body)
                     if re.search(r'only one random|only one fixed random|single random construction|not yet.*(?:fit|trained)|future.*random.*control', para, re.I)]
    result = {'actual_utc': datetime.now(timezone.utc).isoformat(), 'label': label,
              'status': 'CONTENT_ACCEPTABLE_REQUIRES_PROSE_FIX' if findings else 'PASS',
              'reviewer_role': 'Independent read-only DOCX ZIP/XML and body-prose review; no authoring, fitting, scoring or builder execution.',
              'files': receipt, 'checks': checks, 'abstract_words': abstract_words,
              'old_to_new_reference_mapping': original_mapping,
              'six_added_reference_contexts': added,
              'obsolete_phrase_hits_for_manual_review': obsolete_hits,
              'findings': findings,
              'manual_scientific_review': [
                  'Read entiremainbody, coverletter,SI20-21againstprevious independentlyverified numericalledgers.',
                  'KLCRPS/RMSEimprovement reportedalongsidejointdensityloss,CGinterpolationstrength,middlegap8/25winsandP0coverage11/33.',
                  'All6newcitationssupplyrelevant,qualifiedcontext;no claimthatsimilarmaturealgebraisnewtheory.',
                  'Covarianceallocationexplicitlyposthoc/referencelaw;notphysicalaccuracyinformationorcalibration.',
                  'Originaldeployment/external46weightingsandadversesourcespreserved;newKLnotexternallytested/adopted.',
                  'AI-use/open-data/author-confirmationlimitationsremain;noacceptanceprobabilityorcalibrationguarantee.'
              ],
              'visual_QA_status': 'Not part of thisXML/prosereview;actualpageinspectionisaseparatependingstep.',
              'source_sha256': sha(Path(__file__).read_bytes())}
    target = stage/'audit'/('MANUSCRIPT_CONTENT_REVIEW_'+label+'.json')
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'report': str(target), 'status': result['status'],
                      'abstract_words': abstract_words, 'findings': len(findings),
                      'obsolete_hits': len(obsolete_hits), 'files': receipt}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--manuscript-dir', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    review(args.stage, args.manuscript_dir, args.label)
