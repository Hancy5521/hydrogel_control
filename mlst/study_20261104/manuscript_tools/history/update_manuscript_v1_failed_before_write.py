"""Surgical, audited MLST manuscript revision; never overwrites source DOCX.

Run only after the root's successful documents operation-start marker.  Requires
the actual marker receipt and independent scientific audit reports.  Original
mathematical objects, tables, media, styles and section layout are preserved.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

from lxml import etree as E
from PIL import Image

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
NS = {'w': W, 'm': M,
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture', 'r': R}
PINS = {
    'Hydrogel_Continuation_MLST.docx': '67e99f20136871d67fe5ba95b0a0325ece05767a02b52a714c35bd3528e2396e',
    'Supplement_MLST.docx': '5c5346b619245ec17715cd548430eebd45466b014fd1d2a489b46159d0dae01a',
    'MLST_Cover_Letter.docx': '00769a8050d7500cd26cbc16616946bbe6cbdddef2edf1f829583a0ae5a89b99',
}
CITE = re.compile(r'\[(\d+(?:\s*[,–-]\s*\d+)*)\]')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def math_hashes(tree):
    return Counter(hashlib.sha256(E.tostring(n, method='c14n', exclusive=True)).hexdigest()
                   for n in tree.xpath('//m:oMath', namespaces=NS))


def table_hashes(tree):
    return Counter(hashlib.sha256(E.tostring(n, method='c14n', exclusive=True)).hexdigest()
                   for n in tree.xpath('//w:tbl', namespaces=NS))


def xml_bytes(tree):
    return E.tostring(tree, encoding='UTF-8', xml_declaration=True, standalone=True)


class Package:
    def __init__(self, path):
        self.path = Path(path)
        if sha(self.path) != PINS[self.path.name]:
            raise PermissionError('Frozen manuscript input identity changed')
        with zipfile.ZipFile(self.path) as archive:
            self.parts = {i.filename: archive.read(i.filename) for i in archive.infolist()}
        self.tree = E.fromstring(self.parts['word/document.xml'])
        self.body = self.tree.find('w:body', NS)
        self.original_math = math_hashes(self.tree)
        self.original_tables = table_hashes(self.tree)
        self.original_media = {k: v for k, v in self.parts.items() if k.startswith('word/media/')}
        self.original_styles = self.parts['word/styles.xml']
        self.original_section = E.tostring(self.body[-1], method='c14n', exclusive=True)
        self.changed_prose = []

    def find(self, prefix):
        found = [p for p in self.body if p.tag == f'{{{W}}}p' and text(p).startswith(prefix)]
        if len(found) != 1:
            raise ValueError(('Unique paragraph prefix required', prefix, len(found)))
        return found[0]

    def paragraph(self, value, kind='body'):
        prefix = {'body': 'Scientific characterization', 'heading': 'Introduction',
                  'subheading': 'Response coordinates', 'caption': 'Table 1',
                  'reference': '[1]'}
        # SI has its own paragraph templates and keeps its established formatting.
        if self.path.name.startswith('Supplement'):
            prefix = {'body': 'This supplement accompanies', 'heading': 'S1 Paired',
                      'subheading': 'S1 Paired', 'caption': 'Table S1',
                      'reference': 'This supplement accompanies'}
        source = self.find(prefix[kind])
        node = E.Element(f'{{{W}}}p')
        props = source.find('w:pPr', NS)
        if props is not None:
            node.append(deepcopy(props))
        run = E.SubElement(node, f'{{{W}}}r')
        oldrun = source.find('w:r', NS)
        if oldrun is not None and oldrun.find('w:rPr', NS) is not None:
            run.append(deepcopy(oldrun.find('w:rPr', NS)))
        t = E.SubElement(run, f'{{{W}}}t')
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        t.text = value
        self.changed_prose.append(node)
        return node

    def replace(self, prefix, value):
        old = self.find(prefix)
        if old.xpath('.//m:oMath | .//w:drawing', namespaces=NS):
            raise PermissionError('Text replacement would destroy scientific object')
        node = deepcopy(old)
        for child in list(node):
            if child.tag != f'{{{W}}}pPr':
                node.remove(child)
        run = E.SubElement(node, f'{{{W}}}r')
        t = E.SubElement(run, f'{{{W}}}t')
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        t.text = value
        self.body.replace(old, node)
        self.changed_prose.append(node)
        return node

    def after(self, prefix, nodes):
        index = self.body.index(self.find(prefix)) + 1
        for node in nodes:
            self.body.insert(index, node)
            index += 1

    def append(self, nodes):
        for node in nodes:
            self.body.insert(len(self.body) - 1, node)

    def save(self, path):
        if math_hashes(self.tree) != self.original_math:
            raise PermissionError('Original OMML changed or was removed')
        if not (self.original_tables <= table_hashes(self.tree)):
            raise PermissionError('Original scientific table changed or was removed')
        if E.tostring(self.body[-1], method='c14n', exclusive=True) != self.original_section:
            raise PermissionError('Section layout changed')
        self.parts['word/document.xml'] = xml_bytes(self.tree)
        if self.parts['word/styles.xml'] != self.original_styles:
            raise PermissionError('Original styles changed')
        for name, payload in self.original_media.items():
            if self.parts[name] != payload:
                raise PermissionError('Original scientific media changed')
        with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name, payload in self.parts.items():
                archive.writestr(name, payload)


def new_table(headers, rows, widths):
    table = E.Element(f'{{{W}}}tbl')
    props = E.SubElement(table, f'{{{W}}}tblPr')
    E.SubElement(props, f'{{{W}}}tblW', {f'{{{W}}}type': 'dxa', f'{{{W}}}w': str(sum(widths))})
    E.SubElement(props, f'{{{W}}}tblLayout', {f'{{{W}}}type': 'fixed'})
    borders = E.SubElement(props, f'{{{W}}}tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        E.SubElement(borders, f'{{{W}}}{side}', {f'{{{W}}}val': 'single', f'{{{W}}}sz': '4', f'{{{W}}}color': 'D9D9D9'})
    margins = E.SubElement(props, f'{{{W}}}tblCellMar')
    for side, size in [('top', 70), ('left', 90), ('bottom', 70), ('right', 90)]:
        E.SubElement(margins, f'{{{W}}}{side}', {f'{{{W}}}w': str(size), f'{{{W}}}type': 'dxa'})
    grid = E.SubElement(table, f'{{{W}}}tblGrid')
    for width in widths:
        E.SubElement(grid, f'{{{W}}}gridCol', {f'{{{W}}}w': str(width)})
    for i, row in enumerate([headers] + rows):
        tr = E.SubElement(table, f'{{{W}}}tr')
        trpr = E.SubElement(tr, f'{{{W}}}trPr')
        if i == 0:
            E.SubElement(trpr, f'{{{W}}}tblHeader')
        E.SubElement(trpr, f'{{{W}}}cantSplit')
        for j, (value, width) in enumerate(zip(row, widths)):
            tc = E.SubElement(tr, f'{{{W}}}tc')
            cp = E.SubElement(tc, f'{{{W}}}tcPr')
            E.SubElement(cp, f'{{{W}}}tcW', {f'{{{W}}}w': str(width), f'{{{W}}}type': 'dxa'})
            E.SubElement(cp, f'{{{W}}}vAlign', {f'{{{W}}}val': 'center'})
            fill = 'E7E6E6' if i == 0 else ('F7F7F7' if i % 2 == 0 else 'FFFFFF')
            E.SubElement(cp, f'{{{W}}}shd', {f'{{{W}}}fill': fill})
            p = E.SubElement(tc, f'{{{W}}}p')
            pp = E.SubElement(p, f'{{{W}}}pPr')
            E.SubElement(pp, f'{{{W}}}jc', {f'{{{W}}}val': 'left' if j == 0 else 'center'})
            E.SubElement(pp, f'{{{W}}}spacing', {f'{{{W}}}before': '0', f'{{{W}}}after': '0', f'{{{W}}}line': '230', f'{{{W}}}lineRule': 'auto'})
            run = E.SubElement(p, f'{{{W}}}r')
            rp = E.SubElement(run, f'{{{W}}}rPr')
            E.SubElement(rp, f'{{{W}}}rFonts', {f'{{{W}}}ascii': 'Times New Roman', f'{{{W}}}hAnsi': 'Times New Roman'})
            E.SubElement(rp, f'{{{W}}}sz', {f'{{{W}}}val': '19'})
            E.SubElement(rp, f'{{{W}}}color', {f'{{{W}}}val': '000000'})
            if i == 0:
                E.SubElement(rp, f'{{{W}}}b')
            E.SubElement(run, f'{{{W}}}t').text = str(value)
    return table


def add_figure(package, figure):
    template = next(p for p in package.body if p.tag == f'{{{W}}}p'
                    and p.xpath('.//w:drawing', namespaces=NS))
    node = deepcopy(template)
    rels = E.fromstring(package.parts['word/_rels/document.xml.rels'])
    ids = {r.get('Id') for r in rels}
    num = max([int(re.sub(r'\D', '', rid) or '0') for rid in ids] + [0]) + 1
    rid = f'rId{num}'
    name = 'mlst_representation_controls.png'
    E.SubElement(rels, f'{{{REL}}}Relationship', {'Id': rid,
                 'Type': R + '/image', 'Target': 'media/' + name})
    package.parts['word/_rels/document.xml.rels'] = xml_bytes(rels)
    package.parts['word/media/' + name] = Path(figure).read_bytes()
    types = E.fromstring(package.parts['[Content_Types].xml'])
    if not any(t.get('Extension') == 'png' for t in types):
        E.SubElement(types, f'{{{CT}}}Default', {'Extension': 'png', 'ContentType': 'image/png'})
    package.parts['[Content_Types].xml'] = xml_bytes(types)
    with Image.open(figure) as image:
        width, height = image.size
    cx = int(6.45 * 914400)
    cy = round(cx * height / width)
    for blip in node.xpath('.//a:blip', namespaces=NS):
        blip.set(f'{{{R}}}embed', rid)
    for extent in node.xpath('.//wp:extent | .//a:xfrm/a:ext', namespaces=NS):
        extent.set('cx', str(cx)); extent.set('cy', str(cy))
    maxid = max([int(n.get('id', '0')) for n in package.tree.xpath('//wp:docPr', namespaces=NS)] + [0])
    for prop in node.xpath('.//wp:docPr | .//pic:cNvPr', namespaces=NS):
        prop.set('id', str(maxid + 1))
        prop.set('name', 'Additional representation controls')
        prop.set('descr', 'Eight fixed random direction scores, prior-KL, rotated curvature and original reference controls on the same25development formulations')
    return node


def unpack_citations(value):
    out = []
    for block in value.replace('–', '-').split(','):
        ends = [int(x.strip()) for x in block.split('-')]
        out.extend(range(ends[0], ends[-1] + 1))
    return out


def tidy_new_prose(package):
    """Space only newly authored prose; never alter old prose, tables or math."""
    for p in package.changed_prose:
        if p.getparent() is None or p.xpath('.//m:oMath | .//w:drawing', namespaces=NS):
            continue
        for t in p.xpath('.//w:t', namespaces=NS):
            if not t.text:
                continue
            value = t.text
            # Preserve exact scientific identifiers and actualUTC stamps while
            # fixing compact numeric prose; never insert spaces into hashes.
            value = re.sub(r'(commit)(?=[0-9a-f]{40})', r'\1 ', value)
            protected = []
            def protect(match):
                protected.append(match.group(0))
                return 'ZZPROTECTED' + chr(65 + len(protected)) + 'ZZ'
            value = re.sub(r'\d{4}-\d{2}-\d{2}T[0-9:.+\-]+|(?<![0-9a-f])[0-9a-f]{40,64}(?![0-9a-f])', protect, value)
            value = re.sub(r'(?<=[a-z])(?=\d)', ' ', value)
            value = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', value)
            for old, new in [('actualUTC', 'actual UTC '), ('TableS', 'Table S'),
                             ('SectionS', 'Section S'), ('Fig.3', 'Fig. 3'),
                             ('Fig.S', 'Fig. S'), ('logRMSE', 'log RMSE'),
                             ('logdensity', 'log density'), ('outerfits', 'outer fits'),
                             ('degree', 'degree'), ('logCholesky', 'log Cholesky'),
                             ('openingTRAIN', 'opening TRAIN'), ('LetAbe', 'Let A be'),
                             ('anchors,Bthe', 'anchors, B the'), ('andV0 the', 'and V0 the'),
                             ('isB A', 'is B A'), ('bytrace(V0)', 'by trace(V0)'),
                             ('inonefold', 'in one fold')]:
                value = value.replace(old, new)
            value = re.sub(r'\b(Fig\.\s*\d+) ([a-z])\b', r'\1\2', value)
            value = value.replace('capacity-matched', 'parameter-count-matched')
            value = value.replace('Capacity-matched', 'Parameter-count-matched')
            for i, original in enumerate(protected, start=1):
                value = value.replace('ZZPROTECTED' + chr(65 + i) + 'ZZ', original)
            t.text = value


def replace_range(nodes, start, end, replacement):
    offset = 0
    spans = []
    for node in nodes:
        value = node.text or ''
        spans.append((node, offset, offset + len(value)))
        offset += len(value)
    touched = [(n, a, b) for n, a, b in spans if b > start and a < end]
    if len(touched) == 1:
        n, a, _ = touched[0]
        n.text = (n.text or '')[:start - a] + replacement + (n.text or '')[end - a:]
    else:
        first, a, _ = touched[0]
        last, z, _ = touched[-1]
        first.text = (first.text or '')[:start - a] + replacement
        for middle, _, _ in touched[1:-1]:
            middle.text = ''
        last.text = (last.text or '')[end - z:]


def renumber(main, supplement, references):
    paragraphs = main.body.xpath('.//w:p', namespaces=NS) + supplement.body.xpath('.//w:p', namespaces=NS)
    order = []
    for p in paragraphs:
        for match in CITE.finditer(text(p)):
            for number in unpack_citations(match.group(1)):
                if number not in references:
                    raise ValueError(('Bracket is not a declared reference', text(p), number))
                if number not in order:
                    order.append(number)
    if set(order) != set(references):
        raise ValueError(('Uncited references', set(references) - set(order)))
    mapping = {old: i + 1 for i, old in enumerate(order)}
    for p in paragraphs:
        nodes = p.xpath('.//w:t', namespaces=NS)
        for match in reversed(list(CITE.finditer(text(p)))):
            replacement = '[' + ', '.join(str(mapping[x]) for x in unpack_citations(match.group(1))) + ']'
            replace_range(nodes, match.start(), match.end(), replacement)
    refs_heading = main.find('References')
    position = main.body.index(refs_heading) + 1
    for old in order:
        main.body.insert(position, main.paragraph(f'[{mapping[old]}] ' + references[old], 'reference'))
        position += 1
    return mapping


def main(args):
    root, source, output = args.run_root.resolve(), args.source_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not args.operation_receipt.is_file() or args.operation_receipt.stat().st_size == 0:
        raise PermissionError('Actual successful document operation marker receipt is required')
    score_path = root / 'experiment/scores/RESULT.json'
    diagnostic_path = root / 'diagnostics/output/RESULT.json'
    audit = read(root / 'audit/FROZEN_AUDIT.json')
    daudit = read(root / 'audit/COVARIANCE_AUDIT.json')
    if (audit['status'] != 'PASS' or audit['score_result_sha256'] != sha(score_path)
            or daudit['status'] != 'PASS' or daudit['diagnostic_result_sha256'] != sha(diagnostic_path)):
        raise PermissionError('Current independent experiment and diagnostic audits must pass')
    if audit['fit_receipts_checked'] != 250 or audit['frozen_forecasts_checked'] != 750:
        raise PermissionError('All250fits and750forecasts must be independently checked')
    scores, diagnostic = read(score_path), read(diagnostic_path)
    protocol = read(root / 'experiment/PROTOCOL.json')
    ledger = read(root / 'literature/SIX_REFERENCE_INSERT_CHECK.json')
    if len(ledger['references']) != 6 or not ledger['all_six_absent_by_doi_and_title']:
        raise PermissionError('Verified six-reference insert ledger required')
    main_doc = Package(source / 'Hydrogel_Continuation_MLST.docx')
    si = Package(source / 'Supplement_MLST.docx')
    cover = Package(source / 'MLST_Cover_Letter.docx')
    old_refs, ref_nodes = {}, []
    for p in main_doc.body:
        match = re.match(r'^\[(\d+)\] (.+)$', text(p))
        if match:
            old_refs[int(match.group(1))] = match.group(2)
            ref_nodes.append(p)
    if set(old_refs) != set(range(1, 59)):
        raise ValueError('All58original references required')
    references = dict(old_refs)
    for ref in ledger['references']:
        references[ref['provisional_number']] = (
            f"{ref['authors']}. {ref['title']}. {ref['journal']}. "
            f"{ref['year']};{ref['volume']}:{ref['article']}. https://doi.org/{ref['doi']}.")
    # Keep one original reference as a style template while building all prose.
    main_doc.replace('Scientific response curves are often',
        'Scientific response curves are often observed only over part of their measurement range. We study six-anchor continuation of hydrogel precursor-ink viscosity using a Gaussian-process reference and a shared five-parameter Gaussian correction in a two-dimensional code, while retaining the complete reference residual. All anchor layouts condition one joint law. On25previously inspected formulations and three layouts, the original finite-scale curvature code obtains retrospective mean marginal continuous ranked probability score0.043277, versus0.055219 for the reference and0.050252 for a conditional Gaussian control. Its gain concentrates in one-sided continuation. Additional protocol-fixed development controls retain the same folds and scoring: eight random codes all have worse macro scores,0.049848–0.065362, whereas a prior-only Karhunen–Loeve code improves to0.037139. Curvature is therefore useful but not the best representation examined; joint density and interpolation also favour different controls. Rotating the same curvature span changes the constrained result slightly. A no-retraining stress test of the original frozen curvature model contains46native curves from four public sources and781higher-rate targets. Liverpool ink scores improve under original group weighting, but three other sources lose to the reference, equal-family weighting reverses the Liverpool advantage over the conditional control, and some intervals severely undercover. No new deployment model is selected from the additional controls. The evidence supports residual-preserving small-capacity adaptation for a defined precursor-ink continuation task, with unresolved representation selection, transfer and uncertainty calibration.')
    # Restore natural spacing around the deliberately compact numeric tokens.
    abstract = main_doc.find('Scientific response curves are often')
    replace_text = text(abstract)
    for a, b in [('On25previously', 'On 25 previously'), ('score0.', 'score 0.'),
                 ('versus0.', 'versus 0.'), ('reference and0.', 'reference and 0.'),
                 ('scores,0.', 'scores, 0.'), ('to0.', 'to 0.'), ('contains46native', 'contains 46 native'),
                 ('and781higher', 'and 781 higher')]:
        replace_text = replace_text.replace(a, b)
    main_doc.replace('Scientific response curves are often', replace_text)
    main_doc.after('Rheological structure suggests', [main_doc.paragraph(
        'Small-data viscosity prediction already separates model selection, performance estimation and physical interpolation behavior [59]. Here the input is a partial flow curve and the output is a joint law over the remaining rates. Representation selection also has supervised precedents, including principal covariates regression [60]. Quadratic loss approximations and Gaussian uncertainty have clear sensitivity-based precedents [62]; they should not be confused with physical curvature of a flow curve. These connections motivate testing the chosen representation rather than presenting low-dimensional compression or Gaussian approximation as new probability theory.')])
    main_doc.replace('The study makes three linked contributions.',
        'The study makes three linked contributions. It specifies partial hydrogel rheology as a probabilistic curve-continuation task with separate interpolation layouts. It adapts a small functional code while preserving a complete high-dimensional reference residual and a shared joint law. Finally, parameter-count-matched random, prior-KL and rotated-code controls, constitutive alternatives, uncertainty diagnostics and external transfer stress tests identify where that adaptation helps and fails. The original ALG-Ph/HA-Ph ink records [34, 35] support retrospective development; numerical verification and separately sourced curves serve different evidential roles.')
    main_doc.replace('The coordinate GP measures the value of correction.',
        'The coordinate GP measures the value of correction. The original two-by-two comparison crosses one fixed random QR representation and curvature with bounded and Gaussian corrections. An additional registered developmental analysis compares eight specified random QR codes, the top two latent-reference Karhunen–Loeve modes and one fixed45degree curvature rotation, keeping the original reference fits, held-formulation folds, three layouts, five coefficients, objective, bounds, penalty, zero start and optimizer. It adds250correction fits and750forecasts, all frozen before new target scoring, on the same consumed25P records (Section S20). There is no reference refit or new deployment fit. Equal parameter counts do not equalize distribution families or rotation-dependent mean and Cholesky constraints. Recording optimization geometry is relevant to GP interpretation [61]. The separate conditional Gaussian control modifies conditional mean and scale without imposing a common joint law across contexts.')
    p52 = main_doc.find('The coordinate GP measures')
    value = text(p52).replace('fixed45degree', 'fixed 45 degree').replace('adds250correction', 'adds 250 correction').replace('and750forecasts', 'and 750 forecasts').replace('consumed25P', 'consumed 25 P')
    main_doc.replace('The coordinate GP measures', value)
    main_doc.replace('Curvature lowers CRPS within both correction families:',
        'In the original comparison, curvature lowers bounded CRPS from0.052243to0.048253 and Gaussian CRPS from0.049908to0.043277 (Fig.3a). The Gaussian curvature model improves on that original random Gaussian for20/25formulations and on the same-code bounded model for22/25. Eight additional fixed random codes obtain macro CRPS0.049848–0.065362, all worse than the frozen curvature code, but the prior-KL code is stronger at0.037139 (Fig.S6; TableS13). The fixed45degree curvature rotation gives0.043156, demonstrating a small constraint-orientation effect within the same span. These outcomes support a useful low-capacity correction and the value of testing representations; they do not establish curvature as the best examined representation or select a new deployment model.')
    main_doc.after('In the original comparison, curvature lowers', [main_doc.paragraph(
        'The prior-KL control reduces pooled log RMSE to0.098648 and one-sided CRPS to0.073284, with lower overall formulation-averaged CRPS for19/25records. Its mean joint log density63.328is below the original curvature value63.658and conditional Gaussian82.926. Conditional Gaussian remains stronger for both interpolation layouts. Even the small mean middle-gap improvement over curvature is not widespread: KL wins only8/25formulations, loses17/25and covers11/33P0targets. No additional control has been externally evaluated, calibrated or adopted. These are retrospective comparisons, not an independent confirmation cohort.')])
    main_doc.replace('The inducing-feature Gaussian foundations are established',
        'The inducing-feature Gaussian foundations are established [19, 20, 21, 22, 23]. Our objective learns a shared population correction from masked conditional likelihood while preserving response variation outside the adapted code. Fixed curvature contrasts make one transparent rheological choice, but the completed random, prior-KL and within-span rotation comparisons narrow its interpretation: curvature beats all tested random codes, while prior-KL has lower macro CRPS and RMSE. Thus the contribution is the declared precursor-ink task, residual-preserving adaptation and controlled representation evidence, rather than an optimal or uniquely physical curvature representation. The newly stronger development control is retained without replacing the archived deployed model. FPCA, window or dimension selection would define further developmental candidates, not missing confirmations that can be inferred from the present comparisons.')
    main_doc.after('Point prediction, marginal distribution quality', [main_doc.paragraph(
        'Coherent propagation of an assumed uncertainty law is distinct from calibrated intervals, including in efficient shallow-ensemble approximations [63]. A posthoc covariance-only analysis of the new frozen laws likewise shows where a chosen code can act, not how accurately it predicts measurements (Section S21). Its higher continuation covariance allocation for prior-KL cannot be read as a percentage of physical information recovered.')])
    main_doc.after('The external stress test makes', [main_doc.paragraph(
        'Regression under changes of input domain can reorder learner performance [64]. In this study, continuation along shear rate and transfer between material sources are separate challenges. The reported external outcomes belong only to the original frozen curvature deployment model; the new prior-KL control has no external or new-batch evidence.')])
    main_doc.replace('The frozen runtime, deployment weights,',
        'The frozen runtime, deployment weights, saved-score replay and evidence are versioned at https://github.com/Hancy5521/hydrogel_control. The prior author-review baseline is commit668657bf93223268c60f04f3769b9385d2fb62f4. The additional representation protocol, sources, forecast hashes, score summaries and independent checks are archived separately from that baseline. The deployment weights still use all25P records and lawful S anchors, not held-formulation models; no prior-KL deployment weights are produced. Original scientific weights, runtime and historic scores are unchanged. Complete material fitting replay requires lawful processed assets, and no permanent archival DOI or new original-asset reuse license is assigned by this revision.')
    tidy_new_prose(main_doc)
    # SI retains original tables and clearly labels the first single-random result.
    si.replace('All four cells are now available.',
        'These four original cells are retained unchanged. Conditional Gaussian is a separate context-specific baseline. Values are native-target, three-layout development CRPS on the same25formulations. The original random code uses one fixed QR construction; the Gaussian comparison matches dimension, parameter count, bounds, regularization, optimizer and training access. Section S20 adds eight further fixed random codes and two deterministic representation diagnostics without rewriting this historical comparison.')
    si.replace('Within the bounded family, curvature lowers CRPS',
        'Within the original bounded family, curvature lowers CRPS by7.64%relative to its random directions; within the original Gaussian family the reduction is13.29%. The original Gaussian candidate improves on the matched random Gaussian for20/25formulations; its fixed-forecast bootstrap interval is[−0.009569,−0.003581]. These summaries remain conditional on that one original construction. The difference of directional differences,−0.002641, does not identify a causal representation-by-family interaction. Newer multi-representation outcomes, including the stronger prior-KL control, are reported separately in Section S20.')
    si.replace('The generic code uses fixed random QR directions',
        'The original generic code uses fixed random QR directions, and the local code uses the original curvature contrasts. Their saved directions, quadrature settings, parameters and protocols retain their original bindings. Gaussian corrections are independently optimized rather than moment matched from bounded tilts. The original random-Gaussian cell adds25outerfits under the same dimension, five coefficients, bounds, penalty, zero start, optimizer and information access (TableS3). SectionS20 adds a separate registered representation study with250additional outerfits and no reference or candidate refit.')
    labels = {'gp': 'Coordinate GP', 'generic': 'Generic bounded tilt', 'local': 'Curvature bounded tilt',
              'gaussian': 'Conditional Gaussian', 'joint_code_gaussian': 'Frozen curvature Gaussian',
              'random_gaussian': 'Original random Gaussian', 'prior_kl': 'Prior-KL Gaussian',
              'curvature45': 'Curvature Gaussian 45 degrees'}
    order = ['gp', 'generic', 'local', 'gaussian', 'joint_code_gaussian', 'random_gaussian']
    order += [f'random_{seed}' for seed in range(20261105, 20261113)] + ['prior_kl', 'curvature45']
    rows = []
    for method in order:
        a = scores['aggregate'][method]
        rows.append([labels.get(method, 'Random Gaussian ' + method.split('_')[-1]),
                     f"{a['CRPS']:.6f}",
                     *[f"{scores['by_mask'][mask][method]['CRPS']:.6f}" for mask in ('spread', 'middle_gap', 'one_sided')],
                     f"{100 * a['coverage90']:.2f}%"])
    si.append([
        si.paragraph('S20 Additional representation controls on consumed development formulations', 'heading'),
        si.paragraph(f"A new protocol was registered at actualUTC{protocol['actual_utc']}before execution. The fixed list contains eight random seeds20261105–20261112, the top two latent prior-KL modes and a fixed45degree rotation of the original curvature basis. All250new correction fits were admitted;750new conditional laws were frozen globally before their target scores. No original reference, candidate or deployment fit was repeated. The25formulations, three layouts, native target positions, score weights and response scale are unchanged. This is retrospective development, not a new material cohort."),
        si.paragraph('Random bases follow the original17by2standard-normal QR construction. Prior-KL uses the largest two eigenvectors of the latent17point support covariance, with the largest-magnitude component made positive, and converts them into prior-whitened coordinates. No responses select these directions. The rotation preserves the curvature span; it probes coordinate-dependent constraints. Every correction has the original five coefficients, one zero start, bounds, penalty0.02and optimizer. Equal counts do not make the bounded mean/Cholesky parameter spaces invariant under rotation. No failed fit, fallback, retry or best-seed selection occurs.'),
        si.paragraph('Table S13  Additional representation comparison with original controls', 'caption'),
        new_table(['Representation and correction', 'Macro CRPS', 'Spread CRPS', 'Middle gap CRPS', 'One sided CRPS', 'Coverage 90%'], rows, [3000, 1200, 1200, 1300, 1300, 1100]),
        si.paragraph('All rows retain the original25formulation development role. The first six rows are imported historical scores, not target rescoring. The last ten rows are the fixed new comparisons. Every entry averages all native targets and all stipulated formulations; no seed or formulation is selected for the table. Unrounded values, pooled logRMSE, joint logdensity and interval widths are retained in the score ledger.'),
        si.paragraph('The eight new random macro scores range from0.049848to0.065362, all above the original curvature0.043277. Prior-KL obtains0.037139macro CRPS,0.098648pooled logRMSE and0.073284one-sided CRPS, but its mean joint logdensity63.327985is below curvature63.658108and conditional Gaussian82.925879. Overall KL wins19/25formulations. In the middle gap, however, it wins only8/25and loses17/25; its mean difference is−0.00060448but median difference is+0.00083818. Arithmetic removal of P0 leaves a mean difference of only−0.00002107; all official results retain P0. KL covers11/33P0middle-gap targets versus8/33for curvature. This does not repair local reliability.'),
        si.paragraph('The45degree rotation obtains macro CRPS0.043156and one-sided CRPS0.090966, compared with original curvature0.043277and0.090792. It changes the constrained fit while preserving the span, so it is not evidence of new physical information. The prior-KL and rotation controls are not new deployment models, have no external results, and are not selected for adoption from this consumed development set.'),
        add_figure(si, root / 'analysis/plots/REPRESENTATION_CONTROLS.png'),
        si.paragraph('Figure S6  Additional fixed representation controls. All eight random constructions are shown, together with the prior-KL and rotated-curvature controls and retained original reference scores. The protocols use the same25development formulations. The random range is descriptive, not a confidence interval, and prior-KL is not an independently confirmed deployment replacement.', 'caption'),
        si.paragraph('S21 Posthoc allocation under the reference conditional covariance', 'heading'),
        si.paragraph('This explanation was registered only after seeing the completed representation CRPS, at actualUTC2026-11-04T16:02:18.046344+00:00. It reads150frozen covariance/index records and50existing fit receipts, without openingTRAIN, decoding response values or recomputing target scores. LetAbe the remaining standardized-code covariance given reference anchors,Bthe code–target cross covariance andV0the remaining reference target covariance. The code-explained block isB A⁻¹ Bᵀ. This is a reference conditional-Gaussian decomposition before learning, unlike the fitted-model summaries in SectionS17.'),
        si.paragraph('Table S14  Reference conditional covariance allocation after anchors', 'caption'),
        new_table(['Layout', 'KL trace fraction', 'Curvature 45 degree trace fraction', 'KL mean point fraction', 'Curvature 45 degree mean point fraction'], [
            [name, f"{diagnostic['summary']['prior_kl']['by_mask'][key]['explained_trace_fraction']['mean']:.6f}",
             f"{diagnostic['summary']['curvature45']['by_mask'][key]['explained_trace_fraction']['mean']:.6f}",
             f"{diagnostic['summary']['prior_kl']['by_mask'][key]['mean_marginal_explained_fraction']['mean']:.6f}",
             f"{diagnostic['summary']['curvature45']['by_mask'][key]['mean_marginal_explained_fraction']['mean']:.6f}"]
            for name, key in [('Spread', 'spread'), ('Middle gap', 'middle_gap'), ('One sided', 'one_sided')]],
            [2100, 1750, 1950, 1750, 1950]),
        si.paragraph('Trace fractions divide the trace of the code-explained block bytrace(V0); mean point fractions average the33diagonal ratios. They weight positions differently. For example, prior-KL one-sided allocation0.988047does not mean98.80%accuracy, physical information recovered or physical variance explained. All150conditional code covariances have rank2; no eigenmode is discarded, no jitter is added, and every residual is positive definite. The analysis states explicit machine-precision rank/support tolerances and retains all results. It cannot prove prediction quality, calibration, physical identifiability or new-batch transfer.'),
        si.paragraph('All25prior-KL fits and all25rotated-curvature fits hit the lower bound of the second logCholesky diagonal. The rotated version also hits the first mean lower bound in16folds and the second inonefold. This describes constrained endpoints; no bounds are relaxed in response. Independent checks reconstruct all150covariance rows and50parameter receipts without importing the producer. These are numerical verification and posthoc explanation, not additional material experiments.'),
    ])
    tidy_new_prose(si)
    for node in ref_nodes:
        main_doc.body.remove(node)
    # Preserve the bibliography style without retaining an old bibliography in
    # the first-appearance scan: add reference nodes directly from original XML.
    ref_style_template = deepcopy(ref_nodes[0])
    original_paragraph = main_doc.paragraph
    def paragraph_with_reference(value, kind='body'):
        if kind != 'reference':
            return original_paragraph(value, kind)
        n = deepcopy(ref_style_template)
        for child in list(n):
            if child.tag != f'{{{W}}}pPr': n.remove(child)
        rr = E.SubElement(n, f'{{{W}}}r')
        E.SubElement(rr, f'{{{W}}}t').text = value
        return n
    main_doc.paragraph = paragraph_with_reference
    mapping = renumber(main_doc, si, references)
    # Syntactic global numeric spacing can touch URL/DOI reference templates.
    # Bibliography entries are built only after it and verified below verbatim.
    cover.replace('The methodological contribution is',
        'The methodological contribution is a task-specific, low-capacity adaptation of an established Gaussian construction. A shared five-parameter correction modifies a two-dimensional functional code while preserving the complete reference residual. Parameter-count-matched representation controls and constitutive alternatives test its usefulness for small-data continuation. Additional preregistered development controls favour a prior-KL representation over the original curvature code; the manuscript reports this limit rather than claiming curvature optimality.')
    cover.replace('On 25 previously inspected formulations,',
        'On25previously inspected formulations, the original curvature correction improves retrospective macro CRPS over reference and conditional Gaussian controls, principally in one-sided continuation. A no-retraining test of that frozen model on46curves from four public sources retains beneficial and adverse transfers. External rankings depend on material-family weighting, and some probability intervals severely undercover. The paper claims neither universal transfer, calibrated experimental replacement nor measured laboratory savings; the new developmental control is not silently substituted into this external evidence.')
    tidy_new_prose(cover)
    # Checks must pass before writing any document.
    abstract_text = text(main_doc.find('Scientific response curves are often'))
    if len(abstract_text.split()) >= 300:
        raise ValueError('Abstract must remain below300words')
    new_bib = {int(m.group(1)): m.group(2)
               for p in main_doc.body if (m := re.match(r'^\[(\d+)\] (.+)$', text(p)))}
    if len(new_bib) != 64 or any(new_bib[mapping[i]] != old_refs[i] for i in old_refs):
        raise PermissionError('Old reference entry text changed')
    forbidden = ['only one random construction is evaluated',
                 'FPCA, multiple random codes and window sensitivity are worthwhile',
                 'AWAITING_ACTUAL_RESULTS', 'CITE_', 'ACTUAL_PROTOCOL_UTC']
    all_prose = text(main_doc.body) + text(si.body)
    for phrase in forbidden:
        if phrase in all_prose:
            raise ValueError('Outdated/provisional text remains: ' + phrase)
    visible_count = len(text(main_doc.body).split())
    if visible_count > 8500:
        raise ValueError('Main visible XML token count exceeds8500recommendation')
    output.mkdir(parents=True, exist_ok=False)
    documents = []
    for package in (main_doc, si, cover):
        target = output / package.path.name
        package.save(target)
        documents.append({'file': target.name, 'sha256': sha(target),
            'source_sha256': PINS[target.name],
            'old_math_objects_preserved': sum(package.original_math.values()),
            'old_tables_preserved': sum(package.original_tables.values()),
            'new_total_tables': sum(table_hashes(package.tree).values()),
            'old_media_preserved': len(package.original_media),
            'styles_and_section_preserved': True,
            'visible_XML_whitespace_tokens': len(text(package.body).split())})
    receipt = {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'source_builder_sha256': sha(__file__),
        'operation_marker_receipt_sha256': sha(args.operation_receipt),
        'input_scientific_audit_sha256': sha(root / 'audit/FROZEN_AUDIT.json'),
        'input_covariance_audit_sha256': sha(root / 'audit/COVARIANCE_AUDIT.json'),
        'score_result_sha256': sha(score_path), 'diagnostic_result_sha256': sha(diagnostic_path),
        'six_reference_ledger_sha256': sha(root / 'literature/SIX_REFERENCE_INSERT_CHECK.json'),
        'new_figure_sha256': sha(root / 'analysis/plots/REPRESENTATION_CONTROLS.png'),
        'reference_count': 64, 'reference_number_map': mapping,
        'all58original_reference_entry_texts_exact_except_number': True,
        'abstract_words_whitespace': len(abstract_text.split()),
        'new_materials': 0, 'manuscript_builder_new_fits': 0,
        'frozen_model_and_original_result_bytes_changed': False,
        'new_comparisons_classification': 'retrospective consumed25P; no new deployment or externalKL assessment',
        'documents': documents,
        'render_status': 'NOT_YET_RENDERED; root must render and inspect every page before delivery',
        'word_count_scope': 'XML whitespace tokens include references/captions/tables; mathematical nodes omitted; not an official journal word-count determination'}
    write_json(output / 'BUILD_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--operation-receipt', type=Path, required=True)
    main(parser.parse_args())
