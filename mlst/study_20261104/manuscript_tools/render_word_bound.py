"""Use bundled render_docx rasterizer with a hash-bound read-only Word PDF export.
Windows has no soffice on PATH. No managed package is modified.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os
from datetime import datetime,timezone
ap=argparse.ArgumentParser();ap.add_argument('--qa',type=Path,required=True);ap.add_argument('--renderer',type=Path,required=True);ap.add_argument('--poppler',type=Path,required=True);a=ap.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
os.environ['PATH']=str(a.poppler)+os.pathsep+os.environ.get('PATH','')
spec=importlib.util.spec_from_file_location('bundled_render_docx',a.renderer);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
records=json.loads((a.qa/'WORD_EXPORT_RECEIPT.json').read_text('utf-8-sig'));out=[]
for r in records:
 doc=a.qa/(r['stem']+'.docx');pdf=a.qa/(r['stem']+'.pdf');dest=a.qa/r['stem']
 if dest.exists():raise FileExistsError(dest)
 assert sha(doc)==r['input_sha256'] and sha(pdf)==r['pdf_sha256']
 def bound_export(doc_path,*args,**kwargs):
  assert sha(Path(doc_path))==r['input_sha256']
  return str(pdf),'Read-only owned Word COM export; originalDOCX not resaved; strictSHA binding.'
 mod.convert_to_pdf=bound_export
 pages=mod.rasterize(str(doc),str(dest),120,False,False)
 assert len(pages)==r['pages']
 out.append({'stem':r['stem'],'pages':len(pages),'docx_sha256':sha(doc),'pdf_sha256':sha(pdf),'pngs':[{'file':str(Path(p).relative_to(a.qa)),'sha256':sha(Path(p))}for p in pages]})
result={'actual_utc':datetime.now(timezone.utc).isoformat(),'renderer':str(a.renderer),'renderer_sha256':sha(a.renderer),'conversion_engine':'uniquely-owned new Microsoft Word instance, read-only export','rendering_engine':'bundled render_docx.py rasterize + bundledPoppler','reason':'No soffice.exe found on Windows PATH; managedhelper convert_to_pdf overridden in memory with pre-exported PDF afterSHAchecks','source_sha256':sha(Path(__file__)),'documents':out}
(a.qa/'RENDER_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print(json.dumps({r['stem']:r['pages']for r in out},indent=2))
