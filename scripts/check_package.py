"""Validate source syntax, local documentation links and publication hygiene."""
from pathlib import Path
import ast,json,re,hashlib
r=Path(__file__).resolve().parents[1];errors=[];count=0
for p in r.rglob('*'):
 if not p.is_file() or any(x in p.parts for x in ('.git','vendor','__pycache__','runs','data')):continue
 if p.suffix=='.py':
  count+=1
  try:ast.parse(p.read_text(),filename=str(p))
  except SyntaxError as e:errors.append(str(e))
 if p.suffix in ('.py','.md','.yaml','.json','.csv','.txt','.html','.sh'):
  text=p.read_text()
  if re.search(r'/data0[12]/|192\.168\.4\.51|ghp_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}',text):errors.append('private path/credential pattern: '+str(p.relative_to(r)))
 if p.suffix=='.md':
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in target or target.startswith('#'):continue
   if not (p.parent/target.split('#')[0]).exists():errors.append('broken link '+str(p.relative_to(r))+': '+target)
result={'python_files':count,'errors':errors,'pass':not errors}
print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
