"""Hash unchanged service code, data, and pretrained assets in a fresh container."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
args = ap.parse_args()
files = set()
for root in ('/data','/opt/hf','/opt/mfa'):
    files.update(p for p in Path(root).rglob('*') if p.is_file())
files.update(p for p in Path('/app').rglob('*') if p.is_file()
             and not str(p).startswith('/app/vendor/') and '__pycache__' not in p.parts)
hashes = {}
memo = {}
for p in sorted(files):
    st = p.stat()
    key = (st.st_dev,st.st_ino)
    if key not in memo:
        h = hashlib.sha256()
        with p.open('rb') as f:
            while chunk := f.read(4*1024*1024): h.update(chunk)
        memo[key] = h.hexdigest()
    hashes[str(p)] = {'bytes': st.st_size,'sha256': memo[key]}
mfa_pip = subprocess.run(['/opt/conda/envs/mfa/bin/python','-B','-c',
    'import importlib.util; print(importlib.util.find_spec("pip") is not None)'],capture_output=True,text=True,check=True)
report = {'assets': hashes, 'pip_main': importlib.util.find_spec('pip') is not None,
          'pip_mfa': mfa_pip.stdout.strip() == 'True',
          'manager_commands': {s:Path(s).exists() for s in ('/usr/local/bin/pip','/usr/local/bin/micromamba','/opt/conda/envs/mfa/bin/pip')},
          'broken_mfa_symlinks': [str(p) for p in Path('/opt/conda/envs/mfa').rglob('*') if p.is_symlink() and not p.exists()]}
report['all_deletion_roots_broken_symlinks'] = [str(p)
    for root in ('/opt/conda','/usr/local/lib/python3.12/site-packages','/usr/local/include','/usr/share/doc','/usr/share/man')
    for p in Path(root).rglob('*') if p.is_symlink() and not p.exists()]
report['mfa_runtime_pycache_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path('/opt/conda/envs/mfa/lib/python3.11').rglob('*.pyc')
    if '__pycache__' in p.parts and not any(x in ('test','tests','pip') for x in p.parts)
    and (p.parent.parent/(p.name.split('.')[0]+'.py')).is_file()}
report['uvicorn_version']=importlib.metadata.version('uvicorn')
report['mfa_sys_path']=json.loads(subprocess.check_output(['/opt/conda/envs/mfa/bin/python','-I','-B','-c',
    'import json,sys;print(json.dumps(sys.path))'],text=True))
report['vendor_in_mfa_path']='/app/vendor' in report['mfa_sys_path']
for name in ('cleanup_summary.json','vendor_manifest.json','runtime_cache_correction.json'):
    p = Path('/opt/slim-audit',name)
    if p.exists(): report[name] = json.loads(p.read_text())
Path(args.out).write_text(json.dumps(report,indent=2))
print(json.dumps({'assets_hashed':len(hashes),'pip_main':report['pip_main'],'pip_mfa':report['pip_mfa'],'broken_mfa_symlinks':len(report['broken_mfa_symlinks'])}))
