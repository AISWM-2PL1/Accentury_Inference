"""Delete only candidates from the disposable Docker build stage, never a host tree."""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

assert os.environ.get('ACCENTURY_SLIM_TEST') == '1', 'disposable image guard missing'
assert sys.platform == 'linux', 'Linux build stage only'
assert Path('/tmp/accentury-slim-build-guard').read_text().strip() == 'isolated-kan296-slimming-2026-10-29'
assert os.geteuid() == 0
assert os.environ.get('ACCENTURY_CODE_HASH') == '0d72c03', 'unexpected baseline'
assert Path('/app/src/scoring/serve.py').is_file()

roots = [Path('/opt/conda'), Path('/usr/local/lib/python3.12/site-packages'),
         Path('/usr/local/include'), Path('/usr/share/doc'), Path('/usr/share/man')]
cache = Path('/opt/conda/pkgs')
cache_targets = set()
for path in Path('/opt/conda/envs/mfa').rglob('*'):
    if path.is_symlink():
        resolved = path.resolve()
        if resolved.is_relative_to(cache):
            cache_targets.add(str(resolved))

def legal(path):
    return any(re.match(r'(?i)^(licen[cs]e|licen[cs]es|copying|copyright|notice)([._-].*)?$', p)
               for p in path.parts)

def classify(path):
    s = str(path)
    if legal(path): return None
    if path.is_relative_to(cache):
        if any(s == target or s.startswith(target + '/') for target in cache_targets): return None
        return 'conda_cache'
    if '__pycache__' in path.parts and path.suffix in ('.pyc', '.pyo'):
        if (path.parent.parent / (path.name.split('.')[0] + '.py')).exists():
            # Keep MFA import caches: resident startup, recycling and fallback all need them.
            if path.is_relative_to(Path('/opt/conda/envs/mfa')): return None
            return 'pycache'
    if s.startswith(('/opt/conda/envs/mfa/include/', '/usr/local/include/',
                     '/usr/local/lib/python3.12/site-packages/torch/include/')): return 'headers'
    if any(p in ('tests', 'test') for p in path.parts) and ('site-packages/' in s or '/python3.11/test/' in s):
        return 'test_files'
    if '/share/doc/' in s or '/share/man/' in s: return 'docs'
    if path.suffix == '.a': return 'static_archives'
    return None

removed = []
for root in roots:
    for directory, subdirs, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode): continue
            category = classify(path)
            if category:
                removed.append({'path': str(path), 'category': category, 'bytes': st.st_size,
                                'inode': st.st_ino, 'device': st.st_dev, 'links': st.st_nlink})
                path.unlink()

# Package managers are removed only after every build-time installation has completed.
# setuptools/pkg_resources and all runtime distribution metadata are retained.
for base in [Path('/usr/local/lib/python3.12/site-packages'),
             Path('/opt/conda/envs/mfa/lib/python3.11/site-packages')]:
    for target in base.iterdir():
        if target.name != 'pip' and not (target.name.startswith('pip-') and target.name.endswith('.dist-info')):
            continue
        for directory, subdirs, files in os.walk(target, topdown=False):
            for name in files:
                p = Path(directory) / name
                st = p.lstat()
                removed.append({'path': str(p), 'category': 'pip_manager', 'bytes': st.st_size,
                                'inode': st.st_ino, 'device': st.st_dev, 'links': st.st_nlink})
                p.unlink()
            for name in subdirs:
                p = Path(directory) / name
                if p.is_symlink(): p.unlink()
                else: p.rmdir()
        target.rmdir()
for base in [Path('/usr/local/bin'), Path('/opt/conda/envs/mfa/bin')]:
    for p in base.iterdir():
        if re.fullmatch(r'pip(3(\.(11|12))?)?', p.name) or p.name == 'micromamba':
            st = p.lstat()
            removed.append({'path': str(p), 'category': 'installer_command', 'bytes': st.st_size,
                            'inode': st.st_ino, 'device': st.st_dev, 'links': st.st_nlink})
            p.unlink()

bins = Path('/usr/local/lib/python3.12/site-packages/torch/bin')
for p in bins.iterdir():
    if p.is_file() and (p.name.startswith('test_') or p.name.endswith('Test')):
        st = p.lstat()
        removed.append({'path': str(p), 'category': 'torch_test_binaries', 'bytes': st.st_size,
                        'inode': st.st_ino, 'device': st.st_dev, 'links': st.st_nlink})
        p.unlink()

# Remove only dangling links whose targets were removed by this manifest.
removed_targets = {x['path'] for x in removed}
unlinked = []
while True:
    changed = False
    for root in roots:
        for p in root.rglob('*'):
            if p.is_symlink() and not p.exists() and str(p.resolve()) in removed_targets:
                unlinked.append({'path': str(p), 'target': os.readlink(p)})
                removed_targets.add(str(p))
                p.unlink()
                changed = True
    if not changed: break
remaining_broken = [str(p) for root in roots for p in root.rglob('*') if p.is_symlink() and not p.exists()]
assert not any(p.startswith('/opt/conda/envs/mfa/') for p in remaining_broken), remaining_broken
out = Path('/opt/slim-audit')
out.mkdir(exist_ok=True)
payload = json.dumps(removed, ensure_ascii=False, separators=(',', ':')).encode()
with gzip.open(out / 'removed.json.gz', 'wb') as f: f.write(payload)
counts = collections.Counter(x['category'] for x in removed)
unique = {}
for x in removed: unique[(x['device'], x['inode'])] = x['bytes']
summary = {'removed_paths': len(removed), 'categories': dict(counts),
           'removed_manifest_sha256': hashlib.sha256(payload).hexdigest(),
           'removed_unique_inode_bytes_not_image_savings': sum(unique.values()),
           'cache_symlink_targets_preserved': sorted(cache_targets),
           'mfa_runtime_pycache_preserved': True, 'dangling_links_removed': unlinked, 'remaining_broken_links': remaining_broken,
           'setuptools_retained': True, 'runtime_distribution_metadata_retained': True}
(out / 'cleanup_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k:(len(v) if k in ('dangling_links_removed','remaining_broken_links') else v)
                  for k,v in summary.items()}), flush=True)
