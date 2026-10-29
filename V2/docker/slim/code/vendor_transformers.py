"""Whisper-only vendored copy, with upstream bytes and installed metadata retained."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

assert os.environ.get('ACCENTURY_SLIM_TEST') == '1'
assert sys.platform == 'linux' and os.geteuid() == 0
assert Path('/tmp/accentury-slim-build-guard').read_text().strip() == 'isolated-kan296-slimming-2026-10-29'
assert os.environ.get('ACCENTURY_CODE_HASH') == '0d72c03'
assert importlib.metadata.version('transformers') == '4.57.6'
source = Path('/usr/local/lib/python3.12/site-packages/transformers')
target = Path('/app/vendor/transformers')
assert source.is_dir() and not target.exists()
families = {'auto', 'encoder_decoder', 'whisper'}
kept, excluded = [], []
for p in sorted(source.rglob('*')):
    if not p.is_file(): continue
    rel = p.relative_to(source)
    # Keep every common module/resource. Only unrelated model-family directories go.
    keep = not (len(rel.parts) > 2 and rel.parts[0] == 'models' and rel.parts[1] not in families)
    entry = {'path': str(rel), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    if keep:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == entry['sha256']
        kept.append(entry)
    else:
        excluded.append(entry)
# All original dist-info files, including the Apache license, stay in site-packages.
metadata = list(source.parent.glob('transformers-4.57.6.dist-info'))
assert len(metadata) == 1
licenses = [str(p) for p in metadata[0].rglob('*') if p.is_file() and 'license' in p.name.lower()]
assert licenses, 'upstream license missing'
shutil.rmtree(source)
# Register only in the main Python 3.12 interpreter; do not alter MFA Python 3.11's path.
pth=source.parent/'accentury_vendor.pth'
assert not pth.exists()
pth.write_text('/app/vendor\n')
main_check=subprocess.run([sys.executable,'-I','-B','-c',
    'import transformers; from transformers import WhisperForConditionalGeneration, WhisperProcessor; import transformers.models.auto; assert transformers.__file__.startswith("/app/vendor/transformers/"); print(transformers.__version__)'],
    capture_output=True,text=True,check=True)
mfa_check=subprocess.run(['/opt/conda/envs/mfa/bin/python','-I','-B','-c',
    'import sys; assert "/app/vendor" not in sys.path; print("MFA path unchanged")'],
    capture_output=True,text=True,check=True)
report = {'version': '4.57.6', 'kept_model_families': sorted(families), 'kept_files': kept,
          'excluded_files': excluded, 'kept_bytes': sum(x['bytes'] for x in kept),
          'excluded_bytes': sum(x['bytes'] for x in excluded), 'upstream_licenses': licenses,
          'scope': 'Only current Whisper scorer; other Transformers model families are intentionally unsupported.',
          'path_registration':str(pth),'main_import_check':main_check.stdout.strip(),'mfa_path_check':mfa_check.stdout.strip()}
(Path('/opt/slim-audit') / 'vendor_manifest.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['kept_files','excluded_files']}), flush=True)
