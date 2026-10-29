"""Strict baseline comparison; only documented elapsed-time fields are ignored."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('baseline')
ap.add_argument('variant')
ap.add_argument('--out', required=True)
args = ap.parse_args()
base, variant = [json.loads(Path(p).read_text()) for p in (args.baseline, args.variant)]
errors = []
expected = [f'real_{i:02d}' for i in range(1, 7)] + ['invalid_format', 'too_long', 'wrong_transcript', 'unknown_script', 'fallback_align', 'fallback_align_one', 'resident_recycle_01', 'resident_recycle_02', 'resident_recycle_03', 'resident_recovery', 'resident_forced_fallback', 'resident_after_fallback']

def normalize(case):
    c = copy.deepcopy(case)
    c.pop('seconds', None)
    c.pop('expectation_failed', None)
    r = c.get('result', {})
    r.pop('processingMs', None)
    r.get('detail', {}).pop('stageMs', None)
    return c

def differences(a, b, path=''):
    if type(a) is not type(b): return [path + ': type mismatch']
    if isinstance(a, dict):
        out = [path + ': key mismatch'] if a.keys() != b.keys() else []
        for k in a.keys() & b.keys(): out.extend(differences(a[k], b[k], path + '/' + k))
        return out
    if isinstance(a, list):
        out = [path + ': length mismatch'] if len(a) != len(b) else []
        for i, (x, y) in enumerate(zip(a, b)): out.extend(differences(x, y, path + '/' + str(i)))
        return out
    return [] if a == b else [path + ': value mismatch']

for name, data in [('baseline', base), ('variant', variant)]:
    if not data.get('complete') or not data.get('pass'): errors.append(name + ': incomplete/failed')
    if [c['id'] for c in data['cases']] != expected: errors.append(name + ': wrong/missing cases')
    if any(c.get('expectation_failed') for c in data['cases']): errors.append(name + ': case failed')
    for c in data['cases']:
        if c['expect'] == 'OK' and c.get('result', {}).get('status') != 'OK': errors.append(name + ':' + c['id'])
        if c['expect'] == 'KeyError' and c.get('exception', {}).get('type') != 'KeyError': errors.append(name + ':' + c['id'])
    if not data.get('checks', {}).get('api_import'): errors.append(name + ': API import failed')
    for cli in ('mfa_cli', 'ffmpeg'):
        if data.get('checks', {}).get(cli, {}).get('returncode') != 0: errors.append(name + ':' + cli)

errors.extend(differences(base['source_hashes'], variant['source_hashes'], '/source_hashes'))
errors.extend(differences(base['input_hashes'], variant['input_hashes'], '/input_hashes'))
for name,data in [('baseline',base),('variant',variant)]:
    for check in ['default_resident','resident_warmup','six_real_requests_reused_resident','resident_recycle','resident_recovery','resident_fallback_and_restart','resident_shutdown']:
        if data.get('checks',{}).get(check) is not True: errors.append(name+':'+check)
for x,y in zip(base['resident_trace'],variant['resident_trace']):
    if x['fallback_calls']!=y['fallback_calls']:errors.append('/fallback_calls/'+x['id'])
for k in ('torch', 'transformers', 'machine', 'cpu', 'code_hash', 'no_stt'):
    if base['environment'][k] != variant['environment'][k]: errors.append('/environment/' + k)
for a, b in zip(base['cases'], variant['cases']):
    errors.extend(differences(normalize(a), normalize(b), '/cases/' + a['id']))
report = {'pass': not errors, 'case_count': len(variant['cases']), 'differences': errors,
          'ignored_result_paths': ['result.processingMs', 'result.detail.stageMs'],
          'ignored_harness_fields': ['seconds', 'expectation_failed (validated separately)'],
          'baseline_sha256': hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
          'variant_sha256': hashlib.sha256(Path(args.variant).read_bytes()).hexdigest(),
          'pip_available': variant['environment']['pip_available']}
Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if report['pass'] else 1)
