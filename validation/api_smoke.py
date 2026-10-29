"""Actual Uvicorn + real worker HTTP test, entirely inside an offline container."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out)
env = dict(os.environ, ACCENTURY_AI_INTERNAL_TOKEN='isolated-test-token',
           ACCENTURY_AI_INTERNAL_TOKEN_REQUIRED='true', ACCENTURY_AI_TRACK1_DEVICE='cpu',
           ACCENTURY_AI_ANALYSIS_TIMEOUT_SECONDS='300', ACCENTURY_AI_TEMP_DIR='/tmp/api-slim-test')
report = {'timeout_override_for_emulation_seconds': 300, 'checks': {}}
log = out.with_suffix('.server.log').open('w')
p = subprocess.Popen(['uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000', '--workers', '1'],
                     env=env, stdout=log, stderr=subprocess.STDOUT, cwd='/app')

def request(path, data=None, headers=None):
    req = urllib.request.Request('http://127.0.0.1:8000' + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=330) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

try:
    deadline = time.monotonic() + 600
    health = None
    while time.monotonic() < deadline:
        if p.poll() is not None: raise RuntimeError('uvicorn exited before readiness')
        try:
            health = request('/internal/v0/health')
            if health == (200, {'status': 'UP'}): break
        except (ConnectionError, urllib.error.URLError): pass
        time.sleep(0.5)
    assert health == (200, {'status': 'UP'}), health
    report['checks']['health'] = health
    unauthorized = request('/internal/v0/metrics')
    assert unauthorized[0] == 401, unauthorized
    report['checks']['unauthorized'] = unauthorized
    case = json.loads(Path('/inputs/manifest.json').read_text())[0]
    meta = json.dumps({'scriptKey': case['script_key'], 'itemId': 'slim-test', 'scoreVersion': 'slim-test', 'correlationId': 'slim-test'})
    boundary = 'accentury-slim-boundary-20261029'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="meta"\r\n\r\n{meta}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="test.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += Path('/inputs', case['wav']).read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    status, result = request('/internal/v0/analyze', body, {'Content-Type': f'multipart/form-data; boundary={boundary}',
                                                       'X-Accentury-Internal-Token': 'isolated-test-token'})
    report['checks']['analyze'] = {'http_status': status, 'result': result}
    assert status == 200 and result.get('status') == 'OK', (status, result)
    report['checks']['metrics'] = request('/internal/v0/metrics', headers={'X-Accentury-Internal-Token': 'isolated-test-token'})
    report['functional_pass'] = True
except Exception:
    report['pass'] = False
    report['fatal'] = traceback.format_exc()
finally:
    p.terminate()
    try: p.wait(timeout=30)
    except subprocess.TimeoutExpired: p.kill(); p.wait()
    log.close()
    report['server_exit_code'] = p.returncode
    # Uvicorn 0.54 re-raises a captured SIGTERM after completing graceful shutdown.
    report['graceful_shutdown'] = (p.returncode in (0, -15)
        and 'Application shutdown complete.' in out.with_suffix('.server.log').read_text())
    report['pass'] = bool(report.get('functional_pass')) and report['graceful_shutdown']
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({'pass': report['pass'], 'server_exit_code': p.returncode, 'fatal': report.get('fatal')}))
raise SystemExit(0 if report['pass'] else 1)
