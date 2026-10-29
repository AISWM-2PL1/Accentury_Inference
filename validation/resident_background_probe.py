"""Confirm that recycling consumes the actual asynchronously prepared process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,'/app/src')
from scoring import serve
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
case=json.loads(Path('/inputs/manifest.json').read_text())[0]
report={'scores':[],'checks':{}}
resident=serve._RESIDENT
try:
    assert serve._mfa_mode()=='resident'
    resident.recycle=2
    scorer=serve.Track1Scorer(stt=False)
    assert resident.proc and resident.proc.poll() is None
    original_pid=resident.proc.pid
    for i in range(2):
        result=scorer.score('/inputs/'+case['wav'],case['script_key'],transcript=case['reference_transcript'])
        assert result['status']=='OK';report['scores'].append(result)
    assert resident._next is not None,'no background preparation thread'
    thread,box=resident._next
    thread.join(timeout=90)
    assert not thread.is_alive() and 'error' not in box,repr(box.get('error'))
    prepared=box.get('proc')
    assert prepared is not None and prepared.poll() is None,'background preparation failed'
    report['prepared_pid']=prepared.pid
    result=scorer.score('/inputs/'+case['wav'],case['script_key'],transcript=case['reference_transcript'])
    assert result['status']=='OK';report['scores'].append(result)
    assert resident.proc is prepared and resident.proc.pid!=original_pid
    assert resident.jobs==1 and resident._next is None
    report['checks']={'prepared_process_consumed':True,'new_pid':True,'no_sync_respawn':True}
    report['pass']=True
finally:
    resident.shutdown()
    report['shutdown_complete']=resident.proc is None and resident._next is None
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='scores'}))
