"""Compare the actual optimized resident-MFA image with its slim copy offline."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,'/app/src')
sys.path.insert(0,'/app')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out)
    report={'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'checks':{},'cases':[],'resident_trace':[]}
    serve=None
    def save():out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    try:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        report['environment']={'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
            'machine':os.uname().machine,'cpu':torch.backends.cpu.get_cpu_capability(),
            'code_hash':os.environ.get('ACCENTURY_CODE_HASH'),'no_stt':False,
            'pip_available':importlib.util.find_spec('pip') is not None}
        assert report['environment']['code_hash']=='0d72c03'
        import app.main
        report['checks']['api_import']=True
        for cmd,name in [(['/opt/conda/envs/mfa/bin/mfa','--help'],'mfa_cli'),(['/opt/conda/envs/mfa/bin/ffmpeg','-version'],'ffmpeg')]:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            report['checks'][name]={'returncode':p.returncode,'stderr_tail':p.stderr[-1000:]}
            assert p.returncode==0,name
        from scoring import serve
        assert serve._mfa_mode()=='resident'
        report['checks']['default_resident']=True
        started=time.monotonic();scorer=serve.Track1Scorer(whisper_device='cpu')
        report['load_seconds']=time.monotonic()-started
        resident=serve._RESIDENT
        assert resident.proc is not None and resident.proc.poll() is None,'resident warmup failed'
        report['checks']['resident_warmup']=True
        print(json.dumps({'loaded_seconds':report['load_seconds'],'resident_pid':resident.proc.pid}),flush=True)
        fallback={'one':0,'corpus':0}
        for attr,key in [('_align_one','one'),('_align_corpus','corpus')]:
            original=getattr(serve,attr)
            def wrapper(*args,_original=original,_key=key,**kwargs):
                fallback[_key]+=1
                return _original(*args,**kwargs)
            setattr(serve,attr,wrapper)
        def state():return {'pid':resident.proc.pid if resident.proc else None,'jobs':resident.jobs,'pending_recycle':resident._next is not None}
        def execute(case,override=None,expected_fallback=None):
            started=time.monotonic();before=state();counts=dict(fallback)
            record={k:case[k] for k in ('id','kind','expect')}
            try:
                value=override if override is not None else case.get('transcript')
                record['result']=scorer.score('/inputs/'+case['wav'],case['script_key'],transcript=value,with_feedback=True)
                status=record['result'].get('status')
                if case['expect']=='OK' and status!='OK':record['expectation_failed']=True
                if case['expect']=='not_OK' and status=='OK':record['expectation_failed']=True
                if case['expect']=='KeyError':record['expectation_failed']=True
            except Exception as e:
                record['exception']={'type':type(e).__name__,'message':str(e)}
                if case['expect']!=type(e).__name__:record['expectation_failed']=True
            after=state();delta={k:fallback[k]-counts[k] for k in fallback}
            if expected_fallback is not None and delta!=expected_fallback:record['expectation_failed']=True
            if case['kind']=='real' and (after['pid']!=before['pid'] or after['jobs']!=before['jobs']+1):record['expectation_failed']=True
            record['seconds']=time.monotonic()-started
            report['resident_trace'].append({'id':case['id'],'before':before,'after':after,'fallback_calls':delta})
            report['cases'].append(record);save()
            print(json.dumps({'case':case['id'],'seconds':round(record['seconds'],2),'status':record.get('result',{}).get('status'),
                              'failed':record.get('expectation_failed',False),'fallback_calls':delta}),flush=True)
            return record
        cases=json.loads(Path('/inputs/manifest.json').read_text())
        for case in cases:execute(case,expected_fallback={'one':0,'corpus':0})
        report['checks']['six_real_requests_reused_resident']=not any(c.get('expectation_failed') for c in report['cases'][:6])
        first=cases[0]
        def reference_case(name,mode='resident',expected=None):
            os.environ['ACCENTURY_MFA_MODE']=mode
            case=dict(first,id=name,kind='control',expect='OK')
            return execute(case,first['reference_transcript'],expected or {'one':0,'corpus':0})
        reference_case('fallback_align','align',{'one':0,'corpus':1})
        reference_case('fallback_align_one','align_one',{'one':1,'corpus':0})
        os.environ['ACCENTURY_MFA_MODE']='resident'
        resident.shutdown();resident.recycle=2;resident.start();first_pid=resident.proc.pid
        reference_case('resident_recycle_01')
        reference_case('resident_recycle_02')
        pending=resident._next is not None
        reference_case('resident_recycle_03')
        assert pending and resident.proc and resident.proc.pid!=first_pid,'resident recycling did not occur'
        report['checks']['resident_recycle']=True
        resident.recycle=200
        killed=resident.proc.pid;resident.proc.kill();resident.proc.wait()
        reference_case('resident_recovery')
        assert resident.proc and resident.proc.pid!=killed,'dead resident not restarted'
        report['checks']['resident_recovery']=True
        real_ask=resident._ask
        def fail_alignment(msg,timeout,proc=None):
            if msg.get('op')=='align':raise TimeoutError('intentional offline-test fault')
            return real_ask(msg,timeout,proc)
        resident._ask=fail_alignment
        try:reference_case('resident_forced_fallback',expected={'one':1,'corpus':0})
        finally:resident._ask=real_ask
        reference_case('resident_after_fallback')
        assert resident.proc and resident.proc.poll() is None
        report['checks']['resident_fallback_and_restart']=True
        resident.shutdown()
        report['checks']['resident_shutdown']=resident.proc is None and resident._next is None
        report['transformers_modules']=sorted({str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
                                               if n.startswith('transformers') and getattr(m,'__file__',None)})
        report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/app/src').rglob('*.py')}
        report['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/inputs').iterdir() if p.is_file()}
        report['complete']=True
        report['pass']=len(report['cases'])==18 and not any(c.get('expectation_failed') for c in report['cases'])
        save();return 0 if report['pass'] else 2
    except Exception:
        report['fatal']=traceback.format_exc();report['complete']=False;report['pass']=False;save();traceback.print_exc();return 1
    finally:
        if serve is not None:serve._RESIDENT.shutdown()

if __name__=='__main__':raise SystemExit(main())
