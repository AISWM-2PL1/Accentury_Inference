"""Exercise the optimized warmup functions with MFA's own interpreter offline."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ['PATH']=os.environ['ACCENTURY_MFA_BIN']+':'+os.environ.get('PATH','')
sys.path.insert(0,'/app/src/scoring')
import mfa_one
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
log=io.StringIO();started=time.monotonic()
with contextlib.redirect_stderr(log):
    reply=mfa_one._handle({'op':'ping','dic':os.environ['ACCENTURY_MFA_DICT'],'acoustic':'korean_mfa'})
    first=mfa_one._prewarm_audio.cache_info()
    mfa_one._prewarm_audio()
    second=mfa_one._prewarm_audio.cache_info()
errors=[s for s in log.getvalue().splitlines() if '예열 실패' in s]
report={'ping':reply,'lexicon_cache_entries':len(mfa_one._LEXICON),
        'audio_warmup_cache_before':first._asdict(),'audio_warmup_cache_after':second._asdict(),
        'tokenizer_cache':mfa_one._tokenizer.cache_info()._asdict(),'warmup_errors':errors,
        'seconds':time.monotonic()-started,'stderr':log.getvalue()}
report['pass']=reply.get('ok') is True and bool(mfa_one._LEXICON) and first.misses==1 and second.hits==first.hits+1 and not errors
Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='stderr'},ensure_ascii=False))
raise SystemExit(0 if report['pass'] else 1)
