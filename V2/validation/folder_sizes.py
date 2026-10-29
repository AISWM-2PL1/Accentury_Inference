import collections,json,os,stat
from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
roots=[('whisper','/opt/hf'),('mfa_env','/opt/conda/envs/mfa'),('conda_cache','/opt/conda/pkgs'),('mfa_models','/opt/mfa'),('python_packages','/usr/local/lib/python3.12/site-packages'),('transformers_vendor','/app/vendor/transformers'),('reference_data','/data'),('application','/app'),('conda_other','/opt/conda'),('other','/')]
records=[];seen_paths=set()
exclude=('/proc','/sys','/dev','/audit','/out')
for group,root in roots:
 if not Path(root).is_dir():continue
 for d,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[x for x in dirs if not os.path.islink(os.path.join(d,x)) and os.path.join(d,x) not in exclude and not (root=='/' and any(os.path.join(d,x)==r for _,r in roots[:-1]))]
  for f in files:
   p=os.path.join(d,f)
   if p in seen_paths:continue
   seen_paths.add(p)
   if root=='/' and p in ('/etc/hosts','/etc/hostname','/etc/resolv.conf'):continue
   s=os.lstat(p)
   if stat.S_ISREG(s.st_mode):records.append((p,group,s.st_dev,s.st_ino,s.st_size,s.st_blocks*512))
seen=set();groups=collections.defaultdict(lambda:{'unique_file_bytes':0,'allocated_file_bytes':0,'unique_files':0})
for p,g,dev,ino,n,blocks in records:
 key=(dev,ino)
 if key in seen:continue
 seen.add(key);v=groups[g];v['unique_file_bytes']+=n;v['allocated_file_bytes']+=blocks;v['unique_files']+=1
paths=['/opt/hf','/opt/hf/hub','/opt/hf/hub/models--openai--whisper-large-v3/blobs','/opt/conda','/opt/conda/envs/mfa','/opt/conda/envs/mfa/lib','/opt/conda/envs/mfa/lib/python3.11','/opt/conda/envs/mfa/lib/python3.11/site-packages','/opt/conda/envs/mfa/include','/opt/conda/envs/mfa/bin','/opt/conda/envs/mfa/share','/opt/conda/pkgs','/opt/mfa','/usr/local/lib/python3.12/site-packages','/app/vendor/transformers','/data','/app','/usr/local/include','/usr/share/doc','/usr/share/man','/opt/slim-audit']
for parent in ['/opt/conda/envs/mfa/lib/python3.11/site-packages','/usr/local/lib/python3.12/site-packages','/data','/opt/mfa']:
 if Path(parent).exists():paths.extend(str(p) for p in Path(parent).iterdir() if p.is_dir() and not p.is_symlink())
paths += ['/usr/local/lib/python3.12/site-packages/torch/'+x for x in ['lib','include','bin','_inductor','testing','__pycache__']]
paths += [base+'/'+x for base in ['/usr/local/lib/python3.12/site-packages/transformers','/app/vendor/transformers'] for x in ['models','generation','utils']]
folders={}
for folder in sorted(set(paths)):
 own={}
 for p,g,dev,ino,n,blocks in records:
  if p.startswith(folder+'/'):own[(dev,ino)]=(n,blocks)
 folders[folder]={'file_bytes':sum(x[0] for x in own.values()),'allocated_file_bytes':sum(x[1] for x in own.values()),'files':len(own)}
aggregates={}
for group in ['mfa_env','python_packages']:
 for kind in ['pycache','tests','headers','static_archives','pip']:
  own={}
  for p,g,dev,ino,n,blocks in records:
   if g!=group:continue
   parts=Path(p).parts
   match={'pycache':'__pycache__' in parts,'tests':any(x in ('test','tests') for x in parts),'headers':'include' in parts,'static_archives':p.endswith('.a'),'pip':'pip' in parts}[kind]
   if match:own[(dev,ino)]=n
  aggregates[group+'_'+kind]=sum(own.values())
report={'basis':'uncompressed regular-file content bytes; hardlinks counted once globally, MFA env before conda cache; individual folder totals dedup within that folder and may overlap','groups':dict(groups),'total_file_bytes':sum(x['unique_file_bytes'] for x in groups.values()),'folders':folders,'aggregates':aggregates}
Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps({'groups':dict(groups),'total_file_bytes':report['total_file_bytes']}))
