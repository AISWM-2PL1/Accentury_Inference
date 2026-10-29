"""Read docker-save OCI archive as a stream; measure actual compressed/raw layers."""
import argparse
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

ap = argparse.ArgumentParser()
ap.add_argument('image')
ap.add_argument('--out', required=True)
args = ap.parse_args()
inspect = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image]))[0]
p = subprocess.Popen(['docker', 'image', 'save', args.image], stdout=subprocess.PIPE)
objects, blobs = {}, {}
with tarfile.open(fileobj=p.stdout, mode='r|') as archive:
    for member in archive:
        if not member.isfile(): continue
        f = archive.extractfile(member)
        name = member.name.removeprefix('./')
        if member.size < 2_000_000:
            data = f.read()
            if data[:2] == b'\x1f\x8b':
                blobs[name] = {'archive_bytes':member.size,'uncompressed_tar_bytes':len(gzip.decompress(data)),'gzip':True}
            else:
                try: objects[name] = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    blobs[name] = {'archive_bytes':member.size,'uncompressed_tar_bytes':member.size,'gzip':False}
        else:
            reader = io.BufferedReader(f)
            compressed = reader.peek(2)[:2] == b'\x1f\x8b'
            total = 0
            if compressed:
                with gzip.GzipFile(fileobj=reader) as z:
                    while chunk := z.read(4*1024*1024): total += len(chunk)
            else:
                total = member.size
            blobs[name] = {'archive_bytes': member.size, 'uncompressed_tar_bytes': total, 'gzip': compressed}
assert p.wait() == 0
manifests = []
for name, obj in objects.items():
    if isinstance(obj, dict) and 'layers' in obj and 'config' in obj:
        cfg_path = 'blobs/sha256/' + obj['config']['digest'].split(':')[1]
        cfg = objects.get(cfg_path, {})
        if cfg.get('architecture') != 'amd64': continue
        layers = []
        for layer in obj['layers']:
            path = 'blobs/sha256/' + layer['digest'].split(':')[1]
            entry = dict(layer)
            assert path in blobs, ('layer bytes not measured',path)
            entry.update(blobs[path])
            layers.append(entry)
        manifests.append({'manifest': name, 'layers': layers,
                          'compressed_layers_bytes': sum(x['size'] for x in layers),
                          'uncompressed_layers_tar_bytes': sum(x['uncompressed_tar_bytes'] for x in layers)})
report = {'image': args.image, 'image_id': inspect['Id'], 'docker_inspect_size': inspect['Size'],
          'architecture': inspect['Architecture'], 'manifests': manifests,
          'note': 'OCI layer bytes from docker save; uncompressed bytes include tar headers/padding, not filesystem allocated blocks.'}
assert len(manifests) == 1, [x['manifest'] for x in manifests]
Path(args.out).write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'manifests'}))
print(json.dumps({k:v for k,v in manifests[0].items() if k != 'layers'}))
