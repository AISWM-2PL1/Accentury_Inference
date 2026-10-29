"""Verify visible version folders against their preserved source/image records."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = 'a6f409668c95ee10e301090c53265e30227ca85a'
RELEASE_COMMIT = 'b9445ae51726c4c9a8371283697b88eb3d6ad285'
RELEASE_TAG = 'v2026.10.29-kan296-0d72c03-slim.1'
RELEASE_PATH = f'releases/{RELEASE_TAG}'


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True)
    require(result.returncode == 0, 'Git history lookup failed; fetch full history and tags: ' + result.stderr.decode())
    return result.stdout


def snapshot(commit):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', commit))) as archive:
        for entry in archive:
            if entry.isdir():
                continue
            path = Path(entry.name)
            require(entry.isfile() and not path.is_absolute() and '..' not in path.parts, 'Unexpected historical archive entry')
            files[entry.name] = archive.extractfile(entry).read()
    return files


def inventory(folder):
    return {str(p.relative_to(folder)) for p in folder.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def check_files(folder, expected, editable_docs=()):
    require(inventory(folder) == set(expected) | set(editable_docs), f'{folder.relative_to(ROOT)} file inventory changed')
    for relative, data in expected.items():
        path = folder / relative
        require(not path.is_symlink() and path.read_bytes() == data, f'{path.relative_to(ROOT)} differs from its published source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true', help='Also check both ECR images and retention settings')
    parser.add_argument('--out', type=Path, help='Write verification evidence outside the repository')
    args = parser.parse_args()
    require(git('rev-parse', RELEASE_TAG + '^{}').decode().strip() == RELEASE_COMMIT, 'Historical release tag was moved')
    source, published = snapshot(SOURCE_COMMIT), snapshot(RELEASE_COMMIT)
    manifest = json.loads(published[RELEASE_PATH + '/manifest.json'])
    evidence = json.loads(published[RELEASE_PATH + '/validation.json'])
    require(manifest['source']['commit'] == SOURCE_COMMIT, 'Historical source mismatch')
    require(manifest['image']['digest'] != manifest['base']['digest'], 'Version image identities collapsed')
    for removed in ['src', 'registry', 'docker', 'validation', 'tools', 'releases']:
        require(not (ROOT / removed).exists(), f'Unversioned root folder remains: {removed}')

    runtime_roots = ['src', 'registry']
    runtime_files = ['dict_aug_track1.txt', 'requirements.txt']
    records = {}
    for version, key, name, size_key in [('V1', 'base', 'MFA 최적화 원본', 'baseline'), ('V2', 'image', 'V1 용량 경량화본', 'slim')]:
        folder = ROOT / version
        for group in runtime_roots:
            expected = {p[len(group)+1:]: data for p, data in source.items() if p.startswith(group + '/')}
            check_files(folder / group, expected)
        for path in runtime_files:
            require((folder / path).read_bytes() == source[path], f'{version}/{path} differs from source')
        data = json.loads((folder / 'version.json').read_text())
        expected = {'version': version, 'name': name, 'based_on': None if version == 'V1' else 'V1',
                    'source_commit': SOURCE_COMMIT,
                    'image': {k: manifest[key][k] for k in ['tag_uri', 'digest_uri', 'digest']},
                    'platform': 'linux/amd64', 'model_code_hash': '0d72c03',
                    'sizes': evidence['sizes'][size_key],
                    'historical_release': {'tag': RELEASE_TAG, 'commit': RELEASE_COMMIT,
                                           'manifest_path': RELEASE_PATH + '/manifest.json',
                                           'url': 'https://github.com/AISWM-2PL1/Accentury_Inference/releases/tag/' + RELEASE_TAG},
                    'image_rebuilt_for_folder_reorganization': False, 'production_deployed': False}
        require(data == expected, f'{version}/version.json no longer describes its published image')
        readme = (folder / 'README.md').read_text()
        require(data['image']['digest_uri'] in readme and '[src/](src/)' in readme, f'{version} image/source navigation is stale')
        count = sum(path.startswith(('src/', 'registry/')) or path in runtime_files for path in source)
        records[version] = {'source_files_equal': count, 'image_digest': data['image']['digest']}

    # Build and model-test code keep their exact bytes; test README explains its new cwd.
    for group in ['docker', 'validation']:
        expected = {p[len(group)+1:]: data for p, data in published.items() if p.startswith(group + '/')}
        docs = ['README.md'] if group == 'validation' else []
        for path in docs:
            expected.pop(path)
        check_files(ROOT / 'V2' / group, expected, docs)
    readme = (ROOT / 'README.md').read_text()
    for version in records:
        require(f'({version}/README.md)' in readme and f'({version}/src/)' in readme, 'Root version navigation is stale')

    # Validate immutable records in the exact Git snapshot where they were published.
    # This lets current navigation change without rewriting historical checksums.
    with tempfile.TemporaryDirectory(prefix='accentury-preserved-release-') as directory:
        folder = Path(directory)
        for relative, data in published.items():
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        command = [sys.executable, str(folder / 'tools/verify_release.py')]
        if args.remote:
            command.append('--remote')
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
        require(result.returncode == 0, result.stdout + result.stderr)
        print(result.stdout.strip())
    result = {'pass': True, 'versions': records, 'historical_release_commit': RELEASE_COMMIT,
              'historical_tag_preserved': True, 'build_and_test_code_unchanged': True,
              'historical_release_checks_pass': True, 'remote_checked': args.remote,
              'image_rebuilt': False, 'inference_rerun': False}
    if args.out:
        require(not args.out.resolve().is_relative_to(ROOT), 'Write test evidence outside the repository')
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('V1 original / V2 slim: source snapshots, image identities and preserved records verified')


if __name__ == '__main__':
    main()
