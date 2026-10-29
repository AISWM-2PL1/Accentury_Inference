"""Check that the current release still describes these sources and build inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true', help='Also query ECR with existing AWS read credentials')
    args = parser.parse_args()
    current = read_json(ROOT / 'releases/current.json')
    version = current['version']
    require(re.fullmatch(r'v\d{4}\.\d{2}\.\d{2}-kan\d+-[0-9a-f]{7}-slim\.[1-9]\d*', version), 'Invalid release version')
    expected_path = f'releases/{version}/manifest.json'
    require(current['manifest'] == expected_path, 'Current release path mismatch')
    manifest_path = ROOT / expected_path
    release = manifest_path.parent
    manifest = read_json(manifest_path)
    require(manifest['version'] == version, 'Manifest version mismatch')
    require(manifest['status'] == 'published', 'Release publication is incomplete')
    require(manifest['image']['tag_uri'].endswith(':' + version), 'Image tag mismatch')
    require(manifest['image']['digest_uri'].endswith('@' + manifest['image']['digest']), 'Image digest mismatch')
    require(manifest['image']['remote_digest_verified'] is True, 'Image publication is unverified')
    require(manifest['base']['remote_digest_verified'] is True, 'Base mirror publication is unverified')
    require(manifest['image']['tag_mutability'] == 'IMMUTABLE', 'Use immutable image tags')
    require(manifest['archive']['contains_tag'] == manifest['image']['tag_uri'], 'Archive tag mismatch')
    require(manifest['archive']['loaded_image_id'] == manifest['image']['digest'], 'Archive image mismatch')
    require(manifest['archive']['reload_verified'] is True, 'Archive reload is unverified')
    for value in [manifest['base']['digest'], manifest['image']['digest']]:
        require(re.fullmatch(r'sha256:[0-9a-f]{64}', value), 'Invalid digest')
    dockerfile = ROOT / 'docker/slim/Dockerfile'
    require(dockerfile.read_text().splitlines()[0] == f"FROM {manifest['base']['digest_uri']} AS pruned", 'Base image must be digest-pinned')
    tested = (ROOT / 'docker/slim/Dockerfile.tested').read_text()
    mirrored = tested.replace(manifest['base']['original_digest_uri'], manifest['base']['digest_uri'], 1)
    require(mirrored == dockerfile.read_text(), 'Mirror build recipe differs beyond the base repository address')
    for relative, expected in manifest['repository_file_sha256'].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), 'Invalid repository file path')
        require(path.is_file() and sha256(path) == expected, f'Release no longer matches {relative}; create a new tested release')
    sources = read_json(release / 'source-files.json')
    require(sources['github_commit'] == manifest['source']['commit'], 'Source commit mismatch')
    expected_sources = {entry['path'] for entry in sources['files']}
    actual_sources = {str(path.relative_to(ROOT / 'src')) for path in (ROOT / 'src').rglob('*.py')}
    require(actual_sources == expected_sources, 'Runtime source inventory changed; create a new tested release')
    for entry in sources['files']:
        require(entry['equal'] and sha256(ROOT / 'src' / entry['path']) == entry['image_sha256'] == entry['github_sha256'], f"Image/source mismatch: {entry['path']}")
    evidence = read_json(release / 'validation.json')
    require(evidence['baseline_digest'] == manifest['base']['digest'], 'Validation baseline mismatch')
    require(evidence['slim_digest'] == manifest['image']['digest'], 'Validation image mismatch')
    require(evidence['comparison']['pass'] and not evidence['comparison']['differences'], 'Functional comparison failed')
    checksum_names = set()
    for line in (release / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        require(Path(name).name == name, 'Invalid checksum path')
        require(sha256(release / name) == expected, f'Release evidence checksum mismatch: {name}')
        checksum_names.add(name)
    require(checksum_names == {'manifest.json', 'validation.json', 'source-files.json', 'README.md'}, 'Release checksum inventory mismatch')
    require(f'releases/{version}/README.md' in (ROOT / 'README.md').read_text(), 'README current release link is stale')
    if args.remote:
        registry = manifest['registry']
        common = ['--region', registry['region'], '--registry-id', registry['account_id'], '--output', 'json']
        def aws(*arguments):
            return subprocess.run(['aws', 'ecr', *arguments, *common], capture_output=True, text=True)
        response = aws('describe-repositories', '--repository-names', registry['repository'])
        require(response.returncode == 0, 'ECR repository lookup failed: ' + response.stderr)
        repository = json.loads(response.stdout)['repositories'][0]
        require(repository['imageTagMutability'] == 'IMMUTABLE', 'ECR tag immutability changed')
        response = aws('get-lifecycle-policy', '--repository-name', registry['repository'])
        require(response.returncode != 0 and 'LifecyclePolicyNotFoundException' in response.stderr, 'ECR expiry policy is present or lookup failed')
        for entry in [manifest['base'], manifest['image']]:
            tag = entry['tag_uri'].rsplit(':', 1)[1]
            response = aws('describe-images', '--repository-name', registry['repository'], '--image-ids', 'imageTag=' + tag)
            require(response.returncode == 0, 'ECR image lookup failed: ' + response.stderr)
            require(json.loads(response.stdout)['imageDetails'][0]['imageDigest'] == entry['digest'], 'ECR tag/digest mismatch')
        print('ECR: both image digests, immutable tags and absence of automatic expiry policy confirmed')
    print(f'{version}: source, build inputs, evidence checksums and image references match')


if __name__ == '__main__':
    main()
