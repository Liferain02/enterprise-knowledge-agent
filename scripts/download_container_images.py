#!/usr/bin/env python3
"""Download verified Docker archives through a user proxy, without changing dockerd.

Only linux/amd64 official Redis/Qdrant images are accepted. Every compressed layer,
uncompressed diff_id and image config is checked before invoking docker load.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ACCEPT = ', '.join((
    'application/vnd.oci.image.index.v1+json',
    'application/vnd.docker.distribution.manifest.list.v2+json',
    'application/vnd.oci.image.manifest.v1+json',
    'application/vnd.docker.distribution.manifest.v2+json',
))


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()


def download_image(repo: str, reference: str, directory: Path, proxy: str) -> dict:
    if repo not in {'library/redis', 'qdrant/qdrant', 'library/mysql'}:
        raise ValueError('Only the official Redis/Qdrant/MySQL repositories are allowed')
    directory.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))
    token_url = 'https://auth.docker.io/token?' + urllib.parse.urlencode({
        'service': 'registry.docker.io', 'scope': f'repository:{repo}:pull',
    })
    with opener.open(token_url, timeout=30) as response:
        token = json.load(response)['token']
    headers = {'Authorization': 'Bearer ' + token, 'Accept': ACCEPT}
    base = f'https://registry-1.docker.io/v2/{repo}/'

    def get(path: str, target: Path) -> None:
        for attempt in range(4):
            try:
                request = urllib.request.Request(base + path, headers=headers)
                with opener.open(request, timeout=60) as response, target.open('wb') as output:
                    shutil.copyfileobj(response, output)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    manifest_path = directory / 'registry-manifest.json'
    get('manifests/' + reference, manifest_path)
    if reference.startswith('sha256:') and digest(manifest_path) != reference:
        raise ValueError('Manifest digest mismatch')
    manifest = json.loads(manifest_path.read_text())
    if 'manifests' in manifest:
        platform = next(item for item in manifest['manifests']
                        if item.get('platform', {}).get('os') == 'linux'
                        and item.get('platform', {}).get('architecture') == 'amd64')
        get('manifests/' + platform['digest'], manifest_path)
        if digest(manifest_path) != platform['digest']:
            raise ValueError('Platform manifest digest mismatch')
        manifest = json.loads(manifest_path.read_text())
    manifest_digest = digest(manifest_path)
    config_digest = manifest['config']['digest']
    config_path = directory / (config_digest.split(':')[1] + '.json')
    get('blobs/' + config_digest, config_path)
    if digest(config_path) != config_digest:
        raise ValueError('Config digest mismatch')
    config = json.loads(config_path.read_text())
    layers = []
    for index, layer in enumerate(manifest['layers']):
        blob = directory / (layer['digest'].split(':')[1] + '.blob')
        if not blob.exists() or digest(blob) != layer['digest']:
            print(f'{repo} layer {index + 1}/{len(manifest["layers"])}: {layer["size"]} bytes', flush=True)
            get('blobs/' + layer['digest'], blob)
        if digest(blob) != layer['digest']:
            raise ValueError('Compressed layer digest mismatch')
        unpacked = directory / f'layer-{index}.tar'
        reader = gzip.open if 'gzip' in layer.get('mediaType', '') else open
        with reader(blob, 'rb') as source, unpacked.open('wb') as output:
            shutil.copyfileobj(source, output)
        if digest(unpacked) != config['rootfs']['diff_ids'][index]:
            raise ValueError('Uncompressed layer digest mismatch')
        layers.append(unpacked.name)
    tag = repo.removeprefix('library/') + ':offline-' + manifest_digest.split(':')[1][:16]
    archive_manifest = directory / 'manifest.json'
    archive_manifest.write_text(json.dumps([{'Config': config_path.name, 'RepoTags': [tag], 'Layers': layers}]))
    archive = directory / 'image.tar'
    with tarfile.open(archive, 'w') as output:
        for name in [config_path.name, 'manifest.json', *layers]:
            output.add(directory / name, arcname=name)
    subprocess.run(['docker', 'load', '-i', str(archive)], check=True)
    image_id = subprocess.check_output(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'], text=True).strip()
    if image_id != config_digest:
        raise ValueError('Loaded image ID mismatch')
    record = {'repository': repo, 'reference': reference, 'manifest_digest': manifest_digest,
              'image_id': image_id, 'local_tag': tag, 'architecture': config.get('architecture'),
              'compressed_bytes': sum(x['size'] for x in manifest['layers'])}
    (directory / 'verified-image.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--output', type=Path, default=Path('.run/container-images'))
    args = parser.parse_args()
    # Platform manifests inspected on 2026-09-20; mutable latest tags are not used.
    images = [
        ('library/redis', 'sha256:ca0acbb137c1dc3339c8b147a58fd6f42775d4599327b50e7b116c23de501af2'),
        ('qdrant/qdrant', 'sha256:0699e7733a6fa7fa7f6b95dcbed84ebb04584110da525cdfdef9f305c4f57738'),
        ('library/mysql', 'sha256:f015b98a954d6bb92c370d93f354b85f4bcea15642d9fb2975f841a00d400b65'),
    ]
    records = [download_image(repo, ref, args.output / repo.split('/')[-1], args.proxy) for repo, ref in images]
    (args.output / 'images.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
