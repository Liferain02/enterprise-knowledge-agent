#!/usr/bin/env python3
"""Acquire real, version-pinned RDMA reference documents with provenance.

No generated filler or artificial subdivision: one upstream manual = one source.
Downloads an archive through the proxy; never executes/extracts upstream code.
"""
import argparse
import hashlib
import io
import json
import tarfile
import time
import urllib.request
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "linux-rdma/rdma-core"
REVISION = "00357df3a344b3515a9a4d345cf0b8c8efb7f116"


def fetch(proxy, destination):
    url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{REVISION}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    for attempt in range(4):
        try:
            with opener.open(url, timeout=60) as response:
                archive = response.read()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    destination.mkdir(parents=True, exist_ok=True)
    records, hashes = [], set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in sorted(bundle.getmembers(), key=lambda m: m.name):
            relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if not member.isfile() or ".." in relative.parts or relative.is_absolute():
                continue
            # Preserve upstream license files separately from searchable manuals.
            is_license = relative.parts and relative.parts[0] == "COPYING"
            is_manual = "man" in relative.parts and (str(relative).endswith(".md")
                                                     or relative.suffix in {".1", ".3", ".5", ".7", ".8"})
            if not (is_license or is_manual):
                continue
            body = bundle.extractfile(member).read()
            digest = hashlib.sha256(body).hexdigest()
            if is_license:
                path = destination / "licenses" / str(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                continue
            # Exclude symlinks, aliases/stubs and identical content.
            if len(body.strip()) < 250 or digest in hashes:
                continue
            hashes.add(digest)
            path = destination / "manuals" / str(relative)
            original_digest = digest
            if relative.suffix != ".md":
                # Safe-mode roff rendering; keep original alongside provenance.
                original = destination / "upstream" / str(relative)
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(body)
                rendered = subprocess.run(["groff", "-S", "-man", "-Tutf8"], input=body,
                                          capture_output=True, check=True, timeout=10).stdout.decode()
                while "\b" in rendered:
                    rendered = re.sub(r".\x08", "", rendered)
                body = rendered.strip().encode() + b"\n"
                path = path.with_suffix(path.suffix + ".txt")
                digest = hashlib.sha256(body).hexdigest()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            records.append({
                "path": str(path.relative_to(ROOT)), "sha256": digest, "upstream_sha256": original_digest,
                "repository": REPOSITORY, "revision": REVISION,
                "source_url": f"https://github.com/{REPOSITORY}/blob/{REVISION}/{relative}",
                "upstream_path": str(relative), "license_path": str(destination.relative_to(ROOT) / "licenses"),
                "source_kind": "external_reference", "topic": "RDMA / InfiniBand / verbs",
            })
    manifest = {"repository": REPOSITORY, "revision": REVISION,
                "archive_url": url, "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "document_count": len(records), "documents": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"documents": len(records), "destination": str(destination),
                      "archive_sha256": manifest["archive_sha256"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    args = parser.parse_args()
    fetch(args.proxy, ROOT / "data/knowledge/public/rdma-core")
