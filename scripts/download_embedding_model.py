#!/usr/bin/env python3
"""Download a pinned BGE-M3 snapshot; runtime subsequently uses local files only."""
import os
import time
from pathlib import Path

os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

if __name__ == "__main__":
    for attempt in range(4):
        try:
            snapshot_download("BAAI/bge-m3", revision=REVISION,
                              local_dir=ROOT / ".run/models/bge-m3",
                              allow_patterns=["*.json", "1_Pooling/*.json", "pytorch_model.bin",
                                              "sentencepiece.bpe.model", "README.md"], max_workers=3)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    (ROOT / ".run/models/bge-m3/REVISION").write_text(REVISION + "\n")
    snapshot_download("BAAI/bge-reranker-v2-m3", revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                      allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model", "README.md"],
                      max_workers=3)
    print("Pinned BGE-M3 downloaded; runtime requires no Hugging Face connection.")
