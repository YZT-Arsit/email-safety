#!/usr/bin/env python
"""Download a local multilingual BERT checkpoint from ModelScope for offline use.
Inputs: ModelScope model id and cache dir. Outputs: local model directory and download summary JSON."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


DEFAULT_MODEL_ID = "AI-ModelScope/bert-base-multilingual-cased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model from ModelScope to local path")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=str, default="models")
    return parser.parse_args()


def _validate_model_id(model_id: str) -> None:
    if "/" not in model_id:
        raise ValueError(
            "Invalid ModelScope model id. Expected `namespace/name`, "
            f"got `{model_id}`. Example: `{DEFAULT_MODEL_ID}`"
        )


def main() -> None:
    args = parse_args()
    _validate_model_id(args.model_id)

    local_dir = Path(args.cache_dir) / Path(args.model_id).name
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    download_path = snapshot_download(model_id=args.model_id, cache_dir=str(local_dir.parent))
    resolved_dir = Path(download_path)
    if resolved_dir.exists() and resolved_dir.resolve() != local_dir.resolve():
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(resolved_dir, local_dir)
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
    final_dir = local_dir if local_dir.exists() else resolved_dir

    files = {p.name for p in final_dir.iterdir()} if final_dir.exists() else set()
    tokenizer_files = [name for name in files if "tokenizer" in name or name in {"vocab.txt", "sentencepiece.bpe.model"}]
    weight_files = [name for name in files if name.endswith((".bin", ".safetensors"))]

    summary = {
        "model_id": args.model_id,
        "requested_cache_dir": str(local_dir),
        "download_path": str(resolved_dir),
        "resolved_model_dir": str(final_dir),
        "exists": final_dir.exists(),
        "has_config_json": (final_dir / "config.json").exists(),
        "tokenizer_files": tokenizer_files,
        "weight_files": weight_files,
        "has_tokenizer_files": bool(tokenizer_files),
        "has_weight_files": bool(weight_files),
    }

    output_dir = Path("outputs/modelscope_download")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "download_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
