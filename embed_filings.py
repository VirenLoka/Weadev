#!/usr/bin/env python3
"""
Embed SEC filing sections from a financial-reports parquet file using
NVIDIA/NV-Embed-v2 and store the resulting vectors + rich metadata in a
local LanceDB vector database.

Configuration is driven by a YAML file (default: configs/train_config.yaml).
Any field can be overridden on the command line.

Pipeline
--------
1. Load YAML config; apply any CLI overrides.
2. Read the parquet file with Polars.
3. Walk every row -> every filing -> every report section -> every text chunk.
   Each individual string in a section's List(String) becomes one embedding
   (one chunk == one row in LanceDB).
4. Embed chunks in batches with NV-Embed-v2 (4096-dim, bfloat16, normalized).
5. Write vectors + rich metadata into a LanceDB table.

The parquet file does NOT need to exist when this script is written, but it
must exist when the script is *run*.

Example
-------
    python embed_filings.py
    python embed_filings.py --config configs/train_config.yaml
    python embed_filings.py --parquet /data/train.parquet --limit-rows 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml


DEFAULT_CONFIG = "configs/train_config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def build_config(yaml_path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML config with CLI overrides (CLI wins)."""
    cfg = load_yaml(yaml_path)
    for key, value in overrides.items():
        if value is None:
            continue
        # Map flat CLI keys to nested config paths.
        mapping = {
            "parquet":        ("input", "parquet_path"),
            "limit_rows":     ("input", "limit_rows"),
            "lancedb":        ("output", "lancedb_path"),
            "table":          ("output", "table_name"),
            "write_mode":     ("output", "write_mode"),
            "batch_size":     ("processing", "batch_size"),
            "flush_size":     ("processing", "flush_size"),
            "min_chars":      ("processing", "min_chars"),
            "model_name":     ("model", "name"),
            "max_seq_length": ("model", "max_seq_length"),
        }
        if key in mapping:
            section, field = mapping[key]
            cfg.setdefault(section, {})[field] = value
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed SEC filing sections with NV-Embed-v2 into LanceDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to YAML config file.",
    )
    # Optional per-run overrides — all default to None so unset ones are ignored.
    p.add_argument("--parquet", default=None, help="Override input.parquet_path.")
    p.add_argument("--lancedb", default=None, help="Override output.lancedb_path.")
    p.add_argument("--table", default=None, help="Override output.table_name.")
    p.add_argument("--write-mode", dest="write_mode", default=None,
                   choices=["overwrite", "append"],
                   help="Override output.write_mode.")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                   help="Override processing.batch_size.")
    p.add_argument("--flush-size", dest="flush_size", type=int, default=None,
                   help="Override processing.flush_size.")
    p.add_argument("--min-chars", dest="min_chars", type=int, default=None,
                   help="Override processing.min_chars.")
    p.add_argument("--max-seq-length", dest="max_seq_length", type=int, default=None,
                   help="Override model.max_seq_length.")
    p.add_argument("--model-name", dest="model_name", default=None,
                   help="Override model.name.")
    p.add_argument("--limit-rows", dest="limit_rows", type=int, default=None,
                   help="Override input.limit_rows (first N company rows only).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, max_seq_length: int, dtype_str: str, trust_remote_code: bool):
    import torch
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "CUDA is not available! Check your drivers."
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"\nLoading {model_name}...")
    model = SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    model.max_seq_length = max_seq_length
    model.tokenizer.padding_side = "right"
    return model


def add_eos(model, texts: list[str]) -> list[str]:
    """NV-Embed-v2 requires an explicit EOS token appended to each passage."""
    eos = model.tokenizer.eos_token
    return [t + eos for t in texts]


# ---------------------------------------------------------------------------
# LanceDB schema
# ---------------------------------------------------------------------------

def lancedb_schema(embed_dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), embed_dim)),
        pa.field("text", pa.string()),
        pa.field("cik", pa.string()),
        pa.field("name", pa.string()),
        pa.field("section", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("form", pa.string()),
        pa.field("filingDate", pa.string()),
        pa.field("reportDate", pa.string()),
        pa.field("tickers", pa.list_(pa.string())),
        pa.field("label_1d", pa.string()),
        pa.field("label_5d", pa.string()),
        pa.field("label_30d", pa.string()),
        pa.field("ret_1d", pa.float64()),
        pa.field("ret_5d", pa.float64()),
        pa.field("ret_30d", pa.float64()),
    ])


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur


def iter_chunks(row: dict, sections: list[str], min_chars: int):
    """Yield (text, metadata_dict) for every text chunk in every filing."""
    cik = row.get("cik")
    name = row.get("name")
    tickers = row.get("tickers") or []

    for filing in (row.get("filings") or []):
        if filing is None:
            continue
        labels = filing.get("labels") or {}
        returns = filing.get("returns") or {}
        report = filing.get("report") or {}

        meta_common = {
            "cik": cik,
            "name": name,
            "tickers": list(tickers),
            "form": filing.get("form"),
            "filingDate": filing.get("filingDate"),
            "reportDate": filing.get("reportDate"),
            "label_1d": labels.get("1d"),
            "label_5d": labels.get("5d"),
            "label_30d": labels.get("30d"),
            "ret_1d": _safe(returns, "1d", "ret"),
            "ret_5d": _safe(returns, "5d", "ret"),
            "ret_30d": _safe(returns, "30d", "ret"),
        }

        for section in sections:
            for idx, text in enumerate(report.get(section) or []):
                if text is None:
                    continue
                text = text.strip()
                if len(text) < min_chars:
                    continue
                meta = dict(meta_common)
                meta["text"] = text
                meta["section"] = section
                meta["chunk_index"] = idx
                yield text, meta


# ---------------------------------------------------------------------------
# Batch encode + write
# ---------------------------------------------------------------------------

def flush_buffer(model, table, batch_size: int, normalize: bool,
                 texts: list[str], metas: list[dict]) -> int:
    if not texts:
        return 0

    embeddings = model.encode(
        add_eos(model, texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )

    rows = []
    for vec, meta in zip(embeddings, metas):
        record = dict(meta)
        record["vector"] = vec.astype("float32").tolist()
        rows.append(record)

    table.add(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 1

    cfg = build_config(args.config, vars(args))

    # Resolve config sections with defaults.
    parquet_path   = _deep_get(cfg, "input", "parquet_path")
    limit_rows     = _deep_get(cfg, "input", "limit_rows")
    sections       = _deep_get(cfg, "sections") or []
    model_name     = _deep_get(cfg, "model", "name", default="nvidia/NV-Embed-v2")
    embed_dim      = _deep_get(cfg, "model", "embed_dim", default=4096)
    dtype_str      = _deep_get(cfg, "model", "dtype", default="bfloat16")
    max_seq_length = _deep_get(cfg, "model", "max_seq_length", default=32768)
    normalize      = _deep_get(cfg, "model", "normalize_embeddings", default=True)
    trust_rc       = _deep_get(cfg, "model", "trust_remote_code", default=True)
    batch_size     = _deep_get(cfg, "processing", "batch_size", default=256)
    flush_size     = _deep_get(cfg, "processing", "flush_size", default=4096)
    min_chars      = _deep_get(cfg, "processing", "min_chars", default=1)
    lancedb_path   = _deep_get(cfg, "output", "lancedb_path", default="lancedb/")
    table_name     = _deep_get(cfg, "output", "table_name", default="filings_sections")
    write_mode     = _deep_get(cfg, "output", "write_mode", default="overwrite")

    print("=== Configuration ===")
    print(f"  parquet      : {parquet_path}")
    print(f"  sections     : {len(sections)} sections")
    print(f"  model        : {model_name} ({dtype_str}, dim={embed_dim})")
    print(f"  batch_size   : {batch_size}  flush_size: {flush_size}")
    print(f"  lancedb      : {lancedb_path}  table: {table_name}  mode: {write_mode}")
    if limit_rows:
        print(f"  limit_rows   : {limit_rows}")
    print()

    if not os.path.exists(parquet_path):
        print(f"ERROR: parquet file not found: {parquet_path}", file=sys.stderr)
        print("Generate the dataset first, then re-run this script.", file=sys.stderr)
        return 1

    import lancedb
    import polars as pl

    Path(lancedb_path).mkdir(parents=True, exist_ok=True)
    print(f"Connecting to LanceDB at: {lancedb_path}")
    db = lancedb.connect(lancedb_path)
    table = db.create_table(table_name, schema=lancedb_schema(embed_dim), mode=write_mode)
    print(f"Table '{table_name}' ready (mode={write_mode}).")

    model = load_model(model_name, max_seq_length, dtype_str, trust_rc)

    print(f"\nReading parquet: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    if limit_rows:
        df = df.head(limit_rows)
    print(f"Loaded {df.height} company rows.")

    text_buf: list[str] = []
    meta_buf: list[dict] = []
    total_written = 0
    start = time.time()

    for row in df.iter_rows(named=True):
        for text, meta in iter_chunks(row, sections, min_chars):
            text_buf.append(text)
            meta_buf.append(meta)

            if len(text_buf) >= flush_size:
                total_written += flush_buffer(
                    model, table, batch_size, normalize, text_buf, meta_buf
                )
                text_buf, meta_buf = [], []
                print(f"  ... {total_written:,} chunks written so far")

    total_written += flush_buffer(model, table, batch_size, normalize, text_buf, meta_buf)

    elapsed = time.time() - start
    print(f"\nDone. Wrote {total_written:,} embedded chunks in {elapsed:.1f}s.")
    print(f"Table '{table_name}' now has {table.count_rows():,} rows ({embed_dim}-dim vectors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
