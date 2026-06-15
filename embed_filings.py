#!/usr/bin/env python3
"""
Embed SEC filing sections from a financial-reports parquet file using
NVIDIA/NV-Embed-v2 and store the resulting vectors + rich metadata in a
local LanceDB vector database.

Pipeline
--------
1. Read the parquet file with Polars.
2. Walk every row -> every filing -> every report section -> every text chunk.
   Each individual string in a section's List(String) becomes one embedding
   (one chunk == one row in LanceDB).
3. Embed chunks in batches with NV-Embed-v2 (4096-dim, bfloat16, normalized).
4. Write vectors + rich metadata into a LanceDB table.

The parquet file does NOT need to exist when this script is written, but it
must exist when the script is *run*.

Example
-------
    python embed_filings.py \
        --parquet weadev/financial_reports_dataset/main/train_full.parquet \
        --lancedb lancedb/ \
        --table filings_sections \
        --batch-size 256
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import polars as pl
import pyarrow as pa


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PARQUET = "weadev/financial_reports_dataset/main/train_full.parquet"
DEFAULT_LANCEDB = "lancedb/"
DEFAULT_TABLE = "filings_sections"

# NV-Embed-v2 produces 4096-dimensional embeddings.
EMBED_DIM = 4096
MODEL_NAME = "nvidia/NV-Embed-v2"

# All report sections present in the schema (each is a List(String)).
SECTIONS = [
    "section_1", "section_1A", "section_1B", "section_2", "section_3",
    "section_4", "section_5", "section_6", "section_7", "section_7A",
    "section_8", "section_9", "section_9A", "section_9B", "section_10",
    "section_11", "section_12", "section_13", "section_14", "section_15",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed SEC filing sections with NV-Embed-v2 into LanceDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--parquet", default=DEFAULT_PARQUET,
        help="Path to the input parquet file.",
    )
    p.add_argument(
        "--lancedb", default=DEFAULT_LANCEDB,
        help="Directory for the LanceDB database (created if missing).",
    )
    p.add_argument(
        "--table", default=DEFAULT_TABLE,
        help="Name of the LanceDB table to write to.",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Encoding batch size. Raise it if you have spare VRAM.",
    )
    p.add_argument(
        "--flush-size", type=int, default=4096,
        help="How many chunks to buffer before encoding + writing a batch. "
             "Keeps memory bounded on large datasets.",
    )
    p.add_argument(
        "--max-seq-length", type=int, default=32768,
        help="Max sequence length for the model tokenizer.",
    )
    p.add_argument(
        "--min-chars", type=int, default=1,
        help="Skip chunks shorter than this many characters (after strip).",
    )
    p.add_argument(
        "--limit-rows", type=int, default=None,
        help="Optional: only process the first N rows of the parquet "
             "(useful for a quick test run).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (NV-Embed-v2 via sentence-transformers)
# ---------------------------------------------------------------------------
def load_model(max_seq_length: int):
    import torch
    from sentence_transformers import SentenceTransformer

    # 1. Hardware verification (hard-require CUDA, matching the reference usage).
    assert torch.cuda.is_available(), "CUDA is not available! Check your drivers."
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 2. Load the model in bfloat16 to save memory and use Tensor Cores.
    print(f"\nLoading {MODEL_NAME} (7B parameters)...")
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    # NV-Embed-v2 specific settings.
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
def lancedb_schema() -> pa.Schema:
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
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
# Row -> chunk extraction
# ---------------------------------------------------------------------------
def _safe(d, *keys, default=None):
    """Nested dict/struct getter that tolerates missing/None values."""
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur


def iter_chunks(row: dict, min_chars: int):
    """
    Yield (text, metadata_dict) for every text chunk in every section of every
    filing in a single parquet row. The vector is filled in later.
    """
    cik = row.get("cik")
    name = row.get("name")
    tickers = row.get("tickers") or []

    for filing in (row.get("filings") or []):
        if filing is None:
            continue
        form = filing.get("form")
        filing_date = filing.get("filingDate")
        report_date = filing.get("reportDate")
        labels = filing.get("labels") or {}
        returns = filing.get("returns") or {}
        report = filing.get("report") or {}

        meta_common = {
            "cik": cik,
            "name": name,
            "tickers": list(tickers),
            "form": form,
            "filingDate": filing_date,
            "reportDate": report_date,
            "label_1d": labels.get("1d"),
            "label_5d": labels.get("5d"),
            "label_30d": labels.get("30d"),
            "ret_1d": _safe(returns, "1d", "ret"),
            "ret_5d": _safe(returns, "5d", "ret"),
            "ret_30d": _safe(returns, "30d", "ret"),
        }

        for section in SECTIONS:
            chunks = report.get(section) or []
            for idx, text in enumerate(chunks):
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
def flush_buffer(model, table, batch_size: int, texts: list[str], metas: list[dict]) -> int:
    if not texts:
        return 0

    embeddings = model.encode(
        add_eos(model, texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
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

    if not os.path.exists(args.parquet):
        print(f"ERROR: parquet file not found: {args.parquet}", file=sys.stderr)
        print("Generate the dataset first, then re-run this script.", file=sys.stderr)
        return 1

    # Connect / create the LanceDB database (directory created automatically).
    import lancedb
    os.makedirs(args.lancedb, exist_ok=True)
    print(f"Connecting to LanceDB at: {args.lancedb}")
    db = lancedb.connect(args.lancedb)
    table = db.create_table(args.table, schema=lancedb_schema(), mode="overwrite")
    print(f"Created table '{args.table}' (overwrite mode).")

    # Load the embedding model (asserts CUDA).
    model = load_model(args.max_seq_length)

    # Read parquet with Polars.
    print(f"\nReading parquet: {args.parquet}")
    df = pl.read_parquet(args.parquet)
    if args.limit_rows is not None:
        df = df.head(args.limit_rows)
    print(f"Loaded {df.height} company rows.")

    # Stream rows -> chunks -> buffered encode -> LanceDB.
    text_buf: list[str] = []
    meta_buf: list[dict] = []
    total_written = 0
    start = time.time()

    for row in df.iter_rows(named=True):
        for text, meta in iter_chunks(row, args.min_chars):
            text_buf.append(text)
            meta_buf.append(meta)

            if len(text_buf) >= args.flush_size:
                total_written += flush_buffer(
                    model, table, args.batch_size, text_buf, meta_buf
                )
                text_buf, meta_buf = [], []
                print(f"  ... {total_written:,} chunks written so far")

    # Final flush.
    total_written += flush_buffer(model, table, args.batch_size, text_buf, meta_buf)

    elapsed = time.time() - start
    print(f"\nDone. Wrote {total_written:,} embedded chunks in {elapsed:.1f}s.")
    print(f"Table '{args.table}' now has {table.count_rows():,} rows "
          f"({EMBED_DIM}-dim vectors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
