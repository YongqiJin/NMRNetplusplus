#!/usr/bin/env python3
"""Split a source valid.lmdb into three solvent-specific LMDBs."""

from __future__ import annotations

import argparse
import os
import shutil
import pickle
from typing import Any, Dict

import lmdb

CATEGORIES = ["CDCl3", "DMSO-d6", "OTHER"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Split valid.lmdb into subsets by solvent (CDCl3/DMSO-d6/OTHER)"
    )
    ap.add_argument("--valid-path", required=True, help="Source valid.lmdb path")
    ap.add_argument("--out-root", required=True, help="Output root directory")
    ap.add_argument("--dict-path", default="", help="Path to dict.txt to copy")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    ap.add_argument(
        "--map-size", type=int, default=1 << 40,
        help="LMDB map_size (default 1TB)",
    )
    return ap.parse_args()


def ensure_valid_path(path: str) -> str:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "data.mdb")):
        return path
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Invalid LMDB path: {path}")


def normalize_category(name: object) -> str:
    if name is None:
        return "OTHER"
    text = str(name).strip()
    if not text:
        return "OTHER"
    upper = text.upper()
    if upper == "CDCL3":
        return "CDCl3"
    if upper == "DMSO-D6":
        return "DMSO-d6"
    return "OTHER"


def extract_solvent(obj: Any) -> Any:
    """Recursively extract nmr_solvent field."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "nmr_solvent" in obj:
            return obj["nmr_solvent"]
        for key in ("extra_info", "info", "metadata"):
            if key in obj:
                val = extract_solvent(obj[key])
                if val is not None:
                    return val
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            val = extract_solvent(item)
            if val is not None:
                return val
        return None
    return None


def open_writer(root: str, cat: str, overwrite: bool, map_size: int) -> lmdb.Environment:
    out_dir = os.path.join(root, cat)
    os.makedirs(out_dir, exist_ok=True)
    lmdb_path = os.path.join(out_dir, "valid.lmdb")
    if os.path.exists(lmdb_path):
        if overwrite:
            if os.path.isdir(lmdb_path):
                raise IsADirectoryError(f"Cannot overwrite directory: {lmdb_path}")
            os.remove(lmdb_path)
        else:
            raise FileExistsError(f"Output exists: {lmdb_path} (use --overwrite)")
    return lmdb.open(lmdb_path, map_size=map_size, subdir=False, lock=False, max_dbs=1)


def copy_dict(dict_path: str, root: str):
    if not dict_path or not os.path.isfile(dict_path):
        return
    for cat in CATEGORIES:
        dst = os.path.join(root, cat, os.path.basename(dict_path))
        try:
            shutil.copy2(dict_path, dst)
        except Exception as err:
            print(f"[Warn] Failed to copy dict.txt to {dst}: {err}")


def main() -> None:
    args = parse_args()

    src_path = ensure_valid_path(args.valid_path)
    is_dir = os.path.isdir(src_path)
    src_env = lmdb.open(
        src_path,
        readonly=True,
        lock=False,
        readahead=True,
        subdir=is_dir,
        max_readers=2048,
    )

    writers: Dict[str, lmdb.Environment] = {}
    txns: Dict[str, lmdb.Transaction] = {}
    counts: Dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for cat in CATEGORIES:
        writers[cat] = open_writer(args.out_root, cat, args.overwrite, args.map_size)
        txns[cat] = writers[cat].begin(write=True)

    processed = 0
    skipped_meta = 0
    with src_env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b"__meta__":
                skipped_meta += 1
                continue
            raw = bytes(value)
            sample = pickle.loads(raw)
            cat = normalize_category(extract_solvent(sample))
            new_key = str(counts[cat]).encode("utf-8")
            txns[cat].put(new_key, raw)
            counts[cat] += 1
            processed += 1

    for cat in CATEGORIES:
        txns[cat].commit()
        writers[cat].sync()
        writers[cat].close()

    src_env.close()
    copy_dict(args.dict_path, args.out_root)

    summary = {
        "source": src_path,
        "total_records": processed,
        "skipped_non_numeric": skipped_meta,
        "counts_by_category": counts,
        "output_root": args.out_root,
    }
    summary_path = os.path.join(args.out_root, "split_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            import json
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"[Info] Summary written to: {summary_path}")
    except Exception as err:
        print(f"[Warn] Failed to write summary: {err}")

    print("[Done] Split completed")
    for cat in CATEGORIES:
        print(f"  {cat}: {counts[cat]} records -> {os.path.join(args.out_root, cat, 'valid.lmdb')}")


if __name__ == "__main__":
    main()
