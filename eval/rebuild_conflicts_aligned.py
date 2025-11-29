#!/usr/bin/env python3
"""
Rebuild conflict file aligned with current valid.lmdb.

Purpose:
  Resolve index mismatch issues by using local ordinal indices (0..N-1) from the current valid.lmdb.

Features:
  1. Uses local ordinal indices.
  2. Optional dbid output.
  3. Solvent standardization.
  4. Filters SMILES with < min_solvents.
  5. Output compatible with build_dual_solvent_subsets.py.
"""
from __future__ import annotations
import os, sys, json, argparse, math, time
from typing import Dict, Any, List

try:
    import lmdb  # type: ignore
except Exception:
    lmdb = None

try:
    from unicore.data import LMDBDataset  # type: ignore
except Exception as e:
    LMDBDataset = None


def parse_args():
    ap = argparse.ArgumentParser(description='Rebuild aligned conflict file')
    ap.add_argument('--data-root', required=True, help='Directory containing valid.lmdb')
    ap.add_argument('--output', default=None, help='Output JSON path')
    ap.add_argument('--smiles-key', default='smiles')
    ap.add_argument('--solvent-key', default='nmr_solvent')
    ap.add_argument('--progress', type=int, default=20000)
    ap.add_argument('--limit', type=int, default=0, help='Debug: limit to N records')
    ap.add_argument('--min-solvents', type=int, default=2, help='Min unique solvents required')
    ap.add_argument('--standardize-solvent', action='store_true', default=True)
    ap.add_argument('--no-standardize-solvent', dest='standardize_solvent', action='store_false')
    ap.add_argument('--include-dbids', action='store_true', help='Include dbids if valid_dbid.pkl exists')
    ap.add_argument('--max-records', type=int, default=0, help='Max records to output')
    ap.add_argument('--sort-mode', choices=['atoms','counts','solvents'], default='counts', help='Sort mode')
    return ap.parse_args()


def standardize(solvent: str) -> str:
    if solvent is None:
        return 'NA'
    s = solvent.strip().lower()
    if s == 'cdcl3':
        return 'CDCl3'
    if s == 'dmso-d6':
        return 'DMSO-d6'
    return solvent.strip()


def main():
    args = parse_args()
    if LMDBDataset is None:
        print('[Error] unicore.data.LMDBDataset not found', file=sys.stderr)
        sys.exit(2)

    valid_path = os.path.join(args.data_root, 'valid.lmdb')
    if os.path.isdir(valid_path):
        subdir = True
    elif os.path.isfile(valid_path):
        subdir = False
    else:
        sys.exit(f'[Error] valid.lmdb not found: {valid_path}')
    print(f'[Debug] valid.lmdb path={valid_path}')

    ds = LMDBDataset(valid_path)
    n = len(ds)
    print(f'[Info] Opened valid.lmdb: samples={n}')

    limit = args.limit if args.limit > 0 else n
    limit = min(limit, n)

    dbids: List[Any] | None = None
    dbid_path = os.path.join(args.data_root, 'valid_dbid.pkl')
    if args.include_dbids and os.path.isfile(dbid_path):
        import pickle
        with open(dbid_path,'rb') as f:
            dbids = pickle.load(f)
        if len(dbids) != n:
            print(f'[Warn] valid_dbid.pkl length mismatch, ignoring', file=sys.stderr)
            dbids = None
        else:
            print('[Info] Loaded dbids')

    state: Dict[str, Dict[str, List[int]]] = {}
    missing_smiles = 0
    missing_solvent = 0

    t0 = time.time()
    for idx in range(limit):
        try:
            sample = ds[idx]
        except Exception as e:
            print(f'[Warn] Failed to read idx={idx}: {e}')
            continue
        smi = sample.get(args.smiles_key)
        if smi is None:
            missing_smiles += 1
            continue
        solv_raw = sample.get(args.solvent_key)
        if solv_raw is None:
            missing_solvent += 1
            solv_raw = 'NA'
        solv = standardize(solv_raw) if args.standardize_solvent else (solv_raw or 'NA')
        bucket = state.setdefault(smi, {})
        bucket.setdefault(solv, []).append(idx)
        if args.progress and (idx+1) % args.progress == 0:
            elapsed = time.time() - t0
            print(f'[Prog] {idx+1}/{limit} ({(idx+1)/limit:.1%}) elapsed={elapsed:.1f}s')

    ord_to_key: List[int] = []
    if lmdb is not None:
        try:
            env = lmdb.open(valid_path, readonly=True, lock=False, subdir=subdir, max_readers=4096)
            with env.begin() as txn:
                cur = txn.cursor()
                for k,_ in cur:
                    if k == b'__meta__': continue
                    try:
                        ki = int(k.decode())
                        ord_to_key.append(ki)
                    except Exception: pass
            env.close()
        except Exception as e:
            print(f'[Warn] Failed to enumerate keys: {e}')
            ord_to_key = []

    records = []
    for smi, solv_map in state.items():
        solvents = list(solv_map.keys())
        if len(solvents) < args.min_solvents:
            continue
        counts = {s: len(solv_map[s]) for s in solvents}
        if args.sort_mode == 'solvents':
            sort_key = (-len(solvents), -sum(counts.values()))
        else:
            sort_key = (-sum(counts.values()), -len(solvents))
        records.append((sort_key, smi, solvents, counts, solv_map))

    records.sort(key=lambda x: x[0])
    out = []
    for i,(sk, smi, solvents, counts, solv_map) in enumerate(records):
        if args.max_records and i >= args.max_records:
            break
        solvents_sorted = sorted(solvents)
        ord_indices = {s: solv_map[s] for s in solvents_sorted}
        
        key_indices = None
        if ord_to_key and len(ord_to_key) == n:
            key_indices = {s: [ord_to_key[i] for i in solv_map[s]] for s in solvents_sorted}

        entry: Dict[str, Any] = {
            'smiles': smi,
            'num_solvents': len(solvents_sorted),
            'solvents': solvents_sorted,
            'counts': counts,
            'indices': {'valid': ord_indices},
        }
        if key_indices is not None:
            entry['lmdb_keys'] = {'valid': key_indices}
        if dbids is not None:
            entry['dbids'] = {'valid': {s: [dbids[j] for j in solv_map[s]] for s in solvents_sorted}}
        out.append(entry)

    stats = {
        'total_samples_scanned': limit,
        'dataset_length': n,
        'unique_smiles': len(state),
        'conflict_smiles': len(out),
        'missing_smiles': missing_smiles,
        'missing_solvent': missing_solvent,
        'min_solvents_threshold': args.min_solvents,
    }

    output = args.output or os.path.join(args.data_root, 'conflicts_valid.json')
    with open(output,'w') as f:
        json.dump(out, f, indent=2)
    
    stats_path = output.replace('.json','_stats.json')
    with open(stats_path,'w') as f:
        json.dump(stats,f,indent=2)
    print(f'[Done] Written to: {output} (records={len(out)})')

if __name__ == '__main__':
    main()
