#!/usr/bin/env python3
"""
Build paired intersection subsets for counterfactual analysis.
1. Identify SMILES present in BOTH solvent_a and solvent_b.
2. Select samples where original solvent was solvent_a (Source A).
3. Generate 3 versions of each sample:
   - Label A (Original/Forced A)
   - Label B (Counterfactual)
   - Label Base (Blank)
4. Write to 3 parallel LMDBs.

This script can read from:
A) A set of pre-generated "injected" LMDBs (via --source-root)
B) The ORIGINAL valid.lmdb (via --original-lmdb), performing injection on-the-fly.
"""
import argparse
import json
import os
import sys
import lmdb
import shutil
import pickle
try:
    from unicore.data import LMDBDataset
except ImportError:
    LMDBDataset = None

def open_env(path):
    # Handle both directory (with data.mdb) and file paths
    if os.path.isdir(path):
        return lmdb.open(path, readonly=True, lock=False, subdir=True, max_readers=256)
    return lmdb.open(path, readonly=True, lock=False, subdir=False, max_readers=256)

def inject_solvent(sample, solvent_label):
    """
    Inject solvent label into sample.
    Modifies sample in-place.
    """
    # Strategy: Update 'nmr_solvent' if it exists, or add it.
    # Also check 'extra_info' just in case, but prioritize top-level if used by model.
    
    # Based on typical Unimol/UniNMR data structure
    sample['nmr_solvent'] = solvent_label
    
    # If extra_info exists and has solvent, update it too to be consistent
    if 'extra_info' in sample and isinstance(sample['extra_info'], dict):
        if 'nmr_solvent' in sample['extra_info']:
            sample['extra_info']['nmr_solvent'] = solvent_label
            
    return sample

def remove_solvent(sample):
    """
    Remove solvent label for baseline.
    """
    sample.pop('nmr_solvent', None)
    if 'extra_info' in sample and isinstance(sample['extra_info'], dict):
        sample['extra_info'].pop('nmr_solvent', None)
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', required=True, help='Path to subset_triclass_mapping.json')
    
    # Source options (Mutually exclusive-ish, but we handle logic below)
    parser.add_argument('--source-root', help='Dir containing CDCl3, DMSO-d6, blank subdirs (Pre-injected)')
    parser.add_argument('--original-lmdb', help='Path to ORIGINAL valid.lmdb (Inject on-the-fly)')
    
    parser.add_argument('--solvent-a', required=True, help='First solvent (Source), e.g. CDCl3')
    parser.add_argument('--solvent-b', required=True, help='Second solvent (Intersection target), e.g. DMSO-d6')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--dict-path', default='', help='Path to dict.txt to copy')
    args = parser.parse_args()

    if not args.source_root and not args.original_lmdb:
        print("Error: Must provide either --source-root or --original-lmdb")
        sys.exit(1)

    print(f"Loading mapping from {args.mapping}...")
    with open(args.mapping) as f:
        mapping_data = json.load(f)
        if isinstance(mapping_data, dict) and 'mapping' in mapping_data:
            mapping = mapping_data['mapping']
        else:
            mapping = mapping_data

    # Group by SMILES
    smi_to_cats = {}
    smi_to_indices = {} # smi -> {cat: index}

    # Detect format: Conflicts JSON (nested) vs Flat Mapping
    is_conflicts_format = False
    if len(mapping) > 0 and 'indices' in mapping[0] and 'smiles' in mapping[0]:
        is_conflicts_format = True
        print("Detected conflicts file format (e.g. from rebuild_conflicts_aligned.py).")
        print("Assuming indices in this file are valid for the --original-lmdb.")

    if is_conflicts_format:
        for item in mapping:
            smi = item.get('smiles')
            
            # Try to get actual LMDB keys first, fall back to ordinal indices
            # Structure: lmdb_keys -> valid -> solvent -> [list of ints]
            # Structure: indices -> valid -> solvent -> [list of ints]
            
            source_dict = None
            # Check for lmdb_keys first (actual keys)
            # NOTE: If we are using LMDBDataset, we prefer ordinal indices (from 'indices')
            # because LMDBDataset abstracts away the keys.
            # Only if we are forced to use raw LMDB (no unicore) do we care about lmdb_keys.
            
            source_dict = None
            use_lmdb_keys = False
            
            # If we have LMDBDataset, we prefer 'indices' (ordinal)
            if LMDBDataset is not None:
                 if 'indices' in item and isinstance(item['indices'], dict) and 'valid' in item['indices']:
                    source_dict = item['indices']['valid']
            
            # If no source_dict yet (or no LMDBDataset), try lmdb_keys
            if source_dict is None and 'lmdb_keys' in item and isinstance(item['lmdb_keys'], dict) and 'valid' in item['lmdb_keys']:
                source_dict = item['lmdb_keys']['valid']
                use_lmdb_keys = True
            
            # Fallback to indices if still nothing
            if source_dict is None and 'indices' in item and isinstance(item['indices'], dict) and 'valid' in item['indices']:
                source_dict = item['indices']['valid']
            
            if not smi or not source_dict: continue
            
            for cat, idx_list in source_dict.items():
                if not idx_list: continue
                # We take the first index/key for this solvent
                idx = idx_list[0]
                
                if smi not in smi_to_cats:
                    smi_to_cats[smi] = set()
                    smi_to_indices[smi] = {}
                
                smi_to_cats[smi].add(cat)
                smi_to_indices[smi][cat] = idx
    else:
        # Flat mapping format
        # Detect index key if needed
        first_item = mapping[0]
        use_source_index = False
        if args.original_lmdb:
            if 'source_index' in first_item:
                use_source_index = True
                print("Using 'source_index' to read from Original LMDB.")
            elif 'orig_index' in first_item:
                # Some versions use orig_index
                for item in mapping: item['source_index'] = item['orig_index']
                use_source_index = True
                print("Using 'orig_index' (as source_index) to read from Original LMDB.")
            else:
                print("Warning: --original-lmdb specified but no 'source_index' found in mapping. Using 'new_index' (might be wrong if mapping is from a subset).")
        
        for item in mapping:
            smi = item.get('smiles')
            cat = item.get('category')
            
            if use_source_index:
                idx = item.get('source_index')
            else:
                idx = item.get('new_index')
            
            if smi is None or cat is None or idx is None:
                continue
                
            if smi not in smi_to_cats:
                smi_to_cats[smi] = set()
                smi_to_indices[smi] = {}
            smi_to_cats[smi].add(cat)
            smi_to_indices[smi][cat] = idx

    # Find intersection
    KNOWN_SOLVENTS = {'CDCl3', 'DMSO-d6'}

    def check_has_solvent(cats, target_name):
        if target_name == 'not_known':
            # Any solvent NOT in the known list counts as 'not_known'
            return any(c not in KNOWN_SOLVENTS for c in cats)
        return target_name in cats

    def get_source_index(smi, target_name):
        # If specific solvent
        if target_name != 'not_known':
            return smi_to_indices[smi].get(target_name)
        
        # If 'not_known', pick the first available one that isn't known
        for c, idx in smi_to_indices[smi].items():
            if c not in KNOWN_SOLVENTS:
                return idx
        return None

    target_indices = []
    for smi, cats in smi_to_cats.items():
        if check_has_solvent(cats, args.solvent_a) and check_has_solvent(cats, args.solvent_b):
            # Select the index corresponding to solvent_a (Source A)
            idx = get_source_index(smi, args.solvent_a)
            if idx is not None:
                target_indices.append(idx)

    target_indices.sort()
    print(f"Found {len(target_indices)} intersection samples (Source={args.solvent_a}, Intersection={args.solvent_b})")

    if len(target_indices) == 0:
        print("No intersection found. Exiting.")
        sys.exit(0)

    # Setup Source
    src_envs = {}
    src_dataset = None
    
    if args.original_lmdb:
        print(f"Reading from Original LMDB: {args.original_lmdb}")
        if not os.path.exists(args.original_lmdb):
             print(f"Error: {args.original_lmdb} not found.")
             sys.exit(1)
        
        # Try to use LMDBDataset for reliable index-based access
        if LMDBDataset is not None:
            try:
                src_dataset = LMDBDataset(args.original_lmdb)
                print(f"Using unicore.data.LMDBDataset (len={len(src_dataset)})")
            except Exception as e:
                print(f"Failed to init LMDBDataset: {e}. Falling back to raw LMDB.")
                src_dataset = None
        
        if src_dataset is None:
            # We only need one env if not using dataset
            env = open_env(args.original_lmdb)
            src_envs = {'original': env}
    else:
        # Pre-injected mode
        src_paths = {
            'label_a': os.path.join(args.source_root, args.solvent_a, 'valid.lmdb'),
            'label_b': os.path.join(args.source_root, args.solvent_b, 'valid.lmdb'),
            'label_base': os.path.join(args.source_root, 'blank', 'valid.lmdb')
        }
        for k, p in src_paths.items():
            if not os.path.exists(p):
                print(f"Error: Source LMDB not found: {p}")
                sys.exit(1)
        src_envs = {k: open_env(p) for k, p in src_paths.items()}

    # Create output envs
    os.makedirs(args.out_dir, exist_ok=True)
    out_envs = {}
    for key in ['label_a', 'label_b', 'label_base']:
        sub = os.path.join(args.out_dir, key)
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, 'valid.lmdb')
        if os.path.exists(path): os.remove(path)
        out_envs[key] = lmdb.open(path, map_size=1024**3, subdir=False, lock=False, max_dbs=1)
        if args.dict_path and os.path.exists(args.dict_path):
            shutil.copy(args.dict_path, os.path.join(sub, 'dict.txt'))

    # Process
    print("Extracting and processing samples...")
    
    count = 0
    missing_count = 0
    
    # Transactions
    txns_src = {k: env.begin() for k, env in src_envs.items()}
    txns_out = {k: env.begin(write=True) for k, env in out_envs.items()}
    
    try:
        for idx in target_indices:
            k_in = str(idx).encode('ascii')
            k_out = str(count).encode('ascii')
            
            if args.original_lmdb:
                # On-the-fly injection
                sample = None
                
                if src_dataset is not None:
                    # Use dataset by ordinal index
                    try:
                        sample = src_dataset[idx]
                    except Exception:
                        sample = None
                else:
                    # Use raw LMDB by key (assuming key == index)
                    # Note: This fails if keys are not 0..N or if we used lmdb_keys logic incorrectly
                    # But if src_dataset failed, we have no choice but to try
                    raw = txns_src['original'].get(k_in)
                    if raw:
                        sample = pickle.loads(raw)
                
                if sample is not None:
                    # 1. Label A
                    s_a = inject_solvent(sample.copy(), args.solvent_a)
                    txns_out['label_a'].put(k_out, pickle.dumps(s_a))
                    
                    # 2. Label B
                    s_b = inject_solvent(sample.copy(), args.solvent_b)
                    txns_out['label_b'].put(k_out, pickle.dumps(s_b))
                    
                    # 3. Label Base
                    s_base = remove_solvent(sample.copy())
                    txns_out['label_base'].put(k_out, pickle.dumps(s_base))
                    
                    count += 1
                else:
                    missing_count += 1
            else:
                # Pre-injected copy
                val_a = txns_src['label_a'].get(k_in)
                val_b = txns_src['label_b'].get(k_in)
                val_base = txns_src['label_base'].get(k_in)
                
                if val_a and val_b and val_base:
                    txns_out['label_a'].put(k_out, val_a)
                    txns_out['label_b'].put(k_out, val_b)
                    txns_out['label_base'].put(k_out, val_base)
                    count += 1
                else:
                    missing_count += 1

        for txn in txns_out.values():
            txn.commit()
            
    finally:
        for env in src_envs.values(): env.close()
        for env in out_envs.values(): env.close()
    
    print(f"Done. Wrote {count} samples to {args.out_dir}")
    if missing_count > 0:
        print(f"Skipped {missing_count} samples due to missing keys.")

if __name__ == '__main__':
    main()
