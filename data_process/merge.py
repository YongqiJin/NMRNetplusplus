import lmdb

from utils import write_lmdb, load_lmdb

def merge_lmdb(input_paths, output_path):
    merged_data = {}
    for path in input_paths:
        data = load_lmdb(path)
        merged_data.update(data)
        print(f"Loaded {len(data)} entries from {path}")
    
    write_lmdb(merged_data, output_path)
    
if __name__ == "__main__":
    import os
    os.makedirs("data/NMRexp/All", exist_ok=True)
    for split in ["train", "valid"]:
        input_lmdb_paths = [
            f"data/NMRexp/{element}/{split}.lmdb" for element in ["F", "P", "Si", "B"]
        ]
        output_lmdb_path = f"data/NMRexp/All/{split}.lmdb"
        merge_lmdb(input_lmdb_paths, output_lmdb_path)

# (unicore) root@di-20250222174817-qm2t6:/mnt/fs_mol/yongqi/clean-code/NMRNet++# python merge.py 
# Loaded 101568 entries from data/NMRexp/F/train.lmdb
# Loaded 21578 entries from data/NMRexp/P/train.lmdb
# Loaded 1428 entries from data/NMRexp/Si/train.lmdb
# Loaded 10318 entries from data/NMRexp/B/train.lmdb
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/All/train.lmdb
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124574/124574 [00:20<00:00, 5972.80it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/All/train.lmdb
# Loaded 25393 entries from data/NMRexp/F/valid.lmdb
# Loaded 5395 entries from data/NMRexp/P/valid.lmdb
# Loaded 357 entries from data/NMRexp/Si/valid.lmdb
# Loaded 2580 entries from data/NMRexp/B/valid.lmdb
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/All/valid.lmdb
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31145/31145 [00:05<00:00, 5736.36it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/All/valid.lmdb