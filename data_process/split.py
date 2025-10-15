from unicore.data import LMDBDataset
from utils import write_lmdb
import numpy as np
import os

def split_lmdb(source_path, train_path, valid_path, train_ratio=0.8, seed=42):
    alldata = LMDBDataset(source_path)
    np.random.seed(seed)
    train_ids = np.random.choice(len(alldata), int(len(alldata) * train_ratio), replace=False)
    valid_ids = np.setdiff1d(np.arange(len(alldata)), train_ids)
    
    print(len(alldata), len(train_ids), len(valid_ids))
    
    train_data = [alldata[i] for i in train_ids]
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    write_lmdb(train_data, train_path)
    
    valid_data = [alldata[i] for i in valid_ids]
    os.makedirs(os.path.dirname(valid_path), exist_ok=True)
    write_lmdb(valid_data, valid_path)
    
if __name__ == "__main__":
    # for element in ["H", "C"]:
    for element in ["F", "P", "Si", "B"]:
        # source_lmdb_path = f"/mnt/fs_mol/yongqi/ckps/NMRexp/NMRexp_10to24_1_0811_{element}_max_chiral_1_max_atoms_70_ele_C_H_O_N_S_P_F_Cl_filtered.lmdb"
        source_lmdb_path = f"/mnt/fs_mol/yongqi/clean-code/NMRNet++/raw_data/NMRexp_10to24_1_0811_{element}_max_chiral_1_max_atoms_512_ele_All_filtered.lmdb"
        train_lmdb_path = f"data/NMRexp/{element}/train.lmdb"
        valid_lmdb_path = f"data/NMRexp/{element}/valid.lmdb"
        
        split_lmdb(source_lmdb_path, train_lmdb_path, valid_lmdb_path, train_ratio=0.8, seed=42)

# (unicore) root@di-20250222174817-qm2t6:/mnt/fs_mol/yongqi/clean-code/NMRNet++# python split.py 
# 126961 101568 25393
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/F/train.lmdb
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 101568/101568 [00:10<00:00, 9254.17it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/F/train.lmdb
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/F/valid.lmdb
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25393/25393 [00:02<00:00, 9325.56it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/F/valid.lmdb
# 26980 21584 5396
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/P/train.lmdb
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21584/21584 [00:02<00:00, 9361.41it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/P/train.lmdb
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/P/valid.lmdb
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5396/5396 [00:00<00:00, 9212.60it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/P/valid.lmdb
# 1785 1428 357
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/Si/train.lmdb
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [00:00<00:00, 9071.16it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/Si/train.lmdb
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/Si/valid.lmdb
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [00:00<00:00, 11383.01it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/Si/valid.lmdb
# 12902 10321 2581
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/B/train.lmdb
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10321/10321 [00:01<00:00, 9021.31it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/B/train.lmdb
# Remove existing lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/B/valid.lmdb
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2581/2581 [00:00<00:00, 8944.99it/s]
# Write to lmdb: /mnt/fs_mol/yongqi/clean-code/NMRNet++/data/NMRexp/B/valid.lmdb