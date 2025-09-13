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
    # 使用示例
    for element in ["H", "C"]:
        source_lmdb_path = f"/mnt/fs_mol/yongqi/ckps/NMRexp/NMRexp_10to24_1_0811_{element}_max_chiral_1_max_atoms_70_ele_C_H_O_N_S_P_F_Cl_filtered.lmdb"
        train_lmdb_path = f"data/NMRexp/{element}/train.lmdb"
        valid_lmdb_path = f"data/NMRexp/{element}/valid.lmdb"
        
        # 按4:1比例切分（训练集80%，测试集20%）
        split_lmdb(source_lmdb_path, train_lmdb_path, valid_lmdb_path, train_ratio=0.8, seed=42)
