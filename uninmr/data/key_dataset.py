# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
import numpy as np
import torch


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]
    
class SolventDataset(BaseWrapperDataset):
    def __init__(self, dataset, key, solvent_map={'CDCl3':1, 'DMSO-d6':2}):
        self.dataset = dataset
        self.key = key
        self.solvent_map = solvent_map or {}  # 溶剂字符串到整数的映射字典

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        try:
            solvent_str = self.dataset[idx][self.key]
            # 将溶剂字符串映射到整数
            id = self.solvent_map.get(solvent_str, 3)
        except KeyError:
            # 如果数据集中没有溶剂 key，返回默认值
            id = 0
        return torch.tensor(id)

class IndexDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return idx

class ConstantDataset(BaseWrapperDataset):
    def __init__(self, dataset, value):
        self.dataset = dataset
        self.value = value

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.value

class ToTorchDataset(BaseWrapperDataset):
    def __init__(self, dataset, dtype='float32'):
        super().__init__(dataset)
        self.dtype = dtype

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        d = np.array(self.dataset[idx], dtype=self.dtype)
        return torch.from_numpy(d)

class NumericalTransformDataset(BaseWrapperDataset):
    def __init__(self, dataset, ops='log1p'):
        super().__init__(dataset)
        self.dataset = dataset
        self.ops = ops

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.ops == 'log1p':
            d = np.array(self.dataset[idx], dtype='float64')
            d = np.log1p(d).astype('float32')
        else:
            d = np.array(self.dataset[idx])
        return d

class FlattenDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index: int):
        dd = self.dataset[index].copy()
        dd = np.array(dd).reshape(-1,)
        return dd