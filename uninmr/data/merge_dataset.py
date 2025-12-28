# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset, EpochShuffleDataset
import numpy as np


class MergedDataset(BaseWrapperDataset):
    def __init__(self, dataset_list, ratio_list, mode):
        super().__init__(dataset_list[0])
        self.dataset_list = dataset_list
        self.ratio_list = ratio_list
        assert mode in ['truncate', 'append', 'repeat'], f"Invalid mode: {mode}. Choose 'truncate' or 'append' or 'repeat'."
        self.mode = mode
        self.merged_indices = self._generate_indices()
    
    def _generate_indices(self):
        indices = []
        lens = [len(dataset) for dataset in self.dataset_list]
        n = len(self.dataset_list)
        counts = [0] * n
        values = [0] * n
        flags = [True] * n

        while True:
            available_datasets = [i for i in range(n)] if self.mode == 'repeat' else [i for i in range(n) if flags[i]]
            if self.mode == 'truncate' and len(available_datasets) < n:
                break
            elif self.mode == 'append' and sum(self.ratio_list[i] for i in range(n) if flags[i]) == 0:
                break
            elif self.mode == 'repeat' and sum(self.ratio_list[i] for i in range(n) if flags[i]) == 0:
                break
            
            available_datasets.sort(key=lambda i: values[i] + self.ratio_list[i])

            for i in available_datasets:
                values[i] += self.ratio_list[i]
                if values[i] >= 1:
                    indices.append((i, counts[i]))
                    counts[i] += 1
                    values[i] -= 1
            
            if self.mode == 'truncate' or self.mode == 'append':
                for i in range(n):
                    if counts[i] >= lens[i]:
                        flags[i] = False
            elif self.mode == 'repeat':
                for i in range(n):
                    if counts[i] >= lens[i]:
                        counts[i] = 0
                        flags[i] = False

        if self.mode == 'append':
            for i in range(n):
                if self.ratio_list[i] == 0:
                    indices.extend([(i, j) for j in range(lens[i])])
        
        return indices

    def __len__(self):
        return len(self.merged_indices)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        i, j = self.merged_indices[idx]
        dataset = self.dataset_list[i]
        if isinstance(dataset, EpochShuffleDataset):
            return dataset[dataset.sort_order[j]]
        else:
            return dataset[j]
    
    def set_epoch(self, epoch):
        """Set the epoch for all datasets that support set_epoch."""
        for dataset in self.dataset_list:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
                
    def ordered_indices(self):
        return np.arange(len(self.merged_indices), dtype=np.int64)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
    
