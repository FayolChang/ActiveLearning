import collections

import numpy as np
import torch
from torch.utils import data


class ActiveLearningData(object):
    def __init__(self, dataset: data.Dataset):
        self.dataset = dataset
        self.total_size = len(dataset)

        self.train_mask = np.full((self.total_size, ), False)
        self.pool_mask = np.full((self.total_size, ), True)

        self.train_data = data.Subset(dataset, None)
        self.pool_data = data.Subset(dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.train_data.indices = np.nonzero(self.train_mask)[0]
        self.pool_data.indices = np.nonzero(self.pool_mask)[0]

    def get_pool_indices(self, indices):
        return self.pool_data.indices[indices]

    def acquire(self, pool_indices):
        indices = self.get_pool_indices(pool_indices)

        self.train_mask[indices] = True
        self.pool_mask[indices] = False

        self._update_indices()

    def remove_from_pool_data(self, indices):
        indices = self.get_pool_indices(indices)

        self.pool_mask[indices] = False
        self._update_indices()

    def get_random_pool_indices(self, size):
        assert size <= len(self.pool_data)
        indices = torch.randperm(len(self.pool_data))[:size]
        return indices

    def extract_data_from_pool(self, size):
        """

        :param size: 需删除的数据大小
        :return: 更新后的dataset
        """
        self.extract_data_from_pool_from_indices(self.get_random_pool_indices(size))

    def extract_data_from_pool_from_indices(self, pool_indices):
        data_indices = self.get_pool_indices(pool_indices)
        self.remove_from_pool_data(data_indices)
        return data.Subset(self.dataset, data_indices)


def get_balanced_sample_indices(target_classes, num_classes, n_per_class=2):
    permed_indices = torch.randperm(len(target_classes))

    if n_per_class == 0:
        return []

    num_samples_by_class = collections.defaultdict(int)
    initial_samples = []

    for i in range(len(permed_indices)):
        permed_index = int(permed_indices[i])
        index, target = permed_index, int(target_classes[permed_index])

        num_target_samples = num_samples_by_class[target]
        if num_target_samples == n_per_class:
            continue

        initial_samples.append(index)
        num_samples_by_class[target] += 1

        if len(initial_samples) == num_classes * n_per_class:
            break

    return initial_samples





