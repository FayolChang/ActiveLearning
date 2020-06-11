import torch
import numpy as np
from torch.utils.data import Dataset

from utils.utils import seq_padding


class DataGenerator(object):
    def __init__(self, data, batch_size, data_args, vocabulary, intent_labels, shuffle=False):
        """

        :param data:  Subset  用indices选取

        """
        self.batch_size = batch_size
        self.intent_labels = intent_labels
        self.vocabulary = vocabulary
        self.data_args = data_args
        self.data = data
        self.shuffle = shuffle
        self.steps = len(data) // batch_size

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # if self.shuffle:
        #     np.random.shuffle(self.data)

        X, Y, M, T = [], [], [], []
        for i, (text, label_id) in enumerate(self.data):

            text_ids = [self.vocabulary.get('[CLS]')] + [self.vocabulary.get(c, self.vocabulary.get('[UNK]'))
                                                         for c in text[:self.data_args.max_seq_length]]
            att_mask = [1] * len(text_ids)

            if isinstance(label_id, str):
                label_id = self.intent_labels.index(label_id)

            X.append(text_ids)
            M.append(att_mask)
            Y.append(label_id)
            T.append(text)

            if len(X) == self.batch_size or i == len(self.data) - 1:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)
                M = torch.tensor(seq_padding(M), dtype=torch.long)

                yield X, Y, M, T

                X, Y, M, T = [], [], [], []
