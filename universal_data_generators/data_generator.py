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


class DataGeneratorW2V(object):
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
        self.total_data_size = len(data)

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(idxs)

        X, Y, T = [], [], []
        for i, idx in enumerate(idxs):
            text, label_id = self.data[idx]

            text_ids = [self.vocabulary.get(c, self.vocabulary.get('unk'))
                                                         for c in text[:self.data_args.max_seq_length]]

            if isinstance(label_id, str):
                label_id = self.intent_labels.index(label_id)

            X.append(text_ids)
            Y.append(label_id)
            T.append(text)

            if len(X) == self.batch_size or i == len(self.data) - 1:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)

                yield X, Y, T

                X, Y, T = [], [], []


class DataGeneratorW2V_VAE(object):
    def __init__(self, data, batch_size, data_args, vocab_wv, vocab_lm, intent_labels, shuffle=False):
        """

        :param data:  Subset  用indices选取

        """
        self.batch_size = batch_size
        self.intent_labels = intent_labels
        self.vocab_wv = vocab_wv
        self.vocab_lm = vocab_lm
        self.data_args = data_args
        self.data = data
        self.shuffle = shuffle
        self.steps = len(data) // batch_size
        self.total_data_size = len(data)
        self.vocab_lm_size = len(self.vocab_lm)
        self.vocab_wv_size = len(self.vocab_wv)

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(idxs)

        X, Y, V, T = [], [], [], []
        for i, idx in enumerate(idxs):
            text, label_id = self.data[idx]

            text_ids = [self.vocab_wv.get(c, self.vocab_wv.get('unk'))
                                                         for c in text[:self.data_args.max_seq_length]]
            text_vocab_ids = [1 if _ in text else 0 for _ in self.vocab_lm]

            if isinstance(label_id, str):
                label_id = self.intent_labels.index(label_id)

            X.append(text_ids)
            Y.append(label_id)
            V.append(text_vocab_ids)
            T.append(text)

            if len(X) == self.batch_size or i == len(self.data) - 1:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                V = torch.tensor(V, dtype=torch.float)  # [b, V_size]
                Y = torch.tensor(Y, dtype=torch.long)

                yield X, Y, V, T

                X, Y, V, T = [], [], [], []