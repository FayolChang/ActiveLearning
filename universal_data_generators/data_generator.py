import torch
import numpy as np
from torch.utils.data import Dataset

from utils.utils import seq_padding


class DataGenerator_raw(object):
    def __init__(self, data, training_args, data_args, vocabulary, intent_labels, shuffle=False):
        self.training_args = training_args
        self.intent_labels = intent_labels
        self.vocabulary = vocabulary
        self.data_args = data_args
        self.data = data
        self.shuffle = shuffle
        self.steps = len(data) // training_args.batch_size

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data)
        X, Y, M, T = [], [], [], []
        for i, d in enumerate(self.data):
            text, label = d
            text = text.lower()

            text_ids = [self.vocabulary.get('[CLS]')] + [self.vocabulary.get(c, self.vocabulary.get('[UNK]'))
                                                         for c in text[:self.data_args.max_seq_length]]
            att_mask = [1] * len(text_ids)

            label_id = self.intent_labels.index(label)

            X.append(text_ids)
            M.append(att_mask)
            Y.append(label_id)
            T.append(text)

            if len(X) == self.training_args.batch_size or i == len(self.data) - 1:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)
                M = torch.tensor(seq_padding(M), dtype=torch.long)

                yield X, Y, M, T

                X, Y, M, T = [], [], [], []


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
    def __init__(self, data, batch_size, data_args, training_args, vocab_wv, vocab_lm, intent_labels, shuffle=False):
        """

        :param data:  Subset  用indices选取

        """
        self.batch_size = batch_size
        self.intent_labels = intent_labels
        self.vocab_wv = vocab_wv
        self.vocab_lm = vocab_lm
        self.data_args = data_args
        self.training_args = training_args
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

        X, Y, V, M, T = [], [], [], [], []
        for i, idx in enumerate(idxs):
            text, label_id = self.data[idx]

            text_ids = [self.vocab_wv.get(c, self.vocab_wv.get('unk'))
                                                         for c in text[:self.data_args.max_seq_length]]
            text_vocab_ids = [self.vocab_lm.get('[CLS]')] + [self.vocab_lm.get(c, self.vocab_lm.get('[UNK]'))
                                                             for c in text[:self.training_args.rec_max_length-1]]

            att_mask = [1] * len(text_vocab_ids)

            if isinstance(label_id, str):
                label_id = self.intent_labels.index(label_id)

            X.append(text_ids)
            Y.append(label_id)
            V.append(text_vocab_ids)
            M.append(att_mask)
            T.append(text)

            if len(X) == self.batch_size or i == len(self.data) - 1:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                V = torch.tensor(seq_padding(V, self.training_args.rec_max_length), dtype=torch.long)
                M = torch.tensor(seq_padding(M, self.training_args.rec_max_length), dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)

                yield X, Y, V, M, T

                X, Y, V, M, T = [], [], [], [], []