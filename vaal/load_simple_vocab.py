import collections
import json
from pathlib import Path

from configuration.config import bert_vocab_path, data_dir


def load_simple_vocab(vocab_file=bert_vocab_path, remove_sign=True):
    if remove_sign:
        tmp_vocab = json.load((Path(data_dir) / 'tmp_vocab_0622.json').open())

    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            if remove_sign and (token not in tmp_vocab and token not in ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']):
                continue
            vocab[token] = index
            index += 1
    return vocab

