import collections
import json
import re
from pathlib import Path

from configuration.config import bert_vocab_path, data_dir


def load_vocab(vocab_file=bert_vocab_path, remove_sign=False):
    if remove_sign:
        tmp_vocab = json.load((Path(data_dir) / 'tmp_vocab.json').open())

    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            if remove_sign and ((token not in tmp_vocab or
                                 token.startswith('#') or
                                 'unused' in token or
                                 not re.search(r'[\u4e00-\u9fa5]', token))
                                and token != '[UNK]'):
                continue
            vocab[token] = index
            index += 1
    return vocab

