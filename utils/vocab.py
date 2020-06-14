import collections

import numpy
from gensim.models import KeyedVectors

from configuration.config import bert_vocab_path, tencent_w2v_path


def load_vocab(vocab_file=bert_vocab_path):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def load_vocab_w2v():
    c2v_path = tencent_w2v_path / "w2v_char_py3_baidu_1112"
    char_vectors = KeyedVectors.load(str(c2v_path))
    char2vec_vocab = {w: idx + 2 for idx, w in enumerate(char_vectors.vocab)}
    char2vec_vocab['pad'] = 0
    char2vec_vocab['unk'] = 1

    id2embeddings = numpy.zeros((len(char2vec_vocab) + 2, 150))
    for idx, w in enumerate(char_vectors.vocab):
        id2embeddings[idx + 2] = char_vectors.word_vec(w)

    return char2vec_vocab, id2embeddings
