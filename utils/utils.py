import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_all_seed(seed)


def seq_padding(seqs):
    max_len = max(map(len, seqs))
    seq = [seq + [0] * (max_len - len(seq)) for seq in seqs]
    return seq