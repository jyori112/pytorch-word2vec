import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)


def load_wordsim(path, word2id, header=True):
    with open(path) as f:
        if header:
            f.readline()

        wordsim = []

        for line in f:
            line = line.strip().lower()

            if not line:
                continue

            word1, word2, sim = line.split('\t')

            if word1 not in word2id or word2 not in word2id:
                continue

            wordsim.append((word2id[word1], word2id[word2], float(sim)))

    return wordsim

class DataIterator:
    def __init__(self, path, n_epoch=20, batchsize=1024, window=5, negative=5):
        self.path = path
        self.n_epoch = n_epoch
        self.batchsize = batchsize
        self.window = window
        self.negative = negative

        with open(path) as f:
            dataset = f.read().lower().split()

        # Count words
        c = Counter(dataset)

        self.word2id = {}
        self.id2count = {}
        wid = 1
        for word, count in c.items():
            if count >= 5:
                self.word2id[word] = wid
                self.id2count[wid] = count
                wid += 1

        # Convert dataset into word ids
        self.dataset = np.array([self.token2id(token) for token in dataset], dtype=np.int64)

        # Create Offset for convinience
        self.offset = np.concatenate((-np.arange(1, window+1)[::-1], np.arange(1, window+1)))

        # Build Negative Tables
        self.neg_table = self.build_neg_table()

    @property
    def n_vocab(self):
        # +1 for UNK token (0)
        return len(self.word2id) + 1

    def token2id(self, token):
        if token in self.word2id:
            return self.word2id[token]
        else:
            return 0

    def build_neg_table(self, power=3/4, table_size=10000000):
        powered_sum = sum(count ** power for wid, count in self.id2count.items())
        table = np.zeros(shape=(table_size, ),dtype=np.int64)

        idx = 0
        accum = 0.0

        for wid, count in self.id2count.items():
            freq = (count**power) / powered_sum
            accum += freq * table_size
            end_idx = int(accum)
            table[idx: int(accum)] = wid
            idx = end_idx

        return table

    def __iter__(self):
        for epoch in range(self.n_epoch):
            logger.info('Epoch: {}'.format(epoch))
            # Shuffle Dataset
            self.order = np.random.permutation(
                            len(self.dataset) - 2 * self.window) + self.window

            for pos in range(0, len(self.dataset), self.batchsize):
                # Get position of center and context in dataset
                center_pos = self.order[pos:pos+self.batchsize]
                context_pos = center_pos[:, None] + self.offset

                # Get actual value of center and context
                center = self.dataset[center_pos]
                context = self.dataset[context_pos]

                # Sample negative tokens
                negative = self.neg_table[np.random.randint(len(self.neg_table), 
                                    size=(center.shape[0], self.negative))]

                yield (torch.tensor(center, dtype=torch.long),
                        torch.tensor(context, dtype=torch.long),
                        torch.tensor(negative, dtype=torch.long))
