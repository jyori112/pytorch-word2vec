import numpy as np
from scipy.stats import pearsonr
import argparse
import data
import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('emb', type=str)
    parser.add_argument('wordsim')

    args = parser.parse_args()

    logger.info('Load embedding')
    with open(args.emb) as f:
        word2id = {}

        n_vocab, dim = map(int, f.readline().split())
        emb = np.empty((n_vocab, dim))

        for wid, line in enumerate(f):
            word, vec_str = line.split(' ', 1)
            emb[wid] = np.fromstring(vec_str, sep=' ')
            word2id[word] = wid

    logger.info('Load wordsim')
    wordsim = data.load_wordsim(args.wordsim, word2id)

    logger.info('Evaluate')
    models = []
    golds = []
    for word1, word2, sim in wordsim:
        vec1 = emb[word1]
        vec2 = emb[word2]

        models.append(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        golds.append(sim)

    pearson = pearsonr(golds, models)[0]
    logger.info('pearson={:.3f}'.format(pearson))

if __name__ == '__main__':
    main()

