import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model, data
import argparse
import logging, sys

logger = logging.getLogger(__name__)
logger.setLevel(10)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('wordsim', type=str)
    parser.add_argument('--output','-o', type=str)
    parser.add_argument('--dim', '-d', type=int, default=100)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batchsize', '-b', type=int, default=1024)
    parser.add_argument('--window', '-w', type=int, default=5)
    parser.add_argument('--negative', '-n', type=int, default=5)

    args = parser.parse_args()

    logger.info('Load Data')
    iterator = data.DataIterator(args.input, 
            n_epoch=args.epoch, batchsize=args.batchsize,
            window=args.window, negative=args.negative)

    logger.info('Load Wordsim')
    wordsim = data.load_wordsim(args.wordsim, iterator.word2id)

    logger.info('Create Model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model.CBoW(iterator.n_vocab, args.dim)
    m.to(device)

    opt = optim.Adam(m.parameters())

    logger.info('Start Training')
    losses = []
    for i, (center, context, negative) in enumerate(iterator):
        center.to(device)
        context.to(device)
        negative.to(device)

        loss = m.cbow(center, context, negative)
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(float(loss))

        if i % 1000 == 0:
            logger.info('Step={}; loss={:.3f}; pearson={:.3f}'.format(
                i, np.mean(losses), m.eval_sim(wordsim)))
            losses = []
        
    m.save_w2v(args.output)


if __name__ == '__main__':
    main()

