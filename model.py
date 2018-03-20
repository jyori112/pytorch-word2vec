import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchutils.utils import timer

t = timer.get_instance()

class CBoW(nn.Module):
    def __init__(self, n_vocab, dim=100):
        """
            word2id: {(neg, word): (idx, count)}
        """
        super(CBoW, self).__init__()

        self.dim = dim

        # Make embeddings
        self.center_embed = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.context_embed = nn.Embedding(n_vocab, dim, padding_idx=0)

        # Initialize Embeddings
        self.center_embed.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.context_embed.weight.data.zero_()

    def eval_sim(self, wordsim):
        emb = np.array(self.center_embed.weight.data)
        models = []
        golds = []
        for word1, word2, sim in wordsim:
            vec1 = emb[word1]
            vec2 = emb[word2]

            models.append(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            golds.append(sim)

        return pearsonr(golds, models)[0]

    def cbow(self, center, contexts, negatives):
        # center_emb: (batchsize, dim)
        # context_emb: (batchsize, context_len, dim)
        # negative_emb: (batchsize, negative, dim)
        center_emb = self.center_embed(center)
        context_emb = self.context_embed(contexts)
        negative_emb = self.center_embed(negatives)

        # context_vec: (batchsize, dim)
        context_vec = torch.sum(context_emb, dim=1)

        # emb: (batchsize, negative + 1, dim)
        emb = torch.cat((center_emb.unsqueeze(1), -negative_emb), dim=1)
        
        # score: (batchsize, negative + 1)
        score = torch.bmm(emb, context_vec.unsqueeze(2)).squeeze(2)

        # score: (batchsize, negative + 1)
        loss = -torch.mean(F.logsigmoid(score))

        return loss

    def sg(self, center, contexts, negatives):
        # center_emb: (batchsize, dim)
        # context_emb: (batchsize, context_len, dim)
        # negative_emb: (batchsize, negative, dim)
        center_emb = self.center_embed(center)
        context_emb = self.context_embed(contexts)
        negative_emb = self.center_embed(negatives)

        # emb: (batchsize, negative + 1, dim)
        emb = torch.cat((center_emb.unsqueeze(1), -negative_emb), dim=1)

        # score: (batchsize, negatives + 1, context_len)
        score = torch.bmm(emb, context_emb.transpose(1, 2))

        # score: (batchsize, negative + 1)
        loss = -torch.mean(F.logsigmoid(score))

        return loss
