# pytorch-word2vec

pytorch implementation of word2vec.

## Install

```
$ git clone https://github.com/jyori112/pytorch-word2vec
```

## Usage

### Train

```
$ python train.py DATA WORDSIM -o OUTPUT
```

### Evaluate

```
$ python evaluate.py OUTPUT WORDSIM
```

## Evaluation

Experiment is done on English corpus of Europarl, and evaluated with wordsim353.
Spearmanr score was `0.389`.
