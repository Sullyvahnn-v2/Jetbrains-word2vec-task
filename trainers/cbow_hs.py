from __future__ import annotations

import numpy as np

from ..config import Word2VecConfig
from ..vocabulary import Vocabulary
from ..utils import make_sigmoid_table
from .base import BaseTrainer


class CBOWHierarchicalSoftmax(BaseTrainer):
    """
    Continuous Bag of Words (CBOW) with Hierarchical Softmax trainer.

    Predicts the center word from the average of the surrounding context words,
    using the Huffman tree of the center word for O(log V) classification.
    """

    def __init__(
        self,
        W: np.ndarray,
        W_: np.ndarray,
        vocab: Vocabulary,
        config: Word2VecConfig,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(W, W_, vocab, config, rng)
        self._sigmoid = make_sigmoid_table(
            table_size=config.sigmoid_table_size,
            max_val=config.sigmoid_max,
        )

    def train_pair(self, center: int, contexts: np.ndarray, lr: float) -> float:
        cw = len(contexts)
        if cw == 0:
            return 0.0

        h = self.W[contexts].mean(axis=0)

        points = self.vocab.point[center]
        codes = self.vocab.code[center]

        if len(points) == 0:
            return 0.0

        U_path = self.W_[points]

        # Forward
        dots = np.dot(U_path, h)
        scores = self._sigmoid(dots)

        labels = 1 - codes

        # Loss
        loss_pos = np.where(labels == 1, -np.log(scores + 1e-7), 0.0)
        loss_neg = np.where(labels == 0, -np.log(1.0 - scores + 1e-7), 0.0)
        total_loss = float((loss_pos + loss_neg).sum())

        err = scores - labels

        # Accumulated error for the hidden layer `h` (neu1e in word2vec.c)
        neu1e = np.dot(err, U_path)

        # Update output nodes W_
        grad_U_path = err[:, None] * h[None, :]
        np.add.at(self.W_, points, -lr * grad_U_path)

        # Update input contexts W (directly propagating neu1e without cw division
        # to strictly mirror C Word2Vec gradient schedules).
        np.add.at(self.W, contexts, -lr * neu1e)

        return total_loss/len(scores)
