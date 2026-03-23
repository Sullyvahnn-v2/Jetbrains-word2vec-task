from __future__ import annotations

import numpy as np

from word2vec_numpy.word2vec_numpy.config import Word2VecConfig
from ..vocabulary import Vocabulary
from ..utils import make_sigmoid_table
from .base import BaseTrainer


class CBOWNegativeSampling(BaseTrainer):
    """
    Continuous Bag of Words (CBOW) with Negative Sampling trainer.

    Predicts the center word from the average of the surrounding context words.
    Matches the exact gradient scaling quirks of the original C implementation,
    where the error is NOT divided by the number of context words before
    distributing the backpropagated updates.
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

        K = self.config.negative
        u_o = self.W_[center].copy()

        negatives = self.vocab.sample_negatives(K, self.rng, exclude=center)
        U_neg = self.W_[negatives]

        # Forward
        pos_dot = float(np.dot(h, u_o))
        neg_dots = U_neg @ h

        pos_score = float(self._sigmoid(pos_dot))
        neg_scores = self._sigmoid(neg_dots)

        # Loss
        pos_loss = -np.log(pos_score + 1e-7)
        neg_loss = -np.log(1.0 - neg_scores + 1e-7).sum()
        total_loss = float(pos_loss + neg_loss)

        # Gradients
        err_pos = pos_score - 1.0  # label=1
        err_neg = neg_scores       # label=0

        # Accumulated error for the hidden layer `h`
        neu1e = err_pos * u_o + (err_neg[:, None] * U_neg).sum(axis=0)

        # Update output matrix W_
        self.W_[center] -= lr * (err_pos * h)
        grad_U_neg = err_neg[:, None] * h[None, :]
        np.add.at(self.W_, negatives, -lr * grad_U_neg)

        # Update input matrix W
        np.add.at(self.W, contexts, -lr * neu1e)

        return total_loss
